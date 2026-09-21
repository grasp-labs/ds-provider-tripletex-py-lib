"""
**File:** ``tripletex.py``
**Region:** ``ds_provider_tripletex_py_lib/dataset/tripletex``

Read-only dataset for the Tripletex v2 API.

Products are read via ``from``/``count`` offset pagination, or rolling
monthly date windows for ``ledger/posting``/``balance_sheet`` (see
``read_info.py``). Most products cache via ``versionDigest``/``If-None-Match``;
a few (``customer``, ``supplier``, their ``_lite`` variants) instead use
Tripletex's ``changedSince`` filter. All write methods raise
``NotSupportedError``.

Example:
    >>> dataset = TripletexDataset(
    ...     settings=TripletexDatasetSettings(product_name=TripletexProductName.CUSTOMER),
    ...     linked_service=TripletexLinkedService(
    ...         settings=TripletexLinkedServiceSettings(
    ...             consumer_token="my-consumer-token",
    ...             employee_token="my-employee-token",
    ...         ),
    ...     ),
    ... )
    >>> dataset.linked_service.connect()
    >>> dataset.read()
    >>> data = dataset.output
"""

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Generic, NoReturn, TypeVar, cast

import pandas as pd
from ds_common_logger_py_lib import Logger
from ds_common_serde_py_lib import Serializable
from ds_resource_plugin_py_lib.common.resource.dataset import DatasetSettings, TabularDataset
from ds_resource_plugin_py_lib.common.resource.dataset.errors import ReadError
from ds_resource_plugin_py_lib.common.resource.dataset.storage_format import DatasetStorageFormatType
from ds_resource_plugin_py_lib.common.resource.errors import NotSupportedError, ResourceException, ValidationError
from ds_resource_plugin_py_lib.common.resource.linked_service.errors import (
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
)
from ds_resource_plugin_py_lib.common.serde.deserialize import PandasDeserializer
from ds_resource_plugin_py_lib.common.serde.serialize import PandasSerializer

from ..enums import PaginationKind, ResourceType, TripletexProductName
from ..linked_service.tripletex import TripletexLinkedService
from ..read_info import ReadInfo, get_read_info

logger = Logger.get_logger(__name__, package=True)

# Every query param name this module ever generates itself -- from/count/fields on
# every request, changedSince on the changedSince loop, dateFrom/dateTo on date-windowed
# reads, sorting on digest-paginated reads (see _fetch_pages_by_digest). A caller's
# settings.read.params must never be able to override any of these.
_RESERVED_QUERY_PARAMS = frozenset({"from", "count", "fields", "changedSince", "dateFrom", "dateTo", "sorting"})


@dataclass(kw_only=True)
class TripletexReadSettings(Serializable):
    """
    Settings for Tripletex read operations.
    """

    count: int = 5000
    """Page size for ``from``/``count`` offset pagination."""

    fields: list[Any] | None = None
    """Field selector. Required with ``path``; ignored with ``product_name``.

    Each item is a field name (``str``), or a one-key dict expanding a nested
    object/sub-collection, e.g. ``{"account": ["id"]}``.

    Example:
        >>> TripletexReadSettings(path="custom/thing", fields=["id", "name", {"account": ["id"]}])
        # -> fields=id,name,account(id)
    """

    params: dict[str, Any] | None = None
    """Additional query parameters merged into every request (e.g. ``isInactive``).

    Not for ``from``/``count``/``fields``/``changedSince``/``dateFrom``/
    ``dateTo``/``sorting`` -- this module generates all of those itself
    per request, and setting any of them here raises ``ReadError`` instead
    of being silently overridden.
    """

    path: str | None = None
    """Custom API path, for reading an endpoint not in the packaged product catalog.

    Set together with ``fields`` and ``pagination`` instead of
    ``TripletexDatasetSettings.product_name``. Exactly one of
    ``product_name``/``path`` may be set -- both raises ``ReadError``.

    Example:
        >>> TripletexReadSettings(path="custom/thing", fields=["id"], pagination=PaginationKind.OFFSET)
    """

    pagination: PaginationKind | None = None
    """Required when ``path`` is set. Ignored (must be unset) when ``product_name`` is set."""

    changed_since: bool | None = None
    """Whether ``path`` supports Tripletex's ``changedSince`` filter.

    Required when ``path`` is set -- no safe default, since silently
    assuming ``False`` could mask never having checked. Ignored (must be
    unset) when ``product_name`` is set (fixed by its own metadata).
    """

    date_from: str | None = None
    """Start of a date-windowed read (e.g. ``ledger_posting``), in years back from today.

    A relative year offset, e.g. ``"3"`` or ``"-3"`` (sign doesn't matter).
    Defaults to 3 years back when unset. Only valid for
    :attr:`PaginationKind.DATE_WINDOW` reads -- otherwise raises ``ReadError``.

    Example:
        >>> TripletexReadSettings(date_from="3")
    """

    explode_columns: list[str] | None = None
    """Field names that are a list of objects, exploded into one row per element.

    Only meaningful with ``path`` -- ignored (must be unset) when
    ``product_name`` is set, since a packaged product's explode columns are
    fixed by its own metadata. See ``read_info.py`` for why this differs
    from a plain nested object (which flattens in place, no explode needed).
    """


@dataclass(kw_only=True)
class TripletexDatasetSettings(DatasetSettings):
    """
    Settings for the Tripletex dataset.
    """

    product_name: TripletexProductName | None = None
    """Tripletex product to read. Exactly one of ``product_name``/``read.path`` may be set."""

    read: TripletexReadSettings = field(default_factory=TripletexReadSettings)
    """Settings for ``read()``."""


TripletexDatasetSettingsType = TypeVar(
    "TripletexDatasetSettingsType",
    bound=TripletexDatasetSettings,
)
TripletexLinkedServiceType = TypeVar(
    "TripletexLinkedServiceType",
    bound=TripletexLinkedService[Any],
)


@dataclass(kw_only=True)
class _ReadContext:
    """
    Mutable, per-call state threaded through one ``read()`` call's pagination.

    ``watermark_digests``/``watermark_changed_since``/``resume_offsets``/``resume_digests``/
    ``records`` are local, mutable copies the readers write into --
    ``self.checkpoint``/``self.output`` aren't updated until the reader returns.
    """

    url: str
    fields_param: str
    watermark_digests: dict[str, str]
    watermark_changed_since: dict[str, str]
    resume_offsets: dict[str, int]
    resume_digests: dict[str, str]
    records: list[dict[str, Any]]


@dataclass(kw_only=True)
class TripletexDataset(
    TabularDataset[TripletexLinkedServiceType, TripletexDatasetSettingsType, PandasSerializer, PandasDeserializer],
    Generic[TripletexLinkedServiceType, TripletexDatasetSettingsType],
):
    """
    Read-only tabular dataset for Tripletex products.

    Exposed (caller-configurable): ``id``, ``name``, ``description``,
    ``version``, ``settings``, ``linked_service``, ``serializer``,
    ``deserializer``, ``checkpoint``. ``input``/``output`` (inherited from
    ``TabularDataset``) are excluded from serialization -- they hold runtime
    read results, not configuration -- though still constructible, matching
    the base class. No field on this class is fully internal (``init=False``).
    """

    linked_service: TripletexLinkedServiceType
    settings: TripletexDatasetSettingsType

    def __post_init__(self) -> None:
        """Fill in a default serializer/deserializer if the caller didn't provide one."""
        if self.serializer is None:
            self.serializer = PandasSerializer(format=DatasetStorageFormatType.JSON, kwargs={"orient": "records"})
        if self.deserializer is None:
            self.deserializer = PandasDeserializer(format=DatasetStorageFormatType.SEMI_STRUCTURED_JSON, kwargs={"sep": "_"})

    @property
    def type(self) -> ResourceType:
        """Return the dataset resource type."""
        return ResourceType.DATASET

    @property
    def supports_checkpoint(self) -> bool:
        """
        Whether this dataset supports incremental loads via ``self.checkpoint``.

        Checkpoint holds ``{"watermark_digests": {...}, "watermark_changed_since": {...},
        "resume_offsets": {...}, "resume_digests": {...}}`` -- stored
        digest/changed_since_watermark values per request, plus (digest-cached products
        only) a resumable ``from`` offset and its digest, left behind by a
        failed run. An empty checkpoint (``{}``) means a full load.

        Returns:
            bool: Always ``True``.
        """
        return True

    def _resolve_reader(self, read_info: ReadInfo) -> Callable[[_ReadContext], None]:
        """
        Return the read implementation for the resolved endpoint.

        Args:
            read_info: The resolved endpoint to read.

        Returns:
            Callable[[_ReadContext], None]: ``_read_date_windowed`` for
            ``DATE_WINDOW`` pagination; otherwise
            ``_read_paginated_by_changed_since`` for products supporting
            Tripletex's ``changedSince`` filter, or ``_read_paginated_by_digest``
            (``versionDigest``/``If-None-Match``) for every other product.
        """
        if read_info.pagination == PaginationKind.DATE_WINDOW:
            return self._read_date_windowed
        if read_info.changed_since:
            return self._read_paginated_by_changed_since
        return self._read_paginated_by_digest

    def _resolve_product_name(self, product_name: str) -> TripletexProductName:
        """
        Resolve a raw ``settings.product_name`` value into a ``TripletexProductName``.

        Args:
            product_name: The raw product name to validate.

        Returns:
            TripletexProductName: The resolved product.

        Raises:
            ReadError: If not a recognized :class:`TripletexProductName` value.
        """
        try:
            return TripletexProductName(product_name)
        except ValueError as exc:
            raise ReadError(
                message=f"Unknown Tripletex product: {product_name!r}",
                details={"type": self.type.value, "product_name": product_name},
            ) from exc

    def _resolve_packaged_read_info(self, product_name: str) -> ReadInfo:
        """
        Resolve read info for a packaged product, wrapping metadata failures as ``ReadError``.

        Args:
            product_name: The raw ``settings.product_name`` value to resolve.

        Returns:
            ReadInfo: The packaged product's read info.

        Raises:
            ReadError: If not a recognized product, or its packaged read
                metadata is missing or malformed.
        """
        logger.info("Reading Tripletex product: %s", product_name)
        resolved_product_name = self._resolve_product_name(product_name)
        try:
            return get_read_info(resolved_product_name)
        except ValidationError as exc:
            raise ReadError(
                message=exc.message,
                details={**exc.details, "type": self.type.value},
            ) from exc

    def _resolve_read_info(self) -> ReadInfo:
        """
        Resolve the endpoint to read: a packaged product, or a custom path.

        ``settings.product_name`` is resolved via the packaged asset catalog
        if set; otherwise ``settings.read.path``/``fields``/``pagination``
        are used. See ``_validate_read_settings`` for the combination rules.

        Returns:
            ReadInfo: The fully resolved endpoint to read.

        Raises:
            ReadError: If the settings combination is invalid, or the
                resolved product/metadata is unrecognized/malformed.
        """
        if self.settings.product_name:
            read_info = self._resolve_packaged_read_info(self.settings.product_name)
        else:
            read_info = self._resolve_read_info_from_settings()

        self._validate_read_settings(read_info)
        return read_info

    def _resolve_read_info_from_settings(self) -> ReadInfo:
        """
        Build a ``ReadInfo`` directly from ``settings.read``, for a custom (non-packaged) endpoint.

        Only called when ``settings.product_name`` is unset. Raises
        immediately on a missing value, so every field passed to
        ``ReadInfo`` is already guaranteed present.

        Returns:
            ReadInfo: The endpoint built from ``settings.read``.

        Raises:
            ReadError: If ``path``/``fields``/``pagination``/``changed_since``
                aren't all set alongside ``product_name`` being unset.
        """
        if not self.settings.read.path:
            raise ReadError(
                message="Either settings.product_name or settings.read.path must be specified.",
                details={"type": self.type.value},
            )
        if not self.settings.read.fields:
            raise ReadError(
                message="settings.read.fields must be specified when settings.read.path is set.",
                details={"type": self.type.value, "path": self.settings.read.path},
            )
        if not self.settings.read.pagination:
            raise ReadError(
                message="settings.read.pagination must be specified when settings.read.path is set.",
                details={"type": self.type.value, "path": self.settings.read.path},
            )
        if self.settings.read.changed_since is None:
            raise ReadError(
                message="settings.read.changed_since must be specified when settings.read.path is set.",
                details={"type": self.type.value, "path": self.settings.read.path},
            )
        return ReadInfo(
            path=self.settings.read.path,
            fields=self.settings.read.fields,
            pagination=self.settings.read.pagination,
            changed_since=self.settings.read.changed_since,
            explode_columns=self.settings.read.explode_columns or [],
        )

    def _validate_read_settings(self, read_info: ReadInfo) -> None:
        """
        Validate ``settings.product_name``/``settings.read`` against the resolved ``read_info``.

        Args:
            read_info: The resolved endpoint to read.

        Raises:
            ReadError: If any ``read.*`` setting is set alongside
                ``product_name`` (ignored there), ``read.date_from`` is set
                but pagination isn't :attr:`PaginationKind.DATE_WINDOW`,
                ``changed_since`` is combined with date-windowed pagination
                (``_read_date_windowed`` never emits ``changedSince``), or
                ``read.params`` sets a reserved key (``_RESERVED_QUERY_PARAMS``)
                that this module generates internally per request.
        """
        if self.settings.product_name and (
            self.settings.read.path
            or self.settings.read.fields
            or self.settings.read.pagination
            or self.settings.read.changed_since is not None
            or self.settings.read.explode_columns
        ):
            raise ReadError(
                message=(
                    "settings.read.path, settings.read.fields, settings.read.pagination, "
                    "settings.read.changed_since, and settings.read.explode_columns are "
                    "ignored when product_name is set -- remove them, or use read.path "
                    "instead of product_name."
                ),
                details={"type": self.type.value, "product_name": str(self.settings.product_name)},
            )

        if self.settings.read.date_from is not None and read_info.pagination is not PaginationKind.DATE_WINDOW:
            raise ReadError(
                message="settings.read.date_from is only meaningful for date-windowed pagination.",
                details={
                    "type": self.type.value,
                    "date_from": self.settings.read.date_from,
                    "pagination": read_info.pagination.value,
                },
            )

        if read_info.pagination is PaginationKind.DATE_WINDOW and read_info.changed_since:
            raise ReadError(
                message=(
                    "changed_since is not supported for date-windowed pagination -- "
                    "_read_date_windowed never emits changedSince, so this combination "
                    "would silently do nothing."
                ),
                details={"type": self.type.value, "path": read_info.path},
            )

        reserved_params_used = _RESERVED_QUERY_PARAMS & (self.settings.read.params or {}).keys()
        if reserved_params_used:
            raise ReadError(
                message=(
                    f"settings.read.params must not set {sorted(reserved_params_used)} -- these are "
                    "generated internally for every request and would be silently overridden if "
                    "allowed through params. Remove them from settings.read.params."
                ),
                details={"type": self.type.value, "reserved_params": sorted(reserved_params_used)},
            )

    def read(self) -> None:
        """
        Read all rows for the configured product and assign them to ``self.output``.

        See ``_resolve_read_info`` for how the endpoint is resolved.

        Raises:
            AuthenticationError: If authentication fails.
            AuthorizationError: If authorization fails.
            ConnectionError: If the transport cannot reach Tripletex.
            ReadError: If neither a known product nor a custom path is
                configured, or the read fails.
        """
        read_info = self._resolve_read_info()
        fields_param = _build_fields_param(read_info.fields)
        self._execute_read(read_info, fields_param)

    def _execute_read(self, read_info: ReadInfo, fields_param: str) -> None:
        """
        Run the resolved read against Tripletex and populate ``self.output``/``self.checkpoint``.

        Args:
            read_info: The endpoint to read, resolved by ``_resolve_read_info``.
            fields_param: The rendered ``fields`` query parameter value.

        Raises:
            AuthenticationError: If authentication fails.
            AuthorizationError: If authorization fails.
            ConnectionError: If the transport cannot reach Tripletex.
            ReadError: If the read fails for any other reason.
        """
        init_watermark_digests: dict[str, str] = dict(self.checkpoint.get("watermark_digests", {})) if self.checkpoint else {}
        init_watermark_changed_since: dict[str, str] = (
            dict(self.checkpoint.get("watermark_changed_since", {})) if self.checkpoint else {}
        )
        init_resume_offsets: dict[str, int] = dict(self.checkpoint.get("resume_offsets", {})) if self.checkpoint else {}
        init_resume_digests: dict[str, str] = dict(self.checkpoint.get("resume_digests", {})) if self.checkpoint else {}
        records: list[dict[str, Any]] = []
        ctx = _ReadContext(
            url=f"{self.linked_service.settings.host}/{read_info.path}",
            fields_param=fields_param,
            watermark_digests=dict(init_watermark_digests),
            watermark_changed_since=dict(init_watermark_changed_since),
            resume_offsets=dict(init_resume_offsets),
            resume_digests=dict(init_resume_digests),
            records=records,
        )

        reader = self._resolve_reader(read_info)
        try:
            reader(ctx)
        except (AuthenticationError, AuthorizationError, ConnectionError):
            # These are ResourceException subclasses too, so without this clause
            # first, the broader `except ResourceException` below would catch
            # them and reclassify them as ReadError. Re-raising here lets them
            # propagate as themselves instead.
            self.checkpoint = {
                "watermark_digests": init_watermark_digests,
                "watermark_changed_since": init_watermark_changed_since,
                "resume_offsets": ctx.resume_offsets,
                "resume_digests": ctx.resume_digests,
            }
            raise
        except ResourceException as exc:
            self.checkpoint = {
                "watermark_digests": init_watermark_digests,
                "watermark_changed_since": init_watermark_changed_since,
                "resume_offsets": ctx.resume_offsets,
                "resume_digests": ctx.resume_digests,
            }
            product_name = getattr(self.settings.product_name, "value", self.settings.product_name)
            raise ReadError(
                message=exc.message,
                status_code=exc.status_code,
                details={**exc.details, "type": self.type.value, "product_name": product_name, "path": read_info.path},
            ) from exc
        finally:
            deserializer = cast("PandasDeserializer", self.deserializer)
            output = deserializer(records)
            for column in read_info.explode_columns:
                output = _explode_column(output, column)
            self.output = output

        self.checkpoint = {
            "watermark_digests": ctx.watermark_digests,
            "watermark_changed_since": ctx.watermark_changed_since,
            "resume_offsets": {},
            "resume_digests": {},
        }

    def _read_paginated_by_changed_since(self, ctx: _ReadContext) -> None:
        """
        Fetch every page of a ``changedSince``-capable product into ``ctx.records``.

        Tripletex's ``changedSince`` param (format ``YYYY-MM-DDThh:mm:ssZ``)
        filters server-side to only rows changed since that timestamp. The
        next run's changed_since_watermark is captured as this run's start time and only
        stored in ``ctx.watermark_changed_since`` once every page succeeds.

        Always starts at ``offset=0`` -- not resumable. ``changedSince``
        filters a mutable, moving result set, so a stored offset can't be
        trusted: a row added/removed/updated between attempts can shift
        every later row's position, risking a silent skip or duplicate.
        """
        extra_params = self.settings.read.params
        request_key = _request_key(ctx.fields_param, extra_params=extra_params)
        changed_since_value = ctx.watermark_changed_since.get(request_key)
        run_started_at = datetime.now(tz=timezone.utc)
        count = self.settings.read.count
        offset = 0

        while True:
            params: dict[str, Any] = dict(extra_params) if extra_params else {}
            params.update({"from": offset, "count": count, "fields": ctx.fields_param, "sorting": "id"})
            if changed_since_value is not None:
                params["changedSince"] = changed_since_value

            response = self.linked_service.connection.get(url=ctx.url, params=params)
            body = response.json()
            values = body.get("values", [])
            if not values:
                break

            ctx.records.extend(values)
            offset += len(values)
        ctx.watermark_changed_since[request_key] = run_started_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _read_paginated_by_digest(self, ctx: _ReadContext) -> None:
        """
        Fetch every page of an offset-paginated product using ``versionDigest``/``If-None-Match``.

        See ``_fetch_pages_by_digest`` for the mechanism.
        """
        request_key = _request_key(ctx.fields_param, extra_params=self.settings.read.params)
        self._fetch_pages_by_digest(ctx, request_key=request_key)

    def _fetch_pages_by_digest(
        self,
        ctx: _ReadContext,
        *,
        request_key: str,
        extra_query_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Page through ``ctx.url`` via ``versionDigest``/``If-None-Match``, caching under ``request_key``.

        Exits on ``304`` (unchanged); falls back to ``"magic-value"`` since
        ``versionDigest`` can be JSON ``null``. Resumes from
        ``ctx.resume_offsets`` if a prior run failed mid-pagination; cleared
        once this key's pass completes.

        Pins ``sorting=id`` -- IDs are server-assigned and increasing, so
        new rows always append after what's already been seen, never
        shifting earlier offsets. A resumed offset is trusted only if the
        digest recorded when it was saved still matches the first resumed
        response; a mismatch restarts this key from ``offset=0``.

        Args:
            ctx: Read context to append ``records`` into and cache
                ``watermark_digests``/``resume_offsets``/``resume_digests`` on.
            request_key: Cache key identifying this request shape.
            extra_query_params: Extra query params beyond ``from``/``count``/``fields``.
        """
        extra_params = self.settings.read.params
        if_none_match = ctx.watermark_digests.get(request_key, "magic-value")
        count = self.settings.read.count
        offset = ctx.resume_offsets.get(request_key, 0)
        expected_resume_digest = ctx.resume_digests.get(request_key)
        should_verify_resume = offset != 0
        new_digest: str | None = None

        while True:
            params: dict[str, Any] = dict(extra_params) if extra_params else {}
            if extra_query_params:
                params.update(extra_query_params)
            params.update({"from": offset, "count": count, "fields": ctx.fields_param, "sorting": "id"})

            response = self.linked_service.connection.get(url=ctx.url, params=params, headers={"If-None-Match": if_none_match})
            if response.status_code == 304:  # Nothing changed since the last full pass
                break

            body = response.json()
            if new_digest is None:  # Capture scope wide digest, from the first page only
                new_digest = body.get("versionDigest") or "magic-value"

            if should_verify_resume:
                should_verify_resume = False
                if new_digest != expected_resume_digest:
                    # Digest differs from what was recorded when that run was
                    # interrupted -- can't be trusted. Discard this response
                    # and restart from scratch.
                    offset = 0
                    new_digest = None
                    continue

            values = body.get("values", [])
            if not values:
                break

            ctx.records.extend(values)
            offset += len(values)
            ctx.resume_offsets[request_key] = offset
            ctx.resume_digests[request_key] = new_digest

        ctx.resume_offsets.pop(request_key, None)
        ctx.resume_digests.pop(request_key, None)
        if new_digest is not None:
            ctx.watermark_digests[request_key] = new_digest

    def _resolve_date_from(self, *, date_to: date) -> date:
        """
        Resolve ``settings.read.date_from`` into an absolute start date.

        A relative year offset from ``date_to`` (e.g. ``"3"``/``"-3"``, sign
        ignored). Defaults to 3 years back when unset.

        Args:
            date_to: The read window's end date (today).

        Returns:
            date: The resolved start date.

        Raises:
            ReadError: If ``settings.read.date_from`` is not a valid integer.
        """
        raw = self.settings.read.date_from
        if raw is None:
            return _add_years(date_to, -3)

        try:
            years = int(raw)
        except ValueError as exc:
            raise ReadError(
                message=f"settings.read.date_from is not a valid relative year-offset: {raw!r}",
                details={"type": self.type.value, "date_from": raw},
            ) from exc

        return _add_years(date_to, -abs(years))

    def _read_date_windowed(self, ctx: _ReadContext) -> None:
        """
        Fetch a date-windowed product (``ledger/posting``) across rolling monthly windows.

        Tripletex requires a bounded ``dateFrom``/``dateTo`` per request, so
        this walks the read window in monthly steps, offset-paginating and
        digest-caching within each. ``ctx.watermark_changed_since`` is never touched here
        -- but ``ctx`` is still accepted so the calling convention matches
        ``_resolve_reader``'s other two readers.

        Raises:
            ReadError: If ``settings.read.date_from`` is not a valid integer.
        """
        today = datetime.now(tz=timezone.utc).date()
        date_from = self._resolve_date_from(date_to=today)
        date_to = today + timedelta(days=1)  # Tripletex's dateTo is exclusive

        for period_from, period_to in _period_generator(date_from=date_from, date_to=date_to):
            period_from_str = period_from.isoformat()
            period_to_str = period_to.isoformat()
            request_key = _request_key(
                ctx.fields_param,
                date_from=period_from_str,
                date_to=period_to_str,
                extra_params=self.settings.read.params,
            )
            self._fetch_pages_by_digest(
                ctx,
                request_key=request_key,
                extra_query_params={"dateFrom": period_from_str, "dateTo": period_to_str},
            )

    def create(self) -> NoReturn:
        """Create is not supported by the Tripletex provider."""
        raise NotSupportedError("Create operation is not supported for Tripletex datasets")

    def update(self) -> NoReturn:
        """Update is not supported by the Tripletex provider."""
        raise NotSupportedError("Update operation is not supported for Tripletex datasets")

    def delete(self) -> NoReturn:
        """Delete is not supported by the Tripletex provider."""
        raise NotSupportedError("Delete operation is not supported for Tripletex datasets")

    def upsert(self) -> NoReturn:
        """Upsert is not supported by the Tripletex provider."""
        raise NotSupportedError("Upsert operation is not supported for Tripletex datasets")

    def rename(self) -> NoReturn:
        """Rename is not supported by the Tripletex provider."""
        raise NotSupportedError("Rename operation is not supported for Tripletex datasets")

    def purge(self) -> NoReturn:
        """Purge is not supported by the Tripletex provider."""
        raise NotSupportedError("Purge operation is not supported for Tripletex datasets")

    def list(self) -> NoReturn:
        """List is not supported by the Tripletex provider."""
        raise NotSupportedError("List operation is not supported for Tripletex datasets")

    def close(self) -> None:
        """Close the linked-service connection."""
        self.linked_service.close()


def _explode_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Explode a list-of-objects column (e.g. ``bankAccountPresentation``) into one row per element.

    Args:
        df: DataFrame with ``column`` still holding raw list values.
        column: Name of the column to explode.

    Returns:
        pd.DataFrame: ``df`` with ``column`` replaced by one row per element.

    Raises:
        ReadError: If ``column`` is missing from a non-empty ``df`` (likely
            a packaging bug), or a non-null value in ``column`` isn't a
            ``list``. A fully empty ``df`` is a no-op instead.
    """
    if df.empty:
        return df
    if column not in df.columns:
        raise ReadError(
            message=(
                f"explode_columns names {column!r}, but it isn't a field in the read "
                "result -- check it matches an entry in the field selector."
            ),
            details={"column": column, "available_columns": list(df.columns)},
        )
    non_null = df[column].dropna()
    not_a_list = non_null[~non_null.apply(lambda value: isinstance(value, list))]
    if not not_a_list.empty:
        raise ReadError(
            message=(
                f"explode_columns names {column!r}, but not every value is a list -- "
                f"check the field actually returns a list of objects for every row "
                f"(found {not_a_list.iloc[0]!r})."
            ),
            details={"column": column},
        )
    df = df[df[column].notna()]
    df = df[df[column].apply(len) != 0]
    df = df.explode(column).reset_index(drop=True)
    normalized = pd.json_normalize(df[column]).add_prefix(f"{column}_")
    return df.drop(columns=[column]).join(normalized)


def _build_fields_param(fields: list[Any]) -> str:
    """
    Build a Tripletex ``fields`` query parameter value from a field selector list.

    Each item is a field name (``str``), or a one-key dict -- ``{"group": [...]}``
    -- expanding to ``group(...)``.

    Args:
        fields: Field selector list, e.g. ``["id", {"account": ["id"]}]``.

    Returns:
        str: Comma-separated Tripletex field expression, e.g. ``"id,account(id)"``.
    """
    parts: list[str] = []
    for item in fields:
        if isinstance(item, str):
            parts.append(item)
        else:
            for key, children in item.items():
                parts.append(f"{key}({_build_fields_param(children)})")
    return ",".join(parts)


def _request_key(
    fields_param: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> str:
    """
    Derive a stable cache key for the ``versionDigest``/``changedSince`` checkpoint.

    Two requests share a cache entry only if they'd return the same
    records: different ``fields``, ``params``, or date window each get
    their own key, but different pages (``from``/``count``) of the same
    query share one.

    Returns:
        str: A SHA-256 hex digest identifying this request shape.
    """
    payload: dict[str, Any] = {"fields": fields_param}
    if date_from is not None:
        payload["dateFrom"] = date_from
        payload["dateTo"] = date_to
    if extra_params:
        payload["extra_params"] = extra_params
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _add_years(reference: date, years: int) -> date:
    """
    Shift ``reference`` by a number of calendar years, positive or negative.

    Falls back to February 28th when shifting a leap day (Feb 29th) into a
    non-leap year.

    Args:
        reference: The date to shift.
        years: Number of years to shift by; negative shifts into the past.

    Returns:
        date: ``reference`` shifted by ``years`` calendar years.
    """
    try:
        return reference.replace(year=reference.year + years)
    except ValueError:
        return reference.replace(month=2, day=28, year=reference.year + years)


def _period_generator(*, date_from: date, date_to: date) -> Iterator[tuple[date, date]]:
    """
    Generate monthly periods used to page the ``ledger/posting`` endpoint.

    Args:
        date_from: Inclusive date to start generating periods from.
        date_to: Exclusive date to stop generating periods at.

    Yields:
        tuple[date, date]: Each period as ``(period_from, period_to)``, where
        ``period_from`` is inclusive and ``period_to`` is exclusive.
        Callers wanting to include a specific day (e.g.
        today) must pass a ``date_to`` one day past it -- see
        ``_read_date_windowed``.
    """
    while date_from < date_to:
        next_month = date_from.replace(day=28) + timedelta(days=4)
        period_from = date_from.replace(day=1)
        period_to = min(next_month.replace(day=1), date_to)
        yield period_from, period_to
        date_from = next_month.replace(day=1)
