"""
**File:** ``read_info.py``
**Region:** ``ds_provider_tripletex_py_lib/read_info``

Read and field-projection information for Tripletex products.

Each :class:`~ds_provider_tripletex_py_lib.enums.TripletexProductName` has a
packaged ``assets/<product_name>/read/metadata.json`` declaring ``path``,
``pagination`` (``"date_window"`` if the endpoint has ``dateFrom``/``dateTo``,
else ``"offset"``), ``changed_since`` (use Tripletex's ``changedSince``
filter instead of ``versionDigest``/``If-None-Match``), a field selector
(``str``, or a one-key ``dict`` for a nested group, e.g. ``{"account": ["id"]}``),
and an optional ``explode_columns`` (fields that are a list of objects,
needing one row per element rather than in-place flattening). ``_lite``
products reuse a full product's path with fewer fields.
"""

import json
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any, cast

from ds_resource_plugin_py_lib.common.resource.errors import ValidationError

from .enums import OperationType, PaginationKind, TripletexProductName


@dataclass(frozen=True)
class ReadInfo:
    """
    Read information for a single Tripletex product.

    Attributes:
        path: API path segment appended to the linked service host, e.g. ``"ledger/posting"``.
        fields: Field selector list used to build the ``fields`` query parameter.
        changed_since: Whether this product supports Tripletex's ``changedSince``
            filter (replaces ``versionDigest``/``If-None-Match`` when ``True``).
        pagination: How ``read()`` requests are paginated for this product.
        explode_columns: Field names that are a list of objects and should be
            exploded into one row per element, e.g. ``["bankAccountPresentation"]``.
        stable_sort_field: The field ``sorting`` pins to guarantee new rows
            always append after what's already been seen, never shifting
            earlier offsets -- must be server-assigned and monotonic (e.g.
            an id), not a display-order field. Defaults to ``"id"`` when a
            packaged product's metadata doesn't set it -- if that product
            has no top-level ``id`` field, validation catches the mismatch
            and raises rather than silently sorting by a field that isn't
            requested. Set explicitly when the product's own id is nested,
            e.g. ``"account.id"`` for ``balance_sheet`` (a
            ``BalanceSheetAccount`` row has no top-level ``id``, only
            ``account.id``).
    """

    path: str
    fields: list[Any]
    changed_since: bool
    pagination: PaginationKind = PaginationKind.OFFSET
    explode_columns: list[str] = field(default_factory=list)
    stable_sort_field: str | None = None


def _load_metadata(product_name: TripletexProductName, operation: OperationType) -> dict[str, Any]:
    """
    Read and parse one product's packaged operation metadata.

    A plain, mechanical file read -- whether a missing file is an error is
    left to the caller to decide.

    Args:
        product_name: Tripletex product whose assets to load.
        operation: Operation whose metadata file to load.

    Returns:
        dict[str, Any]: Parsed JSON payload.

    Raises:
        FileNotFoundError: If the product has no packaged assets for that operation.
    """
    raw = (
        files("ds_provider_tripletex_py_lib")
        .joinpath("assets", product_name.value, operation.value, "metadata.json")
        .read_text(encoding="utf-8")
    )
    return cast("dict[str, Any]", json.loads(raw))


def get_read_info(product_name: TripletexProductName) -> ReadInfo:
    """
    Load packaged read metadata for a Tripletex product.

    Every :class:`TripletexProductName` member must ship read assets, so a
    missing file is treated the same as a malformed one -- a packaging bug,
    not a condition to degrade gracefully from.

    Args:
        product_name: Tripletex product to load.

    Returns:
        ReadInfo: Read info built from ``assets/<product_name>/read/metadata.json``.

    Raises:
        ValidationError: If the metadata file is missing, isn't valid JSON,
            or exists but is missing a required key (``path``, ``fields``,
            ``pagination``, ``changed_since``) or ``pagination`` is not a
            recognized :class:`PaginationKind` value.
    """
    try:
        payload = _load_metadata(product_name, OperationType.READ)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValidationError(
            message=f"Missing or malformed read metadata for Tripletex product '{product_name.value}': {exc}",
            details={"product_name": product_name.value, "operation": OperationType.READ.value},
        ) from exc
    try:
        return ReadInfo(
            path=payload["path"],
            fields=payload["fields"],
            pagination=PaginationKind(payload["pagination"]),
            changed_since=payload["changed_since"],
            explode_columns=payload.get("explode_columns", []),
            stable_sort_field=payload.get("stable_sort_field", "id"),
        )
    except (KeyError, ValueError) as exc:
        raise ValidationError(
            message=f"Invalid read metadata for Tripletex product '{product_name.value}': {exc}",
            details={"product_name": product_name.value, "operation": OperationType.READ.value},
        ) from exc
