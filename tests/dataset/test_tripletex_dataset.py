"""
**File:** ``test_tripletex_dataset.py``
**Region:** ``tests/dataset``

Unit tests for TripletexDataset.
"""

from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd
import pytest
from ds_resource_plugin_py_lib.common.resource.dataset.errors import ReadError
from ds_resource_plugin_py_lib.common.resource.errors import NotSupportedError, ResourceException, ValidationError
from ds_resource_plugin_py_lib.common.resource.linked_service.errors import AuthenticationError, ConnectionError

import ds_provider_tripletex_py_lib.dataset.tripletex as tripletex_mod
from ds_provider_tripletex_py_lib.dataset.tripletex import (
    TripletexDataset,
    TripletexDatasetSettings,
    TripletexReadSettings,
    _add_years,
    _build_fields_param,
    _period_generator,
    _request_key,
)
from ds_provider_tripletex_py_lib.enums import PaginationKind, ResourceType, TripletexProductName
from ds_provider_tripletex_py_lib.linked_service.tripletex import (
    TripletexLinkedService,
    TripletexLinkedServiceSettings,
)
from ds_provider_tripletex_py_lib.read_info import get_read_info


class FakeResponse:
    """Mock HTTP response."""

    def __init__(self, json_data, status_code: int = 200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json


class FakeSession:
    """Mock ``Http``-like session that returns predefined responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def get(self, url, params=None, headers=None):
        self.requests.append({"url": url, "params": params, "headers": headers})
        if not self.responses:
            raise ConnectionError("No more mock responses available in FakeSession")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class DummyTripletexLinkedService(TripletexLinkedService):
    """Linked service with a session injected directly, bypassing connect()."""

    def __init__(self, settings, session):
        super().__init__(settings=settings, id=uuid4(), name="dummy", version="1.0.0")
        self._session = session


def make_linked_service(responses) -> DummyTripletexLinkedService:
    """Create a linked service with a mocked session."""
    settings = TripletexLinkedServiceSettings(consumer_token="c", employee_token="e")
    return DummyTripletexLinkedService(settings=settings, session=FakeSession(responses))


def make_dataset(
    responses,
    product_name: TripletexProductName | None = TripletexProductName.CUSTOMER,
    checkpoint=None,
    read: TripletexReadSettings | None = None,
) -> TripletexDataset:
    """Create a dataset with a mocked linked service."""
    linked_service = make_linked_service(responses)
    settings = TripletexDatasetSettings(product_name=product_name, read=read or TripletexReadSettings(count=100))
    dataset = TripletexDataset(
        id=uuid4(),
        name="test_dataset",
        version="1.0.0",
        linked_service=linked_service,
        settings=settings,
    )
    if checkpoint is not None:
        dataset.checkpoint = checkpoint
    return dataset


# -----------------------------------------------------------------------------
# Contract: schema/dataclass alignment -- a representative payload must
# deserialize successfully (DATASET_CONTRACT.md, "Construction guarantee")
# -----------------------------------------------------------------------------


def test_deserialize_representative_payload():
    """A representative schema-valid payload constructs successfully via deserialize()."""
    payload = {
        "id": str(uuid4()),
        "name": "tripletex",
        "version": "1.0.0",
        "settings": {"product_name": "customer"},
        "linked_service": {
            "id": str(uuid4()),
            "name": "tripletex",
            "version": "1.0.0",
            "settings": {"consumer_token": "my-consumer-token", "employee_token": "my-employee-token"},
        },
    }

    dataset = TripletexDataset.deserialize(payload)

    assert dataset.settings.product_name is TripletexProductName.CUSTOMER
    assert isinstance(dataset.linked_service, TripletexLinkedService)
    assert dataset.linked_service.settings.consumer_token == "my-consumer-token"


# -----------------------------------------------------------------------------
# Contract: dataset type and checkpoint support
# -----------------------------------------------------------------------------


def test_type_property():
    """type property returns the Tripletex dataset resource type."""
    dataset = make_dataset([])
    assert dataset.type == ResourceType.DATASET


def test_supports_checkpoint_returns_true():
    """supports_checkpoint property must return True."""
    dataset = make_dataset([])
    assert dataset.supports_checkpoint is True


# -----------------------------------------------------------------------------
# serializer/deserializer defaults and caller override
# -----------------------------------------------------------------------------


def test_serializer_and_deserializer_default_when_unset():
    """Omitting serializer/deserializer fills in a default JSON-records/flattening pair."""
    dataset = make_dataset([])
    assert isinstance(dataset.serializer, tripletex_mod.PandasSerializer)
    assert isinstance(dataset.deserializer, tripletex_mod.PandasDeserializer)


def test_deserializer_default_flattens_nested_fields_with_underscore():
    """The default deserializer flattens nested fields into underscore-joined columns."""
    responses = [
        FakeResponse({"values": [{"id": 1, "account": {"id": 5, "number": 1500}}], "versionDigest": "d1"}),
        FakeResponse({"values": [], "versionDigest": "d1"}),
    ]
    dataset = make_dataset(responses)

    dataset.read()

    assert list(dataset.output.columns) == ["id", "account_id", "account_number"]
    assert dataset.output.iloc[0]["account_id"] == 5
    assert dataset.output.iloc[0]["account_number"] == 1500


# -----------------------------------------------------------------------------
# explode_columns behavior (supplier_bank_accounts_lite)
# -----------------------------------------------------------------------------


def test_read_explodes_list_of_objects_column():
    """A list-of-objects field (bankAccountPresentation) explodes into one row per element."""
    responses = [
        FakeResponse(
            {
                "values": [
                    {
                        "id": 1,
                        "isInactive": False,
                        "bankAccountPresentation": [
                            {"iban": "", "bban": "111", "provider": "AUTOPAY"},
                            {"iban": "NO123", "bban": "222", "provider": "NETS"},
                        ],
                    }
                ]
            }
        ),
        FakeResponse({"values": []}),
    ]
    dataset = make_dataset(responses, product_name=TripletexProductName.SUPPLIER_BANK_ACCOUNTS_LITE)

    dataset.read()

    assert len(dataset.output) == 2
    assert list(dataset.output.columns) == [
        "id",
        "isInactive",
        "bankAccountPresentation_iban",
        "bankAccountPresentation_bban",
        "bankAccountPresentation_provider",
    ]
    assert list(dataset.output["bankAccountPresentation_bban"]) == ["111", "222"]


def test_read_explode_drops_rows_with_empty_list():
    """A row whose list-of-objects field is empty is dropped -- nothing to explode into."""
    responses = [
        FakeResponse(
            {
                "values": [
                    {"id": 1, "isInactive": False, "bankAccountPresentation": []},
                    {
                        "id": 2,
                        "isInactive": False,
                        "bankAccountPresentation": [{"iban": "", "bban": "333", "provider": "AUTOPAY"}],
                    },
                ]
            }
        ),
        FakeResponse({"values": []}),
    ]
    dataset = make_dataset(responses, product_name=TripletexProductName.SUPPLIER_BANK_ACCOUNTS_LITE)

    dataset.read()

    assert len(dataset.output) == 1
    assert dataset.output.iloc[0]["id"] == 2


def test_read_explode_handles_empty_result():
    """An empty read (no matching records at all) doesn't crash -- the column doesn't exist yet."""
    responses = [FakeResponse({"values": []})]
    dataset = make_dataset(responses, product_name=TripletexProductName.SUPPLIER_BANK_ACCOUNTS_LITE)

    dataset.read()

    assert dataset.output.empty


def test_read_explode_raises_when_column_missing_from_nonempty_result():
    """A non-empty result missing the named explode column raises ReadError -- likely a packaging bug."""
    responses = [
        FakeResponse({"values": [{"id": 1, "isInactive": False}]}),
        FakeResponse({"values": []}),
    ]
    dataset = make_dataset(responses, product_name=TripletexProductName.SUPPLIER_BANK_ACCOUNTS_LITE)

    with pytest.raises(ReadError, match="bankAccountPresentation"):
        dataset.read()


def test_read_explode_raises_read_error_when_value_is_not_a_list():
    responses = [
        FakeResponse(
            {
                "values": [
                    {"id": 1, "isInactive": False, "bankAccountPresentation": 0},
                ]
            }
        ),
        FakeResponse({"values": []}),
    ]
    dataset = make_dataset(responses, product_name=TripletexProductName.SUPPLIER_BANK_ACCOUNTS_LITE)

    with pytest.raises(ReadError, match="bankAccountPresentation"):
        dataset.read()


def test_read_does_not_advance_checkpoint_when_post_processing_fails():
    """A malformed response that fails _explode_column must not commit a new checkpoint.

    The reader itself succeeds (fetches every page, gets a real versionDigest) --
    it's the *later* deserialize/explode step, in `finally`, that raises. If the
    checkpoint were committed in an `else` clause (which runs before `finally`),
    the new digest would already be persisted despite read() ultimately failing,
    so a later run would see it as already-current and skip data that never
    produced a valid output.
    """
    responses = [
        FakeResponse(
            {
                "values": [
                    {"id": 1, "isInactive": False, "bankAccountPresentation": 0},
                ],
                "versionDigest": "brand-new-digest",
            }
        ),
        FakeResponse({"values": []}),
    ]
    prior_checkpoint = {"watermark_digests": {}, "watermark_changed_since": {}, "resume_offsets": {}}
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.SUPPLIER_BANK_ACCOUNTS_LITE,
        checkpoint=dict(prior_checkpoint),
    )

    with pytest.raises(ReadError, match="bankAccountPresentation"):
        dataset.read()

    assert dataset.checkpoint == prior_checkpoint


def test_read_explode_raises_read_error_when_value_is_a_string_not_a_list():
    responses = [
        FakeResponse(
            {
                "values": [
                    {"id": 1, "isInactive": False, "bankAccountPresentation": "not-a-list"},
                ]
            }
        ),
        FakeResponse({"values": []}),
    ]
    dataset = make_dataset(responses, product_name=TripletexProductName.SUPPLIER_BANK_ACCOUNTS_LITE)

    with pytest.raises(ReadError, match="bankAccountPresentation"):
        dataset.read()


def test_serializer_and_deserializer_default_when_explicitly_none():
    """Explicitly passing serializer=None/deserializer=None also fills in the default."""
    linked_service = make_linked_service([])
    settings = TripletexDatasetSettings(product_name=TripletexProductName.CUSTOMER)
    dataset = TripletexDataset(
        id=uuid4(),
        name="test_dataset",
        version="1.0.0",
        linked_service=linked_service,
        settings=settings,
        serializer=None,
        deserializer=None,
    )
    assert isinstance(dataset.serializer, tripletex_mod.PandasSerializer)
    assert isinstance(dataset.deserializer, tripletex_mod.PandasDeserializer)


def test_serializer_and_deserializer_caller_override_is_honored():
    """A caller-supplied serializer/deserializer instance is kept as-is, not overridden."""
    linked_service = make_linked_service([])
    settings = TripletexDatasetSettings(product_name=TripletexProductName.CUSTOMER)
    custom_serializer = tripletex_mod.PandasSerializer(format=tripletex_mod.DatasetStorageFormatType.CSV, kwargs={})
    custom_deserializer = tripletex_mod.PandasDeserializer(format=tripletex_mod.DatasetStorageFormatType.CSV, kwargs={})
    dataset = TripletexDataset(
        id=uuid4(),
        name="test_dataset",
        version="1.0.0",
        linked_service=linked_service,
        settings=settings,
        serializer=custom_serializer,
        deserializer=custom_deserializer,
    )
    assert dataset.serializer is custom_serializer
    assert dataset.deserializer is custom_deserializer


# -----------------------------------------------------------------------------
# Contract: read() returns None, populates self.output
# -----------------------------------------------------------------------------


def test_read_returns_none():
    """read() must return None per contract."""
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(responses)
    assert dataset.read() is None


def test_read_populates_output_single_page():
    """read() populates self.output with a single page of results."""
    responses = [
        FakeResponse({"values": [{"id": 1, "name": "Acme"}], "versionDigest": "d1"}),
        FakeResponse({"values": [], "versionDigest": "d1"}),
    ]
    dataset = make_dataset(responses)
    dataset.read()
    assert isinstance(dataset.output, pd.DataFrame)
    assert len(dataset.output) == 1
    assert dataset.output.iloc[0]["name"] == "Acme"


def test_digest_pagination_advances_by_actual_rows_not_requested_count():
    responses = [
        FakeResponse({"values": [{"id": 1}, {"id": 2}, {"id": 3}], "versionDigest": "d1"}),
        FakeResponse({"values": [{"id": 4}, {"id": 5}], "versionDigest": "d1"}),
        FakeResponse({"values": [], "versionDigest": "d1"}),
    ]
    dataset = make_dataset(responses, product_name=TripletexProductName.DEPARTMENT, read=TripletexReadSettings(count=10))

    dataset.read()

    requests = dataset.linked_service.connection.requests
    assert requests[0]["params"]["from"] == 0
    assert requests[1]["params"]["from"] == 3  # actual rows from page 1, not requested count=10
    assert requests[2]["params"]["from"] == 5  # 3 + 2 actual rows so far
    assert len(dataset.output) == 5


def test_changed_since_pagination_advances_by_actual_rows_not_requested_count():
    responses = [
        FakeResponse({"values": [{"id": 1}, {"id": 2}, {"id": 3}]}),
        FakeResponse({"values": [{"id": 4}, {"id": 5}]}),
        FakeResponse({"values": []}),
    ]
    dataset = make_dataset(responses, product_name=TripletexProductName.CUSTOMER, read=TripletexReadSettings(count=10))

    dataset.read()

    requests = dataset.linked_service.connection.requests
    assert requests[0]["params"]["from"] == 0
    assert requests[1]["params"]["from"] == 3
    assert requests[2]["params"]["from"] == 5
    assert len(dataset.output) == 5


def test_read_multiple_pages_are_concatenated():
    """read() follows from/count pagination until an empty page is returned."""
    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "d1"}),
        FakeResponse({"values": [{"id": 2}], "versionDigest": "d1"}),
        FakeResponse({"values": [], "versionDigest": "d1"}),
    ]
    dataset = make_dataset(responses, read=TripletexReadSettings(count=1))
    dataset.read()
    assert len(dataset.output) == 2
    assert list(dataset.output["id"]) == [1, 2]


def test_read_pagination_uses_from_and_count():
    """Each page request advances 'from' by the configured 'count'."""
    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "d1"}),
        FakeResponse({"values": [], "versionDigest": "d1"}),
    ]
    dataset = make_dataset(responses, read=TripletexReadSettings(count=1))
    dataset.read()
    requests = dataset.linked_service.connection.requests
    assert requests[0]["params"]["from"] == 0
    assert requests[0]["params"]["count"] == 1
    assert requests[1]["params"]["from"] == 1


def test_read_empty_result_is_not_an_error():
    """read() with an empty first page returns an empty DataFrame, not an error."""
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(responses)
    dataset.read()
    assert isinstance(dataset.output, pd.DataFrame)
    assert dataset.output.empty


def test_read_uses_url_from_endpoint_path():
    """read() builds the request URL from the linked service host and endpoint path."""
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(responses, product_name=TripletexProductName.LEDGER_ACCOUNT_LITE)
    # ledger_account_lite is not date-windowed, so this exercises the plain path.
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["url"] == "https://tripletex.no/v2/ledger/account"


def test_read_raises_when_fields_set_alongside_product_name():
    """read() raises ReadError when read.fields is set alongside product_name -- it would be silently ignored."""
    dataset = make_dataset([], read=TripletexReadSettings(count=100, fields=["id"]))
    with pytest.raises(ReadError):
        dataset.read()


def test_read_raises_when_pagination_set_alongside_product_name():
    """read() raises ReadError when read.pagination is set alongside product_name -- it would be silently ignored."""
    dataset = make_dataset([], read=TripletexReadSettings(count=100, pagination=PaginationKind.OFFSET))
    with pytest.raises(ReadError):
        dataset.read()


def test_read_raises_when_changed_since_set_alongside_product_name():
    """read() raises ReadError when read.changed_since is set alongside product_name -- it would be silently ignored."""
    dataset = make_dataset([], read=TripletexReadSettings(count=100, changed_since=True))
    with pytest.raises(ReadError):
        dataset.read()


def test_read_merges_extra_params():
    """read() merges settings.read.params into every request's query params."""
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.DEPARTMENT,
        read=TripletexReadSettings(count=100, params={"foo": "bar"}),
    )
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["params"]["foo"] == "bar"


def test_digest_pagination_raises_when_params_sets_reserved_key():
    """settings.read.params can't set from/count/fields -- must raise, not silently collide.

    Caught upfront by _validate_read_settings, before any request is made --
    a stray "from" would otherwise pin every request to a fixed offset,
    refetching the same page forever.
    """
    dataset = make_dataset(
        [],
        product_name=TripletexProductName.DEPARTMENT,
        read=TripletexReadSettings(count=1, params={"from": 0, "count": 999, "fields": "bogus"}),
    )
    with pytest.raises(ReadError, match="from"):
        dataset.read()
    assert len(dataset.linked_service.connection.requests) == 0


def test_digest_pagination_raises_when_params_sets_sorting():
    """settings.read.params can't set sorting either -- this module pins sorting=id itself."""
    dataset = make_dataset(
        [],
        product_name=TripletexProductName.DEPARTMENT,
        read=TripletexReadSettings(count=1, params={"sorting": "name"}),
    )
    with pytest.raises(ReadError, match="sorting"):
        dataset.read()
    assert len(dataset.linked_service.connection.requests) == 0


def test_digest_pagination_sends_sorting_id_on_every_request():
    """Every digest-paginated request explicitly pins sorting=id.

    Confirmed live: this matches Tripletex's own default order, and makes
    it explicit/guaranteed rather than relying on an undocumented default --
    new rows (server-assigned, increasing IDs) then always append after
    what's already been seen, never shifting earlier offsets.
    """
    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "d1"}),
        FakeResponse({"values": []}),
    ]
    dataset = make_dataset(responses, product_name=TripletexProductName.DEPARTMENT, read=TripletexReadSettings(count=1))
    dataset.read()
    requests = dataset.linked_service.connection.requests
    assert len(requests) == 2
    assert all(r["params"]["sorting"] == "id" for r in requests)


def test_read_raises_when_no_stable_sort_field_can_be_resolved():
    """A product with no top-level id and no stable_sort_field override raises ReadError.

    Resumable pagination can't guarantee row order without a stable,
    server-assigned, monotonic field to sort by -- silently omitting
    sorting would risk a skip or duplicate on resume.
    """
    dataset = make_dataset(
        [],
        product_name=None,
        read=TripletexReadSettings(
            count=1, path="balanceSheet", fields=["account"], pagination=PaginationKind.OFFSET, changed_since=False
        ),
    )
    with pytest.raises(ReadError, match="stable_sort_field"):
        dataset.read()


def test_digest_pagination_sends_stable_sort_field_override():
    """settings.read.stable_sort_field is sent as sorting when the product has no top-level id."""
    responses = [
        FakeResponse({"values": [{"account": {"id": 1}}], "versionDigest": "d1"}),
        FakeResponse({"values": []}),
    ]
    dataset = make_dataset(
        responses,
        product_name=None,
        read=TripletexReadSettings(
            count=1,
            path="balanceSheet",
            fields=["account"],
            pagination=PaginationKind.OFFSET,
            changed_since=False,
            stable_sort_field="account.id",
        ),
    )
    dataset.read()
    requests = dataset.linked_service.connection.requests
    assert len(requests) == 2
    assert all(r["params"]["sorting"] == "account.id" for r in requests)


def test_stable_sort_field_set_alongside_product_name_is_silently_ignored():
    """settings.read.stable_sort_field alongside product_name has no effect -- the packaged metadata wins."""
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(
        responses, product_name=TripletexProductName.DEPARTMENT, read=TripletexReadSettings(stable_sort_field="name")
    )
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["params"]["sorting"] == "id"


def test_stable_sort_field_defaults_to_id_for_custom_path():
    """A custom path with no explicit stable_sort_field override defaults to sorting=id."""
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(
        responses,
        product_name=None,
        read=TripletexReadSettings(
            path="custom/thing", fields=["id", "name"], pagination=PaginationKind.OFFSET, changed_since=False
        ),
    )
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["params"]["sorting"] == "id"


def test_digest_pagination_sends_no_sorting_when_stable_sort_field_is_none():
    """settings.read.stable_sort_field=None opts out: no sorting is sent, even though fields has no id."""
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(
        responses,
        product_name=None,
        read=TripletexReadSettings(
            path="balanceSheet",
            fields=["account"],
            pagination=PaginationKind.OFFSET,
            changed_since=False,
            stable_sort_field=None,
        ),
    )
    dataset.read()
    assert "sorting" not in dataset.linked_service.connection.requests[0]["params"]


def test_digest_pagination_sends_no_sorting_when_stable_sort_field_is_empty_string():
    """settings.read.stable_sort_field="" opts out the same way as None."""
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(
        responses,
        product_name=None,
        read=TripletexReadSettings(
            path="balanceSheet",
            fields=["account"],
            pagination=PaginationKind.OFFSET,
            changed_since=False,
            stable_sort_field="",
        ),
    )
    dataset.read()
    assert "sorting" not in dataset.linked_service.connection.requests[0]["params"]


def test_changed_since_pagination_sends_no_sorting_when_stable_sort_field_is_none():
    """changedSince pagination also honors the stable_sort_field=None opt-out."""
    responses = [FakeResponse({"values": [], "versionDigest": "irrelevant"})]
    dataset = make_dataset(
        responses,
        product_name=None,
        read=TripletexReadSettings(
            path="custom/thing",
            fields=["name"],
            pagination=PaginationKind.OFFSET,
            changed_since=True,
            stable_sort_field=None,
        ),
        checkpoint={},
    )
    dataset.read()
    assert "sorting" not in dataset.linked_service.connection.requests[0]["params"]


def test_balance_sheet_packaged_product_sends_account_id_as_sorting(monkeypatch):
    """The packaged balance_sheet product resolves stable_sort_field from its own metadata (account.id)."""
    monkeypatch.setattr(tripletex_mod, "_period_generator", lambda **kwargs: iter([(date(2024, 1, 1), date(2024, 2, 1))]))
    responses = [
        FakeResponse({"values": [{"account": {"id": 1}}], "versionDigest": None}),
        FakeResponse({"values": []}),
    ]
    dataset = make_dataset(responses, product_name=TripletexProductName.BALANCE_SHEET)
    dataset.read()
    requests = dataset.linked_service.connection.requests
    assert all(r["params"]["sorting"] == "account.id" for r in requests)


# -----------------------------------------------------------------------------
# Contract: checkpoint (conditional GET via versionDigest)
# -----------------------------------------------------------------------------


def test_full_load_sends_magic_value_if_none_match():
    """An empty checkpoint means a full load: If-None-Match is the sentinel value."""
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(responses, product_name=TripletexProductName.DEPARTMENT, checkpoint={})
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["headers"]["If-None-Match"] == "magic-value"


def test_incremental_load_sends_stored_digest():
    """A populated checkpoint sends the previously stored versionDigest as If-None-Match."""
    department_info = get_read_info(TripletexProductName.DEPARTMENT)
    fields_param = _build_fields_param(department_info.fields)
    key = _request_key(fields_param)
    responses = [FakeResponse({"values": [], "versionDigest": "d2"}, status_code=304)]
    dataset = make_dataset(responses, product_name=TripletexProductName.DEPARTMENT, checkpoint={"watermark_digests": {key: "d1"}})
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["headers"]["If-None-Match"] == "d1"


def test_304_short_circuits_without_further_pages():
    """A 304 response stops pagination immediately with no further requests."""
    responses = [FakeResponse({}, status_code=304)]
    dataset = make_dataset(responses, product_name=TripletexProductName.DEPARTMENT, checkpoint={"watermark_digests": {}})
    dataset.read()
    assert dataset.output.empty
    assert len(dataset.linked_service.connection.requests) == 1


def test_checkpoint_updated_after_successful_read():
    """A successful read stores the response's versionDigest in self.checkpoint."""
    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "new-digest"}),
        FakeResponse({"values": [], "versionDigest": "new-digest"}),
    ]
    dataset = make_dataset(responses, product_name=TripletexProductName.DEPARTMENT, checkpoint={})
    dataset.read()
    assert "new-digest" in dataset.checkpoint["watermark_digests"].values()


def test_checkpoint_preserves_other_products_watermark_digests():
    """Existing watermark_digests for other request keys are preserved across a read()."""
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.DEPARTMENT,
        checkpoint={"watermark_digests": {"unrelated-key": "unrelated-digest"}},
    )
    dataset.read()
    assert dataset.checkpoint["watermark_digests"]["unrelated-key"] == "unrelated-digest"


# -----------------------------------------------------------------------------
# Contract: changedSince (customer/supplier and their _lite variants only)
# -----------------------------------------------------------------------------


def test_changed_since_product_sends_no_filter_on_first_run():
    """A changedSince-capable product with no stored changed_since_watermark reads everything, unfiltered."""
    responses = [FakeResponse({"values": [], "versionDigest": "irrelevant"})]
    dataset = make_dataset(responses, product_name=TripletexProductName.CUSTOMER, checkpoint={})
    dataset.read()
    request = dataset.linked_service.connection.requests[0]
    assert "changedSince" not in request["params"]
    assert request["headers"] is None


def test_changed_since_pagination_sends_sorting_id_on_every_request():
    """changedSince pagination pins sorting=id too, so a row changed mid-run can't shift pages."""
    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "irrelevant"}),
        FakeResponse({"values": []}),
    ]
    dataset = make_dataset(
        responses, product_name=TripletexProductName.CUSTOMER, read=TripletexReadSettings(count=1), checkpoint={}
    )
    dataset.read()
    requests = dataset.linked_service.connection.requests
    assert len(requests) == 2
    assert all(r["params"]["sorting"] == "id" for r in requests)


def test_changed_since_pagination_raises_when_no_stable_sort_field_can_be_resolved():
    """A changedSince-capable custom path with no top-level id and no override raises ReadError."""
    dataset = make_dataset(
        [],
        product_name=None,
        read=TripletexReadSettings(
            count=1, path="custom/thing", fields=["name"], pagination=PaginationKind.OFFSET, changed_since=True
        ),
        checkpoint={},
    )
    with pytest.raises(ReadError, match="stable_sort_field"):
        dataset.read()


def test_changed_since_pagination_sends_stable_sort_field_override():
    """settings.read.stable_sort_field is sent as sorting for changedSince pagination too."""
    responses = [
        FakeResponse({"values": [{"account": {"id": 1}}], "versionDigest": "irrelevant"}),
        FakeResponse({"values": []}),
    ]
    dataset = make_dataset(
        responses,
        product_name=None,
        read=TripletexReadSettings(
            count=1,
            path="custom/thing",
            fields=["account"],
            pagination=PaginationKind.OFFSET,
            changed_since=True,
            stable_sort_field="account.id",
        ),
        checkpoint={},
    )
    dataset.read()
    requests = dataset.linked_service.connection.requests
    assert len(requests) == 2
    assert all(r["params"]["sorting"] == "account.id" for r in requests)


def test_changed_since_product_sends_stored_changed_since_watermark():
    """A changedSince-capable product with a stored changed_since_watermark sends it as the changedSince filter."""
    customer_info = get_read_info(TripletexProductName.CUSTOMER)
    key = _request_key(_build_fields_param(customer_info.fields))
    responses = [FakeResponse({"values": [], "versionDigest": "irrelevant"})]
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.CUSTOMER,
        checkpoint={"watermark_changed_since": {key: "2024-01-01T00:00:00Z"}},
    )
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["params"]["changedSince"] == "2024-01-01T00:00:00Z"


def test_changed_since_product_stores_new_changed_since_watermark_after_success():
    """A successful read stores a new YYYY-MM-DDThh:mm:ssZ changed_since_watermark for the next run."""
    responses = [FakeResponse({"values": [], "versionDigest": "irrelevant"})]
    dataset = make_dataset(responses, product_name=TripletexProductName.CUSTOMER, checkpoint={})
    dataset.read()
    (stored_changed_since_watermark,) = dataset.checkpoint["watermark_changed_since"].values()
    datetime.strptime(stored_changed_since_watermark, "%Y-%m-%dT%H:%M:%SZ")  # raises ValueError if malformed


def test_changed_since_product_merges_extra_params():
    """read() merges settings.read.params into changedSince requests too."""
    responses = [FakeResponse({"values": [], "versionDigest": "irrelevant"})]
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.CUSTOMER,
        read=TripletexReadSettings(count=100, params={"foo": "bar"}),
    )
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["params"]["foo"] == "bar"


def test_changed_since_pagination_raises_when_params_sets_reserved_key():
    """settings.read.params can't set changedSince -- must raise, not silently collide."""
    dataset = make_dataset(
        [],
        product_name=TripletexProductName.CUSTOMER,
        read=TripletexReadSettings(count=1, params={"changedSince": "garbage"}),
    )
    with pytest.raises(ReadError, match="changedSince"):
        dataset.read()
    assert len(dataset.linked_service.connection.requests) == 0


def test_changed_since_product_does_not_advance_changed_since_watermark_on_mid_pagination_failure():
    """A changed_since_watermark captured this run must not be persisted if a later page fails.

    watermark_digests/watermark_changed_since stay byte-identical to what they were before the
    call, per contract. resume_offsets stays empty too -- changedSince
    pagination is deliberately not resumable (a stored offset isn't a safe
    resume point for a moving, changedSince-filtered result set), so the
    next run always restarts at offset=0 rather than a stale position.
    """
    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "irrelevant"}),
        ResourceException(message="unexpected failure"),
    ]
    prior_checkpoint = {"watermark_digests": {}, "watermark_changed_since": {"stale": "2020-01-01T00:00:00Z"}}
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.CUSTOMER,
        read=TripletexReadSettings(count=1),
        checkpoint=prior_checkpoint,
    )
    with pytest.raises(ReadError):
        dataset.read()
    assert dataset.checkpoint["watermark_digests"] == prior_checkpoint["watermark_digests"]
    assert dataset.checkpoint["watermark_changed_since"] == prior_checkpoint["watermark_changed_since"]
    assert dataset.checkpoint["resume_offsets"] == {}


# -----------------------------------------------------------------------------
# Contract: error handling
# -----------------------------------------------------------------------------


def test_read_unknown_product_raises_read_error():
    """read() raises ReadError when the product has no packaged read assets."""
    dataset = make_dataset([])
    dataset.settings.product_name = "not-a-real-product"
    with pytest.raises(ReadError):
        dataset.read()


def test_read_wraps_validation_error_from_malformed_packaged_metadata(monkeypatch):
    """read() wraps get_read_info()'s ValidationError (a packaging bug) into ReadError, per the Error Contract."""

    def _raise_validation_error(product_name):
        raise ValidationError(message="malformed metadata", details={"product_name": product_name.value})

    monkeypatch.setattr(tripletex_mod, "get_read_info", _raise_validation_error)
    dataset = make_dataset([])
    with pytest.raises(ReadError) as exc_info:
        dataset.read()
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_read_wraps_backend_exception_in_read_error():
    """Generic backend exceptions are wrapped in ReadError with chaining."""
    responses = [ResourceException(message="unexpected failure")]
    dataset = make_dataset(responses)
    with pytest.raises(ReadError) as exc_info:
        dataset.read()
    assert exc_info.value.__cause__ is not None


def test_read_authentication_error_passes_through_unwrapped():
    """AuthenticationError is not wrapped -- callers can catch it directly."""
    responses = [AuthenticationError("bad credentials")]
    dataset = make_dataset(responses)
    with pytest.raises(AuthenticationError):
        dataset.read()


def test_read_partial_results_preserved_on_error():
    """self.output retains rows collected before a mid-pagination failure."""
    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "d1"}),
        ResourceException(message="unexpected failure"),
    ]
    dataset = make_dataset(responses, read=TripletexReadSettings(count=1))
    with pytest.raises(ReadError):
        dataset.read()
    assert len(dataset.output) == 1


def test_read_does_not_advance_digest_on_mid_pagination_failure():
    """A digest seen on an early page must not be persisted if a later page fails.

    Otherwise the next run would see that digest as already-current (a 304) and
    skip re-fetching data this run never actually finished retrieving.
    watermark_digests/watermark_changed_since stay byte-identical to what they were before the
    call, per contract -- pagination separately records the resume offset.
    """
    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "d1"}),
        ResourceException(message="unexpected failure"),
    ]
    prior_checkpoint = {"watermark_digests": {"stale": "old-digest"}, "watermark_changed_since": {}}
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.DEPARTMENT,
        read=TripletexReadSettings(count=1),
        checkpoint=prior_checkpoint,
    )
    with pytest.raises(ReadError):
        dataset.read()
    assert dataset.checkpoint["watermark_digests"] == prior_checkpoint["watermark_digests"]
    assert dataset.checkpoint["watermark_changed_since"] == prior_checkpoint["watermark_changed_since"]
    (resume_offset,) = dataset.checkpoint["resume_offsets"].values()
    assert resume_offset == 1
    (resume_digest,) = dataset.checkpoint["resume_digests"].values()
    assert resume_digest == "d1"


def test_read_resumes_digest_pagination_from_stored_offset():
    """A stored offset is used, unmodified, when its paired resume_digest matches the resumed page.

    A match proves nothing changed since the offset was recorded, so it's
    genuinely safe to continue from there instead of re-paging from 0.
    """
    department_info = get_read_info(TripletexProductName.DEPARTMENT)
    fields_param = _build_fields_param(department_info.fields)
    key = _request_key(fields_param)
    responses = [FakeResponse({"values": [{"id": 3}], "versionDigest": "d1"}), FakeResponse({"values": []})]
    prior_checkpoint = {
        "watermark_digests": {},
        "watermark_changed_since": {},
        "resume_offsets": {key: 2},
        "resume_digests": {key: "d1"},
    }
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.DEPARTMENT,
        read=TripletexReadSettings(count=1),
        checkpoint=prior_checkpoint,
    )
    dataset.read()
    assert [r["params"]["from"] for r in dataset.linked_service.connection.requests] == [2, 3]
    assert dataset.checkpoint["resume_offsets"] == {}
    assert dataset.checkpoint["resume_digests"] == {}
    assert len(dataset.output) == 1


def test_read_resuming_sends_unconditional_get_ignoring_stored_watermark_digest():
    """Resuming never sends the real watermark_digest as If-None-Match, even if one is stored.

    A conditional GET only answers "did the whole collection change since my
    last full success" -- irrelevant mid-resume, since rows past the stored
    offset are already known to be unfetched. Trusting a 304 there would
    bypass the mismatch check and could wrongly treat an unverified,
    incomplete resume as "done".
    """
    department_info = get_read_info(TripletexProductName.DEPARTMENT)
    fields_param = _build_fields_param(department_info.fields)
    key = _request_key(fields_param)
    responses = [FakeResponse({"values": [{"id": 3}], "versionDigest": "d1"}), FakeResponse({"values": []})]
    prior_checkpoint = {
        "watermark_digests": {key: "d1"},
        "watermark_changed_since": {},
        "resume_offsets": {key: 2},
        "resume_digests": {key: "d1"},
    }
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.DEPARTMENT,
        read=TripletexReadSettings(count=1),
        checkpoint=prior_checkpoint,
    )
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["headers"]["If-None-Match"] == "magic-value"
    assert len(dataset.output) == 1


def test_read_restarts_digest_pagination_when_resume_digest_does_not_match():
    """A stored offset whose paired digest no longer matches the resumed page is discarded.

    Something changed since that offset was recorded -- rows before it may
    have shifted too, so it can't be trusted; this key restarts from 0
    instead, re-fetching everything rather than risking a skip.
    """
    department_info = get_read_info(TripletexProductName.DEPARTMENT)
    fields_param = _build_fields_param(department_info.fields)
    key = _request_key(fields_param)
    responses = [
        FakeResponse({"values": [{"id": 99}], "versionDigest": "d2"}),  # resumed at from=2 -- digest changed, discarded
        FakeResponse({"values": [{"id": 1}], "versionDigest": "d2"}),  # restarted at from=0
        FakeResponse({"values": [{"id": 99}], "versionDigest": "d2"}),
        FakeResponse({"values": []}),
    ]
    prior_checkpoint = {
        "watermark_digests": {},
        "watermark_changed_since": {},
        "resume_offsets": {key: 2},
        "resume_digests": {key: "d1"},
    }
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.DEPARTMENT,
        read=TripletexReadSettings(count=1),
        checkpoint=prior_checkpoint,
    )
    dataset.read()
    assert [r["params"]["from"] for r in dataset.linked_service.connection.requests] == [2, 0, 1, 2]
    assert dataset.checkpoint["resume_offsets"] == {}
    assert dataset.checkpoint["resume_digests"] == {}
    assert sorted(dataset.output["id"].tolist()) == [1, 99]


def test_changed_since_pagination_ignores_stored_offset_and_starts_at_zero():
    """A stored resume offset is ignored for changedSince pagination -- always restarts at 0.

    changedSince filters a mutable, moving result set; a stored offset from
    a prior attempt isn't a safe resume point (see
    _read_paginated_by_changed_since's docstring), so any leftover
    resume_offsets entry for this key must not be used.
    """
    customer_info = get_read_info(TripletexProductName.CUSTOMER)
    fields_param = _build_fields_param(customer_info.fields)
    key = _request_key(fields_param)
    responses = [FakeResponse({"values": [{"id": 3}]}), FakeResponse({"values": []})]
    prior_checkpoint = {"watermark_digests": {}, "watermark_changed_since": {}, "resume_offsets": {key: 2}}
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.CUSTOMER,
        read=TripletexReadSettings(count=1),
        checkpoint=prior_checkpoint,
    )
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["params"]["from"] == 0
    assert dataset.checkpoint["resume_offsets"] == {}


def test_read_clears_pagination_on_full_success():
    """A prior resume position is dropped once its pass completes successfully."""
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    prior_checkpoint = {"watermark_digests": {}, "watermark_changed_since": {}, "resume_offsets": {"some-other-key": 5}}
    dataset = make_dataset(responses, product_name=TripletexProductName.DEPARTMENT, checkpoint=prior_checkpoint)
    dataset.read()
    assert dataset.checkpoint["resume_offsets"] == {}


def test_read_falls_back_to_magic_value_when_version_digest_is_null():
    """A null versionDigest (Tripletex's documented shape for an empty result) doesn't get stored as None."""
    responses = [FakeResponse({"values": [], "versionDigest": None})]
    dataset = make_dataset(responses, product_name=TripletexProductName.DEPARTMENT)
    dataset.read()
    (stored_digest,) = dataset.checkpoint["watermark_digests"].values()
    assert stored_digest == "magic-value"


def test_read_error_details_include_product_name():
    """ReadError details include the product that failed."""
    responses = [ResourceException(message="unexpected failure")]
    dataset = make_dataset(responses)
    with pytest.raises(ReadError) as exc_info:
        dataset.read()
    assert exc_info.value.details["product_name"] == "customer"


# -----------------------------------------------------------------------------
# Contract: custom endpoint override (read.path used only when product_name is unset)
# -----------------------------------------------------------------------------


def test_read_raises_when_neither_product_name_nor_path_set():
    """read() raises ReadError when both product_name and read.path are None."""
    dataset = make_dataset([], product_name=None)
    with pytest.raises(ReadError):
        dataset.read()


def test_read_raises_when_path_set_without_fields():
    """read() raises ReadError when read.path is set but read.fields is not."""
    dataset = make_dataset([], product_name=None, read=TripletexReadSettings(count=100, path="custom/thing"))
    with pytest.raises(ReadError):
        dataset.read()


def test_read_raises_when_date_from_set_for_offset_paginated_read():
    """read() raises ReadError when read.date_from is set but pagination isn't date-windowed."""
    dataset = make_dataset([], product_name=TripletexProductName.CUSTOMER, read=TripletexReadSettings(date_from="3"))
    with pytest.raises(ReadError):
        dataset.read()


def test_read_raises_when_changed_since_combined_with_date_window_pagination():
    dataset = make_dataset(
        [],
        product_name=None,
        read=TripletexReadSettings(
            path="custom/thing",
            fields=["id"],
            pagination=PaginationKind.DATE_WINDOW,
            changed_since=True,
        ),
    )
    with pytest.raises(ReadError):
        dataset.read()


def test_read_uses_custom_path_bypassing_packaged_catalog():
    """A custom read.path/read.fields is used directly, without any packaged product."""
    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "d1"}),
        FakeResponse({"values": [], "versionDigest": "d1"}),
    ]
    dataset = make_dataset(
        responses,
        product_name=None,
        read=TripletexReadSettings(
            count=100, path="custom/thing", fields=["id"], pagination=PaginationKind.OFFSET, changed_since=False
        ),
    )
    dataset.read()
    request = dataset.linked_service.connection.requests[0]
    assert request["url"] == "https://tripletex.no/v2/custom/thing"
    assert request["params"]["fields"] == "id"
    assert len(dataset.output) == 1


def test_read_raises_when_path_set_alongside_product_name():
    """read() raises ReadError when read.path is set alongside product_name -- it would be silently ignored."""
    dataset = make_dataset(
        [],
        product_name=TripletexProductName.CUSTOMER,
        read=TripletexReadSettings(count=100, path="custom/thing"),
    )
    with pytest.raises(ReadError):
        dataset.read()


def test_read_custom_path_supports_date_window_pagination(monkeypatch):
    """A custom read.path can opt into date-windowed pagination too."""
    periods = [(date(2024, 1, 1), date(2024, 2, 1))]
    monkeypatch.setattr(tripletex_mod, "_period_generator", lambda **kwargs: iter(periods))

    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "d1"}),
        FakeResponse({"values": [], "versionDigest": "d1"}),
    ]
    dataset = make_dataset(
        responses,
        product_name=None,
        read=TripletexReadSettings(
            count=100, path="custom/dated", fields=["id"], pagination=PaginationKind.DATE_WINDOW, changed_since=False
        ),
    )
    dataset.read()

    request = dataset.linked_service.connection.requests[0]
    assert request["url"] == "https://tripletex.no/v2/custom/dated"
    assert request["params"]["dateFrom"] == "2024-01-01"
    assert len(dataset.output) == 1


def test_read_custom_path_uses_offset_pagination_when_requested():
    """A custom read.path with pagination=OFFSET paginates via from/count, not date windows."""
    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "d1"}),
        FakeResponse({"values": [], "versionDigest": "d1"}),
    ]
    dataset = make_dataset(
        responses,
        product_name=None,
        read=TripletexReadSettings(
            count=100, path="custom/thing", fields=["id"], pagination=PaginationKind.OFFSET, changed_since=False
        ),
    )
    dataset.read()
    request = dataset.linked_service.connection.requests[0]
    assert "dateFrom" not in request["params"]
    assert request["params"]["from"] == 0


def test_read_custom_path_supports_changed_since():
    """A custom read.path can opt into Tripletex's changedSince filter via read.changed_since."""
    responses = [FakeResponse({"values": [], "versionDigest": "irrelevant"})]
    key = _request_key("id", extra_params=None)
    dataset = make_dataset(
        responses,
        product_name=None,
        read=TripletexReadSettings(
            count=100,
            path="custom/thing",
            fields=["id"],
            pagination=PaginationKind.OFFSET,
            changed_since=True,
        ),
        checkpoint={"watermark_changed_since": {key: "2024-01-01T00:00:00Z"}},
    )
    dataset.read()
    request = dataset.linked_service.connection.requests[0]
    assert request["params"]["changedSince"] == "2024-01-01T00:00:00Z"
    assert request["headers"] is None


def test_read_raises_when_path_set_without_pagination():
    """read() raises ReadError when read.path is set but read.pagination is not."""
    dataset = make_dataset(
        [],
        product_name=None,
        read=TripletexReadSettings(count=100, path="custom/thing", fields=["id"]),
    )
    with pytest.raises(ReadError):
        dataset.read()


def test_read_raises_when_path_set_without_changed_since():
    """read() raises ReadError when read.path is set but read.changed_since is not (no safe default)."""
    dataset = make_dataset(
        [],
        product_name=None,
        read=TripletexReadSettings(count=100, path="custom/thing", fields=["id"], pagination=PaginationKind.OFFSET),
    )
    with pytest.raises(ReadError):
        dataset.read()


# -----------------------------------------------------------------------------
# Contract: date-windowed pagination (ledger_posting)
# -----------------------------------------------------------------------------


def test_date_windowed_product_sends_date_range_params(monkeypatch):
    """ledger_posting requests include dateFrom/dateTo bounds for each window."""
    periods = [(date(2024, 1, 1), date(2024, 2, 1)), (date(2024, 2, 1), date(2024, 3, 1))]
    monkeypatch.setattr(tripletex_mod, "_period_generator", lambda **kwargs: iter(periods))

    responses = [
        FakeResponse({"values": [], "versionDigest": "d1"}),
        FakeResponse({"values": [], "versionDigest": "d1"}),
    ]
    dataset = make_dataset(responses, product_name=TripletexProductName.LEDGER_POSTING)
    dataset.read()

    requests = dataset.linked_service.connection.requests
    assert len(requests) == 2
    assert requests[0]["params"]["dateFrom"] == "2024-01-01"
    assert requests[0]["params"]["dateTo"] == "2024-02-01"
    assert requests[1]["params"]["dateFrom"] == "2024-02-01"
    assert requests[1]["params"]["dateTo"] == "2024-03-01"


def test_date_windowed_product_paginates_within_a_window(monkeypatch):
    """Each date window is itself offset-paginated until an empty page."""
    periods = [(date(2024, 1, 1), date(2024, 2, 1))]
    monkeypatch.setattr(tripletex_mod, "_period_generator", lambda **kwargs: iter(periods))

    responses = [
        FakeResponse({"values": [{"id": 1}], "versionDigest": "d1"}),
        FakeResponse({"values": [], "versionDigest": "d1"}),
    ]
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.LEDGER_POSTING,
        read=TripletexReadSettings(count=1),
    )
    dataset.read()

    assert len(dataset.output) == 1
    requests = dataset.linked_service.connection.requests
    assert requests[0]["params"]["from"] == 0
    assert requests[1]["params"]["from"] == 1


def test_date_windowed_product_304_short_circuits_a_window(monkeypatch):
    """A 304 response for one date window stops pagination within that window only."""
    periods = [(date(2024, 1, 1), date(2024, 2, 1)), (date(2024, 2, 1), date(2024, 3, 1))]
    monkeypatch.setattr(tripletex_mod, "_period_generator", lambda **kwargs: iter(periods))

    responses = [
        FakeResponse({}, status_code=304),
        FakeResponse({"values": [{"id": 1}], "versionDigest": "d2"}),
        FakeResponse({"values": [], "versionDigest": "d2"}),
    ]
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.LEDGER_POSTING,
        checkpoint={"watermark_digests": {}},
    )
    dataset.read()

    assert len(dataset.output) == 1
    assert len(dataset.linked_service.connection.requests) == 3


def test_date_windowed_product_does_not_advance_digest_of_completed_period_on_later_period_failure(monkeypatch):
    """A period that fully completes must not persist its new digest if a later period fails.

    Per the dataset contract, the incremental watermark advances only after
    the whole read() succeeds end-to-end -- not per period -- so a later
    period's failure must leave every period's digest_watermark unchanged.
    """
    periods = [(date(2024, 1, 1), date(2024, 2, 1)), (date(2024, 2, 1), date(2024, 3, 1))]
    monkeypatch.setattr(tripletex_mod, "_period_generator", lambda **kwargs: iter(periods))

    responses = [
        FakeResponse({"values": [], "versionDigest": "new-digest"}),
        ResourceException(message="unexpected failure"),
    ]
    prior_checkpoint = {"watermark_digests": {}, "watermark_changed_since": {}}
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.LEDGER_POSTING,
        checkpoint=prior_checkpoint,
    )
    with pytest.raises(ReadError):
        dataset.read()

    assert dataset.checkpoint["watermark_digests"] == prior_checkpoint["watermark_digests"]


def test_date_windowed_product_uses_url_from_endpoint_path(monkeypatch):
    """ledger_posting requests target the ledger/posting path."""
    monkeypatch.setattr(tripletex_mod, "_period_generator", lambda **kwargs: iter([(date(2024, 1, 1), date(2024, 2, 1))]))
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(responses, product_name=TripletexProductName.LEDGER_POSTING)
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["url"] == "https://tripletex.no/v2/ledger/posting"


def test_date_windowed_product_merges_extra_params(monkeypatch):
    """read.params is merged into date-windowed requests too, not just plain offset ones."""
    monkeypatch.setattr(tripletex_mod, "_period_generator", lambda **kwargs: iter([(date(2024, 1, 1), date(2024, 2, 1))]))
    responses = [FakeResponse({"values": [], "versionDigest": "d1"})]
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.LEDGER_POSTING,
        read=TripletexReadSettings(count=100, params={"foo": "bar"}),
    )
    dataset.read()
    assert dataset.linked_service.connection.requests[0]["params"]["foo"] == "bar"


def test_date_windowed_product_raises_when_params_sets_reserved_key():
    """settings.read.params can't set dateFrom/dateTo -- must raise, not silently collide.

    The TripletexReadSettings.params docstring documents dateFrom/dateTo as
    "auto-generated per window and would override any values passed here" --
    in practice this is now caught upfront instead, before any request.
    """
    dataset = make_dataset(
        [],
        product_name=TripletexProductName.LEDGER_POSTING,
        read=TripletexReadSettings(count=1, params={"dateFrom": "bogus", "dateTo": "bogus"}),
    )
    with pytest.raises(ReadError, match="dateFrom"):
        dataset.read()
    assert len(dataset.linked_service.connection.requests) == 0


def test_date_windowed_product_date_to_is_exclusive_so_today_is_included(monkeypatch):
    captured_kwargs = {}

    def fake_period_generator(**kwargs):
        captured_kwargs.update(kwargs)
        return iter([])

    monkeypatch.setattr(tripletex_mod, "_period_generator", fake_period_generator)
    dataset = make_dataset([], product_name=TripletexProductName.LEDGER_POSTING)
    dataset.read()
    today = datetime.now(tz=timezone.utc).date()
    assert captured_kwargs["date_to"] == today + timedelta(days=1)


class _FrozenDatetime(datetime):
    frozen_now: "datetime"

    @classmethod
    def now(cls, tz=None):
        return cls.frozen_now


def test_date_windowed_product_date_from_zero_reads_today_even_on_first_of_month(monkeypatch):
    """Regression: date_from==date_to landing on day 1 must not collapse to zero periods."""
    _FrozenDatetime.frozen_now = datetime(2024, 3, 1, tzinfo=timezone.utc)  # today lands on day 1
    monkeypatch.setattr(tripletex_mod, "datetime", _FrozenDatetime)
    responses = [FakeResponse({"values": [{"id": 1}], "versionDigest": "d1"}), FakeResponse({"values": []})]
    dataset = make_dataset(
        responses,
        product_name=TripletexProductName.LEDGER_POSTING,
        read=TripletexReadSettings(date_from="0", count=1),
    )
    dataset.read()
    request = dataset.linked_service.connection.requests[0]
    assert request["params"]["dateFrom"] == "2024-03-01"
    assert request["params"]["dateTo"] == "2024-03-02"
    assert len(dataset.output) == 1


def test_date_windowed_product_uses_configured_date_from(monkeypatch):
    """settings.read.date_from, when set, overrides the default 3-year window start."""
    captured_kwargs = {}

    def fake_period_generator(**kwargs):
        captured_kwargs.update(kwargs)
        return iter([])

    monkeypatch.setattr(tripletex_mod, "_period_generator", fake_period_generator)
    dataset = make_dataset(
        [],
        product_name=TripletexProductName.LEDGER_POSTING,
        read=TripletexReadSettings(date_from="1"),
    )
    dataset.read()
    # date_to passed to the generator is today + 1 day (Tripletex's dateTo is
    # exclusive) -- date_from is offset from today itself, not from that.
    today = captured_kwargs["date_to"] - timedelta(days=1)
    assert captured_kwargs["date_from"] == _add_years(today, -1)


def test_date_windowed_product_uses_default_date_from_when_unset(monkeypatch):
    """settings.read.date_from defaults to 3 years before today when unset."""
    captured_kwargs = {}

    def fake_period_generator(**kwargs):
        captured_kwargs.update(kwargs)
        return iter([])

    monkeypatch.setattr(tripletex_mod, "_period_generator", fake_period_generator)
    dataset = make_dataset([], product_name=TripletexProductName.LEDGER_POSTING)
    dataset.read()
    today = captured_kwargs["date_to"] - timedelta(days=1)
    assert captured_kwargs["date_from"] == _add_years(today, -3)


def test_resolve_date_from_accepts_negative_year_offset():
    """A negative integer string resolves to that many years before date_to."""
    dataset = make_dataset([], read=TripletexReadSettings(date_from="-3"))
    assert dataset._resolve_date_from(date_to=date(2024, 6, 1)) == date(2021, 6, 1)


def test_resolve_date_from_accepts_positive_year_offset():
    """A positive integer string is treated identically to its negative counterpart.

    The sign isn't meaningful -- only the magnitude -- since date_from can
    never be after date_to.
    """
    dataset = make_dataset([], read=TripletexReadSettings(date_from="3"))
    assert dataset._resolve_date_from(date_to=date(2024, 6, 1)) == date(2021, 6, 1)


def test_resolve_date_from_defaults_to_three_years_back_when_unset():
    """An unset date_from defaults to 3 years before date_to."""
    dataset = make_dataset([])
    assert dataset._resolve_date_from(date_to=date(2024, 6, 1)) == date(2021, 6, 1)


def test_resolve_date_from_handles_leap_day_reference():
    """A leap-day date_to shifts to Feb 28th when the target year isn't a leap year."""
    dataset = make_dataset([], read=TripletexReadSettings(date_from="-1"))
    assert dataset._resolve_date_from(date_to=date(2024, 2, 29)) == date(2023, 2, 28)


def test_resolve_date_from_raises_on_unparseable_value():
    """A date_from that isn't a valid integer year-offset raises ReadError."""
    dataset = make_dataset([], read=TripletexReadSettings(date_from="not-a-number"))
    with pytest.raises(ReadError):
        dataset._resolve_date_from(date_to=date(2024, 6, 1))


# -----------------------------------------------------------------------------
# Contract: unsupported write methods
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("method_name", ["create", "update", "delete", "upsert", "rename", "purge", "list"])
def test_write_methods_raise_not_supported(method_name):
    """All write/discovery methods raise NotSupportedError -- Tripletex is read-only here."""
    dataset = make_dataset([])
    with pytest.raises(NotSupportedError):
        getattr(dataset, method_name)()


def test_close_closes_linked_service():
    """close() delegates to the linked service's close()."""
    dataset = make_dataset([])
    dataset.close()  # Should not raise; DummyTripletexLinkedService has no real HTTP client.


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def test_build_fields_param_flat_fields():
    """Plain strings render as their bare field name."""
    assert _build_fields_param(["id", "name"]) == "id,name"


def test_build_fields_param_nested_group():
    """A one-key dict expands to key(children)."""
    assert _build_fields_param([{"account": ["id"]}]) == "account(id)"


def test_build_fields_param_nested_group_within_group():
    """Groups can nest arbitrarily deep."""
    selector = [{"bankAccountPresentation": ["iban", {"country": ["id"]}]}]
    assert _build_fields_param(selector) == "bankAccountPresentation(iban,country(id))"


def test_request_key_is_stable_for_same_input():
    """_request_key returns the same hash for identical inputs."""
    assert _request_key("id,name") == _request_key("id,name")


def test_request_key_differs_by_fields():
    """_request_key differs when the fields projection differs."""
    assert _request_key("id") != _request_key("id,name")


def test_request_key_differs_by_date_window():
    """_request_key differs between date windows for the same fields."""
    key_a = _request_key("id", date_from="2024-01-01", date_to="2024-02-01")
    key_b = _request_key("id", date_from="2024-02-01", date_to="2024-03-01")
    assert key_a != key_b


def test_request_key_differs_by_extra_params():
    """_request_key differs between different settings.read.params filters for the same fields.

    Two different filter configurations (e.g. isInactive=true vs. false) must never
    share one versionDigest/changedSince cache entry -- they read different data.
    """
    key_a = _request_key("id", extra_params={"isInactive": True})
    key_b = _request_key("id", extra_params={"isInactive": False})
    key_none = _request_key("id")
    assert len({key_a, key_b, key_none}) == 3


def test_period_generator_yields_monthly_windows():
    """_period_generator yields consecutive, non-overlapping monthly windows."""
    periods = list(_period_generator(date_from=date(2024, 1, 15), date_to=date(2024, 4, 1)))
    assert periods == [
        (date(2024, 1, 1), date(2024, 2, 1)),
        (date(2024, 2, 1), date(2024, 3, 1)),
        (date(2024, 3, 1), date(2024, 4, 1)),
    ]
