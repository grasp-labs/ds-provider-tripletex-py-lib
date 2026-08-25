"""
**File:** ``test_read_info.py``
**Region:** ``tests``

Unit tests for the Tripletex packaged asset loader.
"""

import pytest
from ds_resource_plugin_py_lib.common.resource.errors import ValidationError

from ds_provider_tripletex_py_lib.enums import OperationType, PaginationKind, TripletexProductName
from ds_provider_tripletex_py_lib.read_info import _load_metadata, get_read_info


def test_every_product_has_read_assets():
    """Every TripletexProductName member must have packaged read metadata."""
    for product_name in TripletexProductName:
        get_read_info(product_name)  # raises ValidationError if assets are missing/malformed


def test_every_endpoint_has_a_non_empty_field_projection():
    """Every endpoint must declare at least one field to read."""
    for product_name in TripletexProductName:
        info = get_read_info(product_name)
        assert info.fields, f"{product_name!r} has an empty field projection"
        assert info.path, f"{product_name!r} has an empty path"


def test_lite_variants_share_the_base_endpoint_path():
    """_lite products read from the same path as their full counterpart."""
    pairs = [
        (TripletexProductName.CUSTOMER_LITE, TripletexProductName.CUSTOMER),
        (TripletexProductName.DEPARTMENT_LITE, TripletexProductName.DEPARTMENT),
        (TripletexProductName.EMPLOYEE_LITE, TripletexProductName.EMPLOYEE),
        (TripletexProductName.PROJECT_LITE, TripletexProductName.PROJECT),
        (TripletexProductName.SUPPLIER_LITE, TripletexProductName.SUPPLIER),
        (TripletexProductName.SUPPLIER_BANK_ACCOUNTS_LITE, TripletexProductName.SUPPLIER),
        (TripletexProductName.LEDGER_ACCOUNT_LITE, TripletexProductName.LEDGER_ACCOUNT),
    ]
    for lite_product, base_product in pairs:
        lite_info = get_read_info(lite_product)
        base_info = get_read_info(base_product)
        assert lite_info.path == base_info.path


def test_only_expected_products_are_date_windowed():
    """Only unbounded, ever-growing time series use date-window pagination."""
    date_windowed = {TripletexProductName.LEDGER_POSTING, TripletexProductName.BALANCE_SHEET}
    for product_name in TripletexProductName:
        info = get_read_info(product_name)
        expected = PaginationKind.DATE_WINDOW if product_name in date_windowed else PaginationKind.OFFSET
        assert info.pagination is expected, f"{product_name!r} pagination should be {expected}"


def test_only_expected_product_declares_explode_columns():
    """Only supplier_bank_accounts_lite has a list-of-objects field needing explosion."""
    for product_name in TripletexProductName:
        info = get_read_info(product_name)
        if product_name is TripletexProductName.SUPPLIER_BANK_ACCOUNTS_LITE:
            assert info.explode_columns == ["bankAccountPresentation"]
        else:
            assert info.explode_columns == [], f"{product_name!r} should have no explode_columns"


def test_nested_url_paths_use_slashes():
    """Products backed by nested Tripletex resources use slash-separated paths."""
    ledger_account = get_read_info(TripletexProductName.LEDGER_ACCOUNT)
    ledger_posting = get_read_info(TripletexProductName.LEDGER_POSTING)
    ledger_vat_type = get_read_info(TripletexProductName.LEDGER_VAT_TYPE)
    assert ledger_account.path == "ledger/account"
    assert ledger_posting.path == "ledger/posting"
    assert ledger_vat_type.path == "ledger/vatType"


# -----------------------------------------------------------------------------
# Asset loading: only read/ exists today, but create/update/delete are
# supported by the layout for a future write-capable product.
# -----------------------------------------------------------------------------


def test_load_metadata_raises_for_unimplemented_operation():
    """No product defines create/update/delete assets yet -- loading one raises."""
    with pytest.raises(FileNotFoundError):
        _load_metadata(TripletexProductName.CUSTOMER, OperationType.CREATE)


# -----------------------------------------------------------------------------
# Fail loudly on missing or malformed read metadata -- every TripletexProductName
# member is expected to ship read assets, so a packaged asset that's absent,
# missing a required key, or declaring an unrecognized pagination kind is a
# packaging bug, not a condition to silently paper over.
# -----------------------------------------------------------------------------


def test_get_read_info_raises_when_read_metadata_missing(monkeypatch):
    """get_read_info() raises ValidationError when a product has no read assets."""

    def _raise_not_found(product_name: object, operation: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr("ds_provider_tripletex_py_lib.read_info._load_metadata", _raise_not_found)
    with pytest.raises(ValidationError) as exc_info:
        get_read_info(TripletexProductName.CUSTOMER)
    assert "customer" in exc_info.value.message
    assert exc_info.value.details["product_name"] == "customer"


@pytest.mark.parametrize("missing_key", ["path", "fields", "pagination", "changed_since"])
def test_get_read_info_raises_on_missing_key(monkeypatch, missing_key):
    """A metadata payload missing a required key raises ValidationError."""
    payload = {"path": "customer", "fields": ["id"], "pagination": "offset", "changed_since": False}
    del payload[missing_key]
    monkeypatch.setattr(
        "ds_provider_tripletex_py_lib.read_info._load_metadata",
        lambda product_name, operation: payload,
    )
    with pytest.raises(ValidationError) as exc_info:
        get_read_info(TripletexProductName.CUSTOMER)
    assert "customer" in exc_info.value.message
    assert exc_info.value.details["product_name"] == "customer"


def test_get_read_info_defaults_explode_columns_when_absent(monkeypatch):
    """explode_columns is optional -- absent from the payload defaults to []."""
    payload = {"path": "customer", "fields": ["id"], "pagination": "offset", "changed_since": False}
    monkeypatch.setattr(
        "ds_provider_tripletex_py_lib.read_info._load_metadata",
        lambda product_name, operation: payload,
    )
    info = get_read_info(TripletexProductName.CUSTOMER)
    assert info.explode_columns == []


def test_get_read_info_raises_on_unrecognized_pagination_kind(monkeypatch):
    """An unrecognized pagination value raises ValidationError, not a silent fallback."""
    payload = {"path": "customer", "fields": ["id"], "pagination": "not-a-real-kind", "changed_since": False}
    monkeypatch.setattr(
        "ds_provider_tripletex_py_lib.read_info._load_metadata",
        lambda product_name, operation: payload,
    )
    with pytest.raises(ValidationError):
        get_read_info(TripletexProductName.CUSTOMER)
