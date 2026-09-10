"""
**File:** ``enums.py``
**Region:** ``ds_provider_tripletex_py_lib/enums``

Constants for the Tripletex provider.

Example:
    >>> ResourceType.LINKED_SERVICE
    'ds.resource.linked-service.tripletex'
    >>> ResourceType.DATASET
    'ds.resource.dataset.tripletex'
    >>> TripletexProductName.CUSTOMER
    'customer'
"""

from enum import StrEnum


class ResourceType(StrEnum):
    """
    Constants for Tripletex provider resource identifiers.
    """

    LINKED_SERVICE = "ds.resource.linked-service.tripletex"
    DATASET = "ds.resource.dataset.tripletex"


class OperationType(StrEnum):
    """
    Dataset operations a product's packaged assets may define.

    Only ``READ`` is implemented -- this provider is read-only.
    """

    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class PaginationKind(StrEnum):
    """
    How a Tripletex product's ``read`` requests are paginated.

    Declared per product in ``assets/<product_name>/read/metadata.json``.
    """

    OFFSET = "offset"
    """Plain ``from``/``count`` offset pagination -- the default for most entities."""

    DATE_WINDOW = "date_window"
    """Rolling date-window pagination, required by the ``ledger/posting`` endpoint."""


class TripletexProductName(StrEnum):
    """
    Tripletex product names available through the Tripletex v2 API.

    Each value names an ``assets/<product_name>/read/metadata.json`` file loaded
    via ``read_info.get_read_info()``, which defines the API path and
    field projection for that product. The ``_lite`` variants read from the
    same endpoint as their full counterpart but project a smaller field set.
    """

    CUSTOMER = "customer"
    CUSTOMER_LITE = "customer_lite"
    DEPARTMENT = "department"
    DEPARTMENT_LITE = "department_lite"
    EMPLOYEE = "employee"
    EMPLOYEE_LITE = "employee_lite"
    LEDGER_ACCOUNT = "ledger_account"
    LEDGER_ACCOUNT_LITE = "ledger_account_lite"
    LEDGER_POSTING = "ledger_posting"
    LEDGER_VAT_TYPE = "ledger_vat_type"
    PRODUCT = "product"
    PROJECT = "project"
    PROJECT_LITE = "project_lite"
    SUPPLIER = "supplier"
    SUPPLIER_LITE = "supplier_lite"
    SUPPLIER_BANK_ACCOUNTS_LITE = "supplier_bank_accounts_lite"
    CURRENCY = "currency"
    BALANCE_SHEET = "balance_sheet"
