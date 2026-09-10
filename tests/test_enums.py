"""
**File:** ``test_enums.py``
**Region:** ``tests``

Unit tests for Tripletex provider enums.
"""

from ds_provider_tripletex_py_lib.enums import ResourceType, TripletexProductName


def test_resource_type_values():
    """ResourceType members expose the expected identifiers."""
    assert ResourceType.LINKED_SERVICE == "ds.resource.linked-service.tripletex"
    assert ResourceType.DATASET == "ds.resource.dataset.tripletex"


def test_tripletex_product_name_values_are_lowercase():
    """Every TripletexProductName member value is a lowercase string identifier."""
    for product_name in TripletexProductName:
        assert product_name.value == product_name.value.lower()
