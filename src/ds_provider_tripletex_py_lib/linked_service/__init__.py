"""
**File:** ``__init__.py``
**Region:** ``ds_provider_tripletex_py_lib/linked_service``

Description
-----------
Linked service package for the Tripletex provider.
"""

from .tripletex import TripletexLinkedService, TripletexLinkedServiceSettings

__all__ = ["TripletexLinkedService", "TripletexLinkedServiceSettings"]
