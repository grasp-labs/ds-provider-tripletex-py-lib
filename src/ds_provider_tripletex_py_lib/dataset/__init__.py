"""
**File:** ``__init__.py``
**Region:** ``ds_provider_tripletex_py_lib/dataset``

Description
-----------
Dataset package for the Tripletex provider.
"""

from .tripletex import TripletexDataset, TripletexDatasetSettings, TripletexReadSettings

__all__ = ["TripletexDataset", "TripletexDatasetSettings", "TripletexReadSettings"]
