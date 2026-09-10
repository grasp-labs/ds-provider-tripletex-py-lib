"""
**File:** ``__init__.py``
**Region:** ``ds-provider-tripletex-py-lib``

Description
-----------
A Python package from the ds-provider-tripletex-py-lib library.

Example
-------
.. code-block:: python

    from ds_provider_tripletex_py_lib import __version__

    print(f"Package version: {__version__}")
"""

from importlib.metadata import version

__version__ = version("ds-provider-tripletex-py-lib")
__all__ = ["__version__"]
