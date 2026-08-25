# ds-provider-tripletex-py-lib

![Python Versions](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue)
[![PyPI version](https://badge.fury.io/py/ds-provider-tripletex-py-lib.svg?kill_cache=1)](https://badge.fury.io/py/ds-provider-tripletex-py-lib)
[![Build Status](https://github.com/grasp-labs/ds-provider-tripletex-py-lib/actions/workflows/build.yaml/badge.svg)](https://github.com/grasp-labs/ds-provider-tripletex-py-lib/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/grasp-labs/ds-provider-tripletex-py-lib/graph/badge.svg?token=EO3YCNCZFS)](https://codecov.io/gh/grasp-labs/ds-provider-tripletex-py-lib)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

DS package for ds-provider-tripletex-py-lib

## Tripletex Products

This provider ships a fixed catalog of Tripletex products, each backed by a
packaged metadata file at
`src/ds_provider_tripletex_py_lib/assets/<product_name>/read/metadata.json`.
Each file declares everything needed to read that product -- the API path,
how pagination works, whether Tripletex's `changedSince` incremental filter
is available, and which fields to request. See
`src/ds_provider_tripletex_py_lib/read_info.py`'s module docstring for the
full schema and selector syntax, and
`src/ds_provider_tripletex_py_lib/enums.py`'s `TripletexProductName` for the
current catalog.

### Reading data that isn't in the catalog yet

A caller doesn't have to wait for a product to be packaged. Set
`TripletexReadSettings.path`/`fields`/`pagination` directly instead of
`TripletexDatasetSettings.product_name`:

```python
from ds_provider_tripletex_py_lib.dataset.tripletex import (
    TripletexDatasetSettings,
    TripletexReadSettings,
)
from ds_provider_tripletex_py_lib.enums import PaginationKind

settings = TripletexDatasetSettings(
    read=TripletexReadSettings(
        path="some/endpoint",
        fields=["id", "name"],
        pagination=PaginationKind.OFFSET,
        changed_since=False,
    ),
)
```

### Adding query parameters / filters

Extra query parameters (e.g. `isInactive`) go in `TripletexReadSettings.params`,
merged into every request for any `product_name`/`path`. Exception:
`date_window` products (`ledger/posting`, `balance_sheet`) auto-generate
`dateFrom`/`dateTo` per window, overriding any values passed here.

```python
TripletexReadSettings(params={"isInactive": True})
```

This is provider-side, not something you package in `metadata.json` -- the
metadata file only declares the *product's own* fixed shape (path,
pagination, fields, `changed_since`), not per-run filters.

### Adding a new packaged product

To make a product part of the permanent catalog (so callers can use
`product_name=` instead of `path`/`fields`/`pagination`):

1. **Find the real endpoint in Tripletex's docs.** Open
   `https://tripletex.no/v2-docs/#/` and find the endpoint's `get`
   operation.

   Note:
   - The exact path (some are nested, e.g. `ledger/account`,
     `ledger/vatType` -- match Tripletex's exact casing).
   - Its declared `parameters` -- does it support `changedSince`? Any
     filters worth documenting as candidates for `read.params`?
   - Whether it requires a bounded date range to return anything at all
     (like `balanceSheet`'s mandatory `dateFrom`/`dateTo`).

2. **Add the enum member** in
   `src/ds_provider_tripletex_py_lib/enums.py`'s `TripletexProductName` --
   lowercase, matching the asset directory name you create next.

3. **Add the packaged metadata file** at
   `src/ds_provider_tripletex_py_lib/assets/<product_name>/read/metadata.json`.
   The first four keys are required -- there's no default for any of them;
   `explode_columns` is optional (default `[]`, only needed by a handful of
   products):

   ```json
   {
     "path": "customer",
     "pagination": "offset",
     "changed_since": true,
     "fields": ["id", "name", {"accountManager": ["id"]}]
   }
   ```

   - `path`: the API path segment (no leading slash), from step 1.
   - `pagination`: `"date_window"` if the endpoint has `dateFrom`/`dateTo`
     params, else `"offset"`.
   - `changed_since`: `true` only if step 1 found a real, working
     `changedSince` parameter (verify against the live API).
   - `fields`: the field projection -- a plain field name (`str`), or a
     one-key `dict` (`{"group": [...]}`) expanding a nested object or
     sub-collection.
   - `explode_columns` (optional): field names that are a *list* of objects
     rather than a single nested object, e.g. `supplier`'s
     `bankAccountPresentation` (a supplier can have several bank accounts).
     A plain nested object flattens into columns automatically; a list
     can't, so it's exploded into one row per element instead. Check the
     live API's actual response shape for a field before deciding -- most
     nested fields are single objects and don't need this.

4. **Verify against the live API before trusting the spec.** The OpenAPI
   spec can be silent on real behavior -- this provider's own history
   includes several things the spec didn't make obvious: an endpoint
   requiring a literal `>` prefix, `changedSince`'s exact expected date
   format only surfacing via its own validation error message, and
   `versionDigest` staying identical across different pages of the same
   query (confirmed by direct testing, not assumed from the docs' wording).
   A real request is the only way to be sure:

   ```bash
   curl -s -X GET "https://tripletex.no/v2/<path>" -G \
       --data-urlencode "from=0" --data-urlencode "count=1" \
       --data-urlencode "fields=id,name" \
       -H "Authorization: Basic <base64(client_id:session_token)>"
   ```

   See `src/ds_provider_tripletex_py_lib/linked_service/tripletex.py`'s
   `connect()` for how the session token and Basic auth header are built.

5. **Add a `_lite` variant only if it's genuinely useful** -- a smaller
   field projection over the *same* `path` as an existing product (see
   `customer_lite`, `supplier_lite`), not a new endpoint.

6. **Run the test suite.** `tests/test_read_info.py`'s
   `test_every_product_has_read_assets` and its neighbors iterate every
   `TripletexProductName` member automatically, so a new product is
   exercised by the existing suite without writing new per-product tests.

## Quick Start

### Quick Setup

```shell
# 1. Install dependencies
uv sync --all-extras --dev

# 2. Install pre-commit hooks
uv run pre-commit install

# 3. Verify setup
make test
```

## Development

### Available Commands

Use the Makefile for all development tasks:

```shell
# Show all available commands
make help

# Code Quality
make lint           # Check code quality with ruff
make format         # Format code with ruff
make type-check     # Run mypy type checking
make security-check # Run security checks with bandit

# Testing
make test          # Run tests
make test-cov      # Run tests with coverage (requires 95%)

# Build and Publish
make build         # Build package
make docs          # Build documentation
make publish-test  # Upload to TestPyPI
make publish       # Upload to PyPI
```

### Version Management

```shell
# Show current version
make version

# Tag and release
make tag           # Create git tag and push (triggers release)
```

> **⚠️ Warning**: The `make tag` command will create a git tag and
> push it to the remote repository, which may trigger automated
> releases. Ensure you have updated `pyproject.toml` with the new version
> and committed all changes before running this command.

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality:

```shell
install pre-commit
```

### Building Documentation

```shell
# Build documentation
make docs

# View documentation (macOS)
open docs/build/html/index.html

# View documentation (Linux)
xdg-open docs/build/html/index.html
```

### Testing

```shell
# Run basic tests
make test

# Run tests with coverage (requires 95% coverage)
make test-cov

# Run specific test file
uv run pytest tests/test_example.py -v
```

## Project Structure

```text
.
├── .config/                   # Configuration tooling files
├── .github/
│   ├── workflows/            # CI/CD workflows
│   └── CODEOWNERS            # Code ownership file
├── src/
│   └── ds_provider_tripletex_py_lib/     # Rename to your module name
│       └── __init__.py
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── tests/                    # Test files
├── docs/                     # Sphinx documentation
├── LICENSE-APACHE            # License file
├── pyproject.toml            # Project configuration
├── Makefile                  # Development commands
├── codecov.yaml              # Codecov configuration
├── CONTRIBUTING.md           # Contribution guidelines
├── PyPI.md                   # PyPI publishing guide
├── README.md                 # This file
```

## Features

- **Modern Python Tooling**: Uses `uv` for fast dependency management
- **Type Safety**: Strict mypy configuration with full type hints
- **Code Quality**: Ruff for linting and formatting
- **Testing**: Pytest with 95% coverage requirement
- **Documentation**: Sphinx with autoapi for automatic API docs
- **CI/CD**: GitHub Actions for testing, building, and publishing
- **Pre-commit Hooks**: Automated code quality checks
- **Docker Support**: Containerized build environment

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Make (for development commands)

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [PyPI.md](PyPI.md) - PyPI publishing guide
- [README.md](README.md) - This file

## License

This package is licensed under the Apache License 2.0.
See [LICENSE-APACHE](LICENSE-APACHE) for details.
