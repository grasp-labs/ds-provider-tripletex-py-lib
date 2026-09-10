"""
**File:** ``02_dataset_read.py``
**Region:** ``examples/02_dataset_read``

Example 02: Read data from Tripletex using a dataset.

This example demonstrates how to:
- Create a Tripletex linked service and connect
- Create a Tripletex dataset for a specific product
- Read customer data from the Tripletex API
- Persist and reuse the checkpoint for incremental loads

Prerequisites:
    Set environment variables or provide credentials directly:
    - TRIPLETEX_CONSUMER_TOKEN: Your Tripletex consumer token
    - TRIPLETEX_EMPLOYEE_TOKEN: Your Tripletex employee token
"""

from __future__ import annotations

import logging
import os
from uuid import uuid4

from ds_common_logger_py_lib import Logger

from ds_provider_tripletex_py_lib.dataset.tripletex import (
    TripletexDataset,
    TripletexDatasetSettings,
    TripletexReadSettings,
)
from ds_provider_tripletex_py_lib.enums import TripletexProductName
from ds_provider_tripletex_py_lib.linked_service.tripletex import (
    TripletexLinkedService,
    TripletexLinkedServiceSettings,
)

Logger.configure(level=logging.DEBUG)
logger = Logger.get_logger(__name__)


def main() -> None:
    """Main function demonstrating Tripletex dataset read operations."""
    consumer_token = os.getenv("TRIPLETEX_CONSUMER_TOKEN", "your-consumer-token")
    employee_token = os.getenv("TRIPLETEX_EMPLOYEE_TOKEN", "your-employee-token")

    linked_service = TripletexLinkedService(
        settings=TripletexLinkedServiceSettings(
            consumer_token=consumer_token,
            employee_token=employee_token,
        ),
        id=uuid4(),
        name="Tripletex Linked Service",
        description="Linked service for connecting to the Tripletex API",
        version="1.0",
    )

    dataset = TripletexDataset(
        id=uuid4(),
        name="Tripletex Customers Dataset",
        version="1.0",
        linked_service=linked_service,
        settings=TripletexDatasetSettings(
            product_name=TripletexProductName.CUSTOMER,
            read=TripletexReadSettings(count=1000),
        ),
    )

    try:
        logger.info("Connecting to Tripletex...")
        linked_service.connect()
        logger.info("✓ Connected successfully!")

        # Reuse a persisted checkpoint here to only re-fetch changed data
        # (an empty/omitted checkpoint performs a full load):
        # dataset.checkpoint = state_store.load(dataset_id)

        logger.info("Reading customer data from Tripletex...")
        dataset.read()

        if dataset.output is not None and not dataset.output.empty:
            logger.info("✓ Read %d customers", len(dataset.output))
            logger.debug("Columns: %s", list(dataset.output.columns))
            logger.debug("First few rows:\n%s", dataset.output.head())
        else:
            logger.info("No customer data returned (nothing changed since the last run)")

        # Persist the checkpoint for the next run:
        # state_store.save(dataset_id, dataset.checkpoint)
        logger.debug("Checkpoint for next run: %s", dataset.checkpoint)

    except Exception as exc:
        logger.error("Failed to read data: %s", exc)
        raise

    finally:
        dataset.close()
        logger.info("Connection closed")


if __name__ == "__main__":
    main()
