"""
**File:** ``01_linked_service_connect.py``
**Region:** ``examples/01_linked_service_connect``

Example 01: Connect to Tripletex using a linked service.

This example demonstrates how to:
- Create a Tripletex linked service with consumer/employee tokens
- Connect and test the connection to Tripletex
- Clean up the connection

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

from ds_provider_tripletex_py_lib.linked_service.tripletex import (
    TripletexLinkedService,
    TripletexLinkedServiceSettings,
)

Logger.configure(level=logging.DEBUG)
logger = Logger.get_logger(__name__)


def main() -> None:
    """Main function demonstrating Tripletex linked service connection."""
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

    try:
        logger.info("Connecting to Tripletex...")
        linked_service.connect()
        logger.info("✓ Connected successfully!")

        logger.info("Testing connection to Tripletex...")
        success, message = linked_service.test_connection()

        if success:
            logger.info("✓ Connection test successful!")
        else:
            logger.error("✗ Connection test failed: %s", message)

    except Exception as exc:
        logger.error("Failed to connect to Tripletex: %s", exc)
        raise

    finally:
        linked_service.close()
        logger.info("Connection closed")


if __name__ == "__main__":
    main()
