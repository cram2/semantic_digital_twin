import os

from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine


def persistent_database_available() -> bool:
    """
    Check if a persistent database is available for connection.

    This function validates if the environment variable `semantic_digital_twin_DATABASE_URI`
    is set and attempts to establish a connection with the database using the
    provided URI. If the connection is successful, it returns True, indicating the
    database is available. Otherwise, it returns False.

    :return: Indicates whether the persistent database is accessible
    """
    semantic_digital_twin_database_uri = os.environ.get(
        "semantic_digital_twin_DATABASE_URI"
    )
    if semantic_digital_twin_database_uri is None:
        return False

    try:
        engine = create_engine(f"{semantic_digital_twin_database_uri}")
        with engine.connect():
            ...
    except OperationalError as e:
        return False
    return True
