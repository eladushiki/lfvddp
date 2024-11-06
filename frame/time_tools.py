from time import time


def get_unix_timestamp() -> int:
    """
    Get the current unix timestamp.
    """
    return int(time())
