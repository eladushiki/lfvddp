from time import time, strftime


def get_unix_timestamp() -> int:
    """
    Get the current unix timestamp.
    """
    return int(time())


def get_time_and_date_string() -> str:
    """
    Get the current time and date as a string.
    """
    return strftime(r"%Y%m%d_%H%M%S")
