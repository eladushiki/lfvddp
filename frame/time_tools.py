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


def get_unique_run_dir_name() -> str:
    """
    Get a unique run directory name.
    """
    return f"run_{get_time_and_date_string()}_{time():.6f}"
