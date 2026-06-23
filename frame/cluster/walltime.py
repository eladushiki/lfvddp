MAX_PBS_WALLTIME_HOURS = 72
MAX_PBS_WALLTIME_SECONDS = MAX_PBS_WALLTIME_HOURS * 60 * 60


def parse_walltime(walltime: str) -> int:
    parts = walltime.split(":")
    if len(parts) != 3:
        raise ValueError(f"Walltime must be in HH:MM:SS format, got {walltime!r}")

    hours, minutes, seconds = (int(part) for part in parts)
    if hours < 0 or minutes < 0 or seconds < 0:
        raise ValueError(f"Walltime values must be non-negative, got {walltime!r}")
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Walltime minutes and seconds must be below 60, got {walltime!r}")

    total_seconds = hours * 60 * 60 + minutes * 60 + seconds
    if total_seconds <= 0:
        raise ValueError(f"Walltime must be positive, got {walltime!r}")
    return total_seconds


def format_walltime(total_seconds: int) -> str:
    if total_seconds <= 0:
        raise ValueError(f"Walltime must be positive, got {total_seconds!r} seconds")

    hours, remainder = divmod(total_seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def split_walltime(
    walltime: str,
    max_seconds: int = MAX_PBS_WALLTIME_SECONDS,
) -> list[str]:
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")

    remaining_seconds = parse_walltime(walltime)
    chunks: list[str] = []
    while remaining_seconds > 0:
        chunk_seconds = min(remaining_seconds, max_seconds)
        chunks.append(format_walltime(chunk_seconds))
        remaining_seconds -= chunk_seconds

    return chunks
