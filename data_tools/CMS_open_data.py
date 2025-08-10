from logging import error
from typing import Any, List


def parse_CMS_open_data_sources_json(params: dict[str, Any]) -> List[str]:
    """
    Get the list of files of an Open CMS dataset sources format,
    as can be downloaded from the site.
    """
    try:
        return [
            file["uri"] for file in params["files"]
        ]
    except KeyError as ke:
        error(f"CMS open data sources file not according to template")
        raise ke
