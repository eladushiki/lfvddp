"""Dataset invariants and signal-shape grouping for significance comparisons."""

import json
from collections.abc import Mapping
from dataclasses import MISSING, fields

from data_tools.data_utils import DataSet
from data_tools.dataset_config import DatasetConfig
from data_tools.event_generation.distribution import (
    generator_selection_as_config,
    normalize_generator_selection,
)

# These are configuration definitions, not sampled DatasetParameters: an explicitly
# fixed background count is a scientific setting and must not be discarded.
SIGNAL_STRENGTH_FIELDS = DatasetConfig.SIGNAL_EVENT_CONFIGURATION_FIELDS | (
    DatasetConfig.RUN_VARIATION_FIELDS - {"dataset__number_of_background_events"}
)
SIGNAL_SHAPE_FIELDS = (
    DatasetConfig.SIGNAL_CONFIGURATION_FIELDS
    - DatasetConfig.SIGNAL_EVENT_CONFIGURATION_FIELDS
) | {"dataset_loaded__signal_observable_names"}


def dataset_comparison_values(config: Mapping, *, include_signal_shape=False) -> dict:
    """Normalize declared defaults without sampling Poisson counts or reading data.

    Regions are keyed by category, so definition ordering cannot change matching.
    All non-signal settings must match; signal strength varies within a curve,
    while the retained signal shape defines which curve a point belongs to.
    """
    definitions = config["dataset__definitions"]
    if not definitions:
        raise ValueError("Missing dataset definitions for significance comparison")
    schemas = DatasetConfig(dataset__definitions=[])._dataset__types
    values = {}
    categories = set()
    ignored = SIGNAL_STRENGTH_FIELDS | (
        set() if include_signal_shape else SIGNAL_SHAPE_FIELDS
    )
    for definition in definitions:
        category = DataSet.DataSetCategory.from_string(definition["category"])
        if category == DataSet.DataSetCategory.UNDEFINED or category in categories:
            raise ValueError(
                f"Invalid or duplicate dataset category: {definition['category']}"
            )
        categories.add(category)
        schema = schemas[definition["type"]]
        declared = {field.name: field for field in fields(schema) if field.init}
        unknown = definition.keys() - declared.keys()
        if unknown:
            raise ValueError(f"Unknown dataset configuration fields: {sorted(unknown)}")
        for key, field in declared.items():
            if key in ignored or key == "category":
                continue
            if key in definition:
                value = definition[key]
            elif field.default is not MISSING:
                value = field.default
            elif field.default_factory is not MISSING:
                value = field.default_factory()
            else:
                raise ValueError(f"Missing dataset field {category.name}.{key}")
            if (
                key
                in (
                    "dataset_generated__background_generator",
                    "dataset__signal_generator",
                )
                and value is not None
            ):
                value = generator_selection_as_config(
                    normalize_generator_selection(value)
                )
            values[f"dataset__definitions.{category.name}.{key}"] = value
    return values


def signal_curve_key(config: Mapping) -> str:
    """Use exactly the allowed signal-shape variation to separate curves."""
    return json.dumps(
        dataset_comparison_values(config, include_signal_shape=True), sort_keys=True
    )
