"""Require the same fitting procedure for a calibrated significance comparison."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path

from data_tools.detector.detector_config import DetectorConfig
from train.comparison_datasets import dataset_comparison_values
from train.train_config import TrainConfig


class ComparisonConfigError(ValueError):
    """A reference and its signal runs use different fitting configurations."""


def validate_comparison_configs(configs: Sequence[tuple[str, Mapping]]) -> None:
    """Compare every training and detector field, including future declared fields.

    Inputs are resolved configuration dictionaries, as persisted in run contexts.
    Dataset settings must agree per region except for injected signal strength
    and shape, which define the plotted points and curves. Seeds, array sizes,
    output paths and plotting controls are outside this fitting signature. No fitting field is allowlisted
    away: checkpoint cadence and optimizer phase controls must match too.
    """
    if not configs:
        raise ComparisonConfigError("Significance comparison has no configurations")

    def signature(label, config):
        for required in (
            field.name
            for schema in (TrainConfig, DetectorConfig)
            for field in fields(schema)
        ):
            if required not in config:
                raise ComparisonConfigError(f"{label}: missing resolved {required}")
        if "dataset__definitions" not in config:
            raise ComparisonConfigError(f"{label}: missing dataset__definitions")
        try:
            dataset_values = dataset_comparison_values(config)
        except (ValueError, KeyError) as error:
            raise ComparisonConfigError(f"{label}: {error}") from error
        return dataset_values | {
            key: json.loads(json.dumps(value, sort_keys=True))
            for key, value in config.items()
            if key.startswith(
                (
                    "train__",
                    "detector__",
                    "dataset__",
                    "dataset_loaded__",
                    "dataset_generated__",
                )
            )
            and key != "dataset__definitions"
        }

    reference_name, reference_config = configs[0]
    reference = signature(reference_name, reference_config)
    differences = []
    for name, config in configs[1:]:
        candidate = signature(name, config)
        for key in sorted(reference.keys() | candidate.keys()):
            before = reference.get(key, "<missing>")
            after = candidate.get(key, "<missing>")
            if before != after:
                differences.append(f"{name}: {key}: {before!r} -> {after!r}")
    if differences:
        raise ComparisonConfigError(
            f"Incompatible significance comparison against {reference_name}:\n"
            + "\n".join(differences)
        )


def saved_comparison_configs(directory: Path) -> list[tuple[str, Mapping]]:
    """Read staged and available worker configs without loading model artifacts."""
    paths = [directory / "context.json"]
    paths.extend(sorted(directory.glob("*_run_of_single_train.py_*/context.json")))
    return [(str(path), json.loads(path.read_text())["config"]) for path in paths]
