"""Load only the dataset and detector inputs needed for signal calibration."""

from dataclasses import fields
from pathlib import Path

from data_tools.dataset_config import DatasetConfig, GeneratedDatasetParameters
from data_tools.detector.detector_config import DetectorConfig
from frame.file_system.textual_data import (
    expand_config_paths,
    load_config_params_from_paths,
)


class SignalCalibrationConfig(DatasetConfig, DetectorConfig):
    """Calibration does not require a training model or execution context."""

    def __init__(self, parameters: dict):
        DatasetConfig.__init__(
            self, dataset__definitions=parameters["dataset__definitions"]
        )
        DetectorConfig.__init__(
            self,
            **{
                field.name: parameters[field.name]
                for field in fields(DetectorConfig)
                if field.name in parameters
            },
        )

    @property
    def signal_dataset_parameters(self) -> GeneratedDatasetParameters:
        signals = [p for p in self.dataset_parameters if p.dataset__has_signal]
        if len(signals) != 1:
            raise ValueError(
                "Calibration requires exactly one configured signal dataset"
            )
        (parameters,) = signals
        if not isinstance(parameters, GeneratedDatasetParameters):
            raise TypeError("Analytic signal calibration requires a generated dataset")
        return parameters


def load_signal_calibration_config(paths: list[Path]) -> SignalCalibrationConfig:
    """Reuse the repository's ordered JSON/YAML configuration composition."""
    return SignalCalibrationConfig(
        load_config_params_from_paths(expand_config_paths(paths))
    )
