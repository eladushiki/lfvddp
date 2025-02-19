from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List

from train.train_config import TrainConfig


@dataclass
class PlotInstructions:
    """
    Class for structuring the data needed for a single plot instruction.
    """

    name: str
    instructions: Dict[str, Any]


@dataclass
class PlottingConfig(TrainConfig, ABC):
    """
    Class for structuring all the data needed for plotting instructions.
    """

    # General plot settings
    pass

    # Additional settings for each plot
    plot__plot_specifications: List[Dict[str, Any]]

    @property
    def plot_instructions(self) -> List[PlotInstructions]:
        return [PlotInstructions(**spec) for spec in self.plot__plot_specifications]

    def __iter__(self):
        return iter(self.plot_instructions)
