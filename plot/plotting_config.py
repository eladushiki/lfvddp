from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from frame.file_structure import PLOT_FILE_EXTENSION
from train.train_config import TrainConfig


@dataclass
class PlotInstructions:
    """
    Class for structuring the data needed for a single plot instruction.
    """

    name: str
    instructions: Dict[str, Any]

    @property
    def plot_filename(self):
        return f"{self.name}.{PLOT_FILE_EXTENSION}"


@dataclass
class PlottingConfig:
    """
    Class for structuring all the data needed for plotting instructions.
    """

    # General plot settings
    ## Styling
    plot__pyplot_styling: Dict[str, Any]
    plot__figure_styling: Dict[str, Any]

    ## Sizing
    plot__figure_size: Tuple[int, int]

    # Additional settings for each plot
    plot__plot_specifications: List[Dict[str, Any]]

    @property
    def plot_instructions(self) -> List[PlotInstructions]:
        return [PlotInstructions(**spec) for spec in self.plot__plot_specifications]

    def __iter__(self):
        return iter(self.plot_instructions)
