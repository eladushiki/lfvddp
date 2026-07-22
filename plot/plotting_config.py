from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from frame.file_structure import PLOT_FILE_EXTENSION


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

    plot__target_run_parent_directory: str

    # General plot settings
    ## Styling
    plot__pyplot_styling: Dict[str, Any]
    plot__figure_styling: Dict[str, Any]

    ## Sizing
    plot__figure_size: Tuple[int, int]

    # Additional settings for each plot
    plot__plot_specifications: List[Dict[str, Any]] = field(default_factory=list)
    plot__prediction_process_number_of_bins: int = 30
    # Normalize every upper data/prediction histogram independently to unit probability.
    plot__prediction_process_normalize_each_prediction: bool = True

    @property
    def plot_instructions(self) -> List[PlotInstructions]:
        return [PlotInstructions(**spec) for spec in self.plot__plot_specifications]

    def __iter__(self):
        return iter(self.plot_instructions)
