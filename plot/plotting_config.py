from dataclasses import dataclass, field
from enum import Enum
from types import FunctionType
from typing import Any, Callable, Dict, List, Tuple

from frame.file_structure import PLOT_FILE_EXTENSION


DEFAULT_PYPLOT_STYLING = {
    "rcParams": {"font.family": "serif", "font.size": 24},
    "style.use": "classic",
}
DEFAULT_FIGURE_STYLING = {
    "patch_set_facecolor": "white",
    "plot": {
        "histogram_color": "plum",
        "edge_color": "darkorchid",
        "chi2_color": "grey",
        "linewidth": 5,
        "alpha": 0.8,
    },
}
DEFAULT_FIGURE_SIZE = (10, 9)


class PlotScope(Enum):
    SINGLE_SUBMISSION = "single_submission"
    MULTI_RUN = "multi_run"


def plot_for_scope(scope: PlotScope) -> Callable[[FunctionType], FunctionType]:
    """Declare the execution scope of a plot next to its implementation."""
    def decorate(plot_function: FunctionType) -> FunctionType:
        plot_function.plot_scope = scope
        return plot_function

    return decorate


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

    # Plot specifications intentionally remain explicit: they define the
    # requested output rather than a communal styling default.
    plot__plot_specifications: List[Dict[str, Any]]

    # General plot settings
    ## Styling
    plot__pyplot_styling: Dict[str, Any] = field(
        default_factory=lambda: {
            "rcParams": DEFAULT_PYPLOT_STYLING["rcParams"].copy(),
            "style.use": DEFAULT_PYPLOT_STYLING["style.use"],
        }
    )
    plot__figure_styling: Dict[str, Any] = field(
        default_factory=lambda: {
            "patch_set_facecolor": DEFAULT_FIGURE_STYLING["patch_set_facecolor"],
            "plot": DEFAULT_FIGURE_STYLING["plot"].copy(),
        }
    )

    ## Sizing
    plot__figure_size: Tuple[int, int] = DEFAULT_FIGURE_SIZE

    plot__prediction_process_number_of_bins: int = 30
    # Normalize every upper data/prediction histogram independently to unit probability.
    plot__prediction_process_normalize_each_prediction: bool = True

    # Shared layout values used by Carpenter for every plot.
    plot__run_stamp_row_height: float = 0.12
    plot__run_stamp_y: float = 0.02
    plot__run_stamp_font_size: int = 10
    plot__standard_left_border: float = 0.125
    plot__standard_right_border: float = 0.9
    plot__standard_top_border: float = 0.88

    @property
    def plot_instructions(self) -> List[PlotInstructions]:
        return [PlotInstructions(**spec) for spec in self.plot__plot_specifications]

    def __iter__(self):
        return iter(self.plot_instructions)
