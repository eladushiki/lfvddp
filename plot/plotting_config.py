from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from matplotlib.figure import Figure

from frame.config_handle import Config
from plot.plots import *


@dataclass
class PlotInstructions:
    """
    Class for structuring the data needed for a single plot instruction.
    """

    PLOT_FUNCTIONS_BY_NAME = {  # todo: this can be dynamically retrieved from the module itself
        "Plot_Percentiles_ref": Plot_Percentiles_ref,
        "plot_t_distribution": plot_t_distribution,
        "plot_t_2distributions": plot_t_2distributions,
        "plot_t_multiple_distributions": plot_t_multiple_distributions,
        "exp_performance_plot": exp_performance_plot,
        "exp_multiple_performance_plot": exp_multiple_performance_plot,
        "em_performance_plot": em_performance_plot,
        "em_performance_plot_BR": em_performance_plot_BR,
        "em_luminosity_plot": em_luminosity_plot,
        "animated_t_distribution": animated_t_distribution,
        "animated_t_2distributions": animated_t_2distributions,
    }

    name: str
    instructions: Dict[str, Any]

    @property
    def __generating_function(self) -> Callable:
        try:
            return PlotInstructions.PLOT_FUNCTIONS_BY_NAME[self.name]
        except KeyError as ke:
            raise KeyError(f"Could not find a plot answering to the name: {self.name}")
        
    def __call__(self) -> Figure:
        return self.__generating_function(**self.instructions)

        
@dataclass
class PlottingConfig(Config, ABC):  # todo: this should probably not be direct inheritance
    """
    Class for structuring all the data needed for plotting instructions.
    """

    # General plot settings
    pass

    # Additional settings for each plot
    plot__plot_specifications: List[Dict[str, Any]]

    @property
    def _plot_instructions(self) -> List[PlotInstructions]:
        return [PlotInstructions(**spec) for spec in self.plot__plot_specifications]

    def __iter__(self):
        return iter(self._plot_instructions)
