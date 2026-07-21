from inspect import isfunction
from types import FunctionType

from matplotlib.figure import Figure

from data_tools.detector.detector_config import DetectorConfig
from frame.context.execution_context import ExecutionContext
import plot.plots as plots
from plot.plotting_config import PlotInstructions, PlottingConfig


class PlotFactory:
    _instance = None
    _context: ExecutionContext

    def __new__(cls, context: ExecutionContext):
        if not cls._instance:
            cls._instance = super(PlotFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self, context: ExecutionContext):
        self._context = context

        if not isinstance(config := context.config, PlottingConfig):
            raise TypeError(
                "Can't instantiate a PlotFactory without a PlottingConfig, "
                f"got {type(config)}"
            )

        self._config = config

    @property
    def plot_functions_by_name(self):
        return {
            name: function
            for name in dir(plots)
            if isfunction(function := getattr(plots, name))
        }

    def __getitem__(self, plot_name: str) -> FunctionType:
        plot_functions = self.plot_functions_by_name
        if plot_name in plot_functions:
            return plot_functions[plot_name]

        if not isinstance(self._config, DetectorConfig):
            raise KeyError(
                f"Could not find plot '{plot_name}' and cannot infer a dimensional "
                "variant without a DetectorConfig."
            ) from None

        required_dimensions = min(2, self._config.detector__number_of_dimensions)
        if required_dimensions <= 0:
            raise ValueError(
                f"Cannot resolve dimensional plot '{plot_name}' without configured "
                "detector observables."
            )
        dimension_specific_name = f"{plot_name}_{required_dimensions}d"
        try:
            return plot_functions[dimension_specific_name]
        except KeyError:
            raise KeyError(
                f"Could not find a {required_dimensions}D implementation for plot "
                f"'{plot_name}'. Expected function '{dimension_specific_name}'."
            ) from None

    def generate_plot(self, plot_instructions: PlotInstructions) -> Figure:
        generating_function = self[plot_instructions.name]
        return generating_function(self._context, **plot_instructions.instructions)
