from inspect import isfunction
from types import FunctionType
from frame.context.execution_context import ExecutionContext
from plot.plotting_config import PlotInstructions

import plot.plots as plots

class PlotFactory:
    _instance = None
    _context: ExecutionContext

    def __new__(cls, context: ExecutionContext):
        if not cls._instance:
            cls._instance = super(PlotFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self, context: ExecutionContext):
        self._context = context

    @property
    def plot_functions_by_name(self):
        return {
            name: function
            for name in dir(plots)
            if isfunction(function := getattr(plots, name))
        }
    
    def __getitem__(self, plot_name: str) -> FunctionType:
        try:
            return self.plot_functions_by_name[plot_name]
        except KeyError as ke:
            raise KeyError(f"Could not find a plot answering to the name: {plot_name}")

    def generate_plot(self, plot_instructions: PlotInstructions):
        generating_function = self[plot_instructions.name]
        return generating_function(self._context, **plot_instructions.instructions)
