
from frame.command_line.handle_args import context_controlled_execution
from frame.config_handle import ExecutionContext
from plot.plotting_config import PlottingConfig


@context_controlled_execution
def create_plots(context: ExecutionContext):
    # Make sure we have a plot config
    if not isinstance(plotting_config := context.config, PlottingConfig):
        raise TypeError("The configuration must be a PlotConfig")
    
    # Draw all plots
    for plot in plotting_config:
        figure = plot()
        figure.savefig(str(context.unique_out_dir / f"{plot.name}.png"))


if __name__ == "__main__":
    create_plots()