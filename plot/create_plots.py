
from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from plot.plot_factory import PlotFactory
from plot.plotting_config import PlottingConfig


@context_controlled_execution
def create_plots(context: ExecutionContext):
    # Make sure we have a plot config
    if not isinstance(plotting_config := context.config, PlottingConfig):
        raise TypeError("The configuration must be a PlotConfig")
    
    # Draw all plots
    plot_factory = PlotFactory(context=context)
    for plot in plotting_config:
        figure = plot_factory.generate_plot(plot)

        plot_filename = context.unique_out_dir / f"{plot.name}.png"  # todo: extract hardcoded file extension
        figure.savefig(plot_filename)
        context.document_created_product(plot_filename)  # todo: this should not be possible to forget


if __name__ == "__main__":
    create_plots()
