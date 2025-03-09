
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

        image_filename = context.unique_out_dir / plot.plot_filename
        context.save_and_document_figure(figure, image_filename)


if __name__ == "__main__":
    create_plots()
