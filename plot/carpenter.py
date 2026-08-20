from frame.context.execution_context import ExecutionContext
from plot.plotting_config import PlottingConfig
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import rcParams


class Carpenter:
    """Create figures with a consistent, crop-safe run-stamp row."""

    RUN_STAMP_ROW_HEIGHT = 0.12
    RUN_STAMP_Y = 0.02
    RUN_STAMP_FONT_SIZE = 10
    STANDARD_LEFT_BORDER = 0.125
    STANDARD_RIGHT_BORDER = 0.9
    STANDARD_TOP_BORDER = 0.88
    _instance = None

    def __new__(cls, context: ExecutionContext):
        if not cls._instance:
            cls._instance = super(Carpenter, cls).__new__(cls)
        return cls._instance

    def __init__(self, context: ExecutionContext):
        self._context = context

        if not isinstance(config := context.config, PlottingConfig):
            raise TypeError(f"Can't instantiate a Carpenter without a PlottingConfig, got {type(config)}")
        
        self._config = config
        self.initialize_styling()

        self._figure_styling = self._config.plot__figure_styling

    def initialize_styling(self):
        """
        Everything about styling that should be configured once per run
        """
        try:
            rcParams.update(self._config.plot__pyplot_styling["rcParams"])
        except KeyError:
            pass

        plt.style.use(self._config.plot__pyplot_styling["style.use"])

    def figure(self) -> Figure:
        fig = plt.figure(figsize=self._config.plot__figure_size)

        # Apply styling
        fig.patch.set_facecolor(self._figure_styling["patch_set_facecolor"])

        # Stamp for run
        fig.text(
            x=0,
            y=self.RUN_STAMP_Y,
            s=f"run hash: {self._context.run_hash}",
            fontsize=self.RUN_STAMP_FONT_SIZE,
            verticalalignment="bottom",
            horizontalalignment="left",
        )
        self.reserve_run_stamp_row(fig)

        return fig

    @staticmethod
    def reserve_run_stamp_row(fig: Figure, **subplot_adjustments) -> None:
        """Reserve a crop-safe bottom row exclusively for the run stamp."""
        requested_bottom = subplot_adjustments.pop("bottom", 0.0)
        fig.subplots_adjust(
            bottom=max(requested_bottom, Carpenter.RUN_STAMP_ROW_HEIGHT),
            **subplot_adjustments,
        )

    @classmethod
    def standardize_plot_borders(cls, fig: Figure) -> None:
        """Apply the common one-panel plot borders after all artists are added."""
        fig.subplots_adjust(
            left=cls.STANDARD_LEFT_BORDER,
            right=cls.STANDARD_RIGHT_BORDER,
            bottom=cls.RUN_STAMP_ROW_HEIGHT,
            top=cls.STANDARD_TOP_BORDER,
        )
