from frame.context.execution_context import ExecutionContext
from plot.plotting_config import PlottingConfig
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import rcParams

class Carpenter:
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

        return fig
