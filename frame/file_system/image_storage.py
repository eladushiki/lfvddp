from pathlib import Path
from matplotlib.figure import Figure


def save_figure(figure: Figure, path: Path):
    figure.savefig(str(path))
