# Force import of all subtypes of TrainConfig for correct dynamic class resolution
from train.train_config import TrainConfig

# Subtype list
from data_tools.histogram_generation.exp_histogram import exp
from data_tools.histogram_generation.gauss_histogram import gauss
from data_tools.histogram_generation.physics_histogram import physics
