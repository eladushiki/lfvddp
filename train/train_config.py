from dataclasses import dataclass
from typing import List

from neural_networks.NPLM.src.NPLM.PLOTutils import compute_df


@dataclass
class TrainConfig:
    ## Training for nuisance parameters
    # Correction - what should be taken into account about the nuisance parameters?
    # - "SHAPE" - both normalization and shape uncertainties are considered
    # - "NORM" - only normalization uncertainties are considered
    # - "" - systematic uncertainties are neglected (simple NPLM is run - no Delta calculation and Tau is calculated without nuisance parameters)
    train__nuisance_correction: str  # "SHAPE", "NORM" or "".
    @property
    def train__data_is_train_for_nuisances(self) -> bool:
        return self.train__nuisance_correction != ""

    ## Training parameters
    train__epochs: int
    train__number_of_epochs_for_checkpoint: int

    # NN parameters
    train__nn_weight_clipping: float

    train__nn_input_dimension: int
    train__nn_inner_layer_nodes: int
    train__nn_output_dimension: int
    @property
    def train__nn_architecture(self) -> List[int]:
        return [self.train__nn_input_dimension, self.train__nn_inner_layer_nodes, self.train__nn_output_dimension]
    @property
    def train__nn_degrees_of_freedom(self) -> int:
        return compute_df(
            input_size=self.train__nn_input_dimension,
            hidden_layers=self.train__nn_architecture[1:-1],
            output_size=self.train__nn_output_dimension,
        ) - 1  # The substraction is due to the argument about another constraint on the DoF in our paper
