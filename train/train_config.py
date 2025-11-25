from dataclasses import dataclass
from logging import warning
from typing import List


@dataclass
class TrainConfig:
    ## Training for nuisance parameters
    # Correction - what should be taken into account about the nuisance parameters?
    # - "SHAPE" - both normalization and shape uncertainties are considered
    # - "NORM" - only normalization uncertainties are considered
    # - "" - systematic uncertainties are neglected (simple NPLM is run - no Delta calculation and Tau is calculated without nuisance parameters)
    train__nuisance_correction_types: str      # "SHAPE", "NORM" or "". Which compensations for uncertainties to use.
    train__data_is_train_for_nuisances: bool  # Should the nuisance change or stick with initial values?

    # Recovery of nuisances parameters
    train__shape_nuisance_std: float             # shape nuisance sigma
    train__shape_nuisance_mean: float       # shape nuisance reference, in terms of std
    train__shape_nuisance_reference: float  # norm nuisance reference, in terms of std
    
    train__norm_nuisance_std: float              # norm nuisance sigma
    train__norm_nuisance_mean: float        # in terms of std
    train__norm_nuisance_reference: float   # in terms of std

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
        # Calculate total trainable parameters (weights + biases) in the dense NN
        # For architecture [n0, n1, n2, ...], params = sum over layers i: (n[i] * n[i+1] + n[i+1])
        architecture = self.train__nn_architecture
        total_params = sum(
            architecture[i] * architecture[i+1] + architecture[i+1]
            for i in range(len(architecture) - 1)
        )
        return total_params - 1  # The substraction is due to the argument about another constraint on the DoF in our paper

    train__like_NPLM: bool = False  # Should we trian with NPLM's train_model or tensorflow's model.fit like LFVNN

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.train__epochs < 1e5 and self.train__like_NPLM or \
                self.train__epochs < 5e5 and not self.train__like_NPLM:
            warning("Training epochs not sufficient, train may not converge")

        if not self.train__like_NPLM and \
                (self.train__nuisance_correction_types != "" or self.train__data_is_train_for_nuisances):
            warning("You probably meant to mimic LFVNN, but it does not deal with nuisances.")
