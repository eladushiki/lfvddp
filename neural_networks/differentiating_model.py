import keras
import numpy as np
from tensorflow import Variable

from data_tools.data_utils import DataSet
from frame.context.execution_context import ExecutionContext
from train.train_config import TrainConfig


class DifferentiatingModel(keras.models.Model):
    def __init__(
        self,
        context: ExecutionContext,
        name: str,
        bin_edges: np.ndarray,
        bin_centers: np.ndarray,
        **kwargs
    ):
        self._config: TrainConfig = context.config

        # Add layers by spec
        input_layer = keras.Input(shape=(self._config.train__nn_input_dimension,))
        last_layer = input_layer
        for secondary_layer_size in self._config.train__nn_architecture[1:]:
            layer = keras.layers.Dense(
                secondary_layer_size,
                activation='relu',
            )(last_layer)
            last_layer = layer
        
        # Build model
        super(DifferentiatingModel, self).__init__(
            name=name,
            inputs=input_layer,
            outputs=last_layer,
            **kwargs
        )

        # Add nuisane learnable parameters
        self._bin_edges = bin_edges
        self._bin_centers = bin_centers
        detector_deltas = [
            Variable(trainable=True)
            for _ in self._bin_centers
        ]

    def __prediction_loss(self) -> float:
        raise NotImplementedError("Prediction loss not implemented yet")
    
    def __nuisance_aux_loss(self) -> float:
        raise NotImplementedError("Nuisance auxiliary loss not implemented yet")

    def call(self, data_set: DataSet) -> np.ndarray:
        return self.predict(data_set)
