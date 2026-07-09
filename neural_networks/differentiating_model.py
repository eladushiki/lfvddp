from __future__ import annotations

from contextlib import contextmanager
from logging import info
from time import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from data_tools.data_generation import DataBatch
from data_tools.detector.detector_effect import DetectorEffect
from data_tools.data_utils import DataSet
from data_tools.detector.detector_config import DetectorConfig
from frame.context.execution_context import ExecutionContext
from frame.file_system.training_history import HistoryKeys
from neural_networks.utils import (
    ContextedModel,
    save_training_outcomes,
    get_model_logging_dir,
)
from train.checkpoints import find_latest_training_checkpoint, save_training_checkpoint
from train.train_config import TrainConfig


def _calculate_loss_weights(data: DataBatch) -> npt.NDArray:
    return np.concatenate(
        [
            ds._weight_mask
            * ds.corrected_n_samples
            / data.unified_data.corrected_n_samples
            for ds, params in data
        ],
        axis=0,
    )


class _ZeroEstimator(nn.Module):
    def __init__(self, output_dimension: int):
        super().__init__()
        self._output_dimension = output_dimension

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return events.new_zeros((events.shape[0], self._output_dimension))


class DifferentiatingModel(nn.Module, ContextedModel):
    """
    Symmetrized DDP's model used to estimate the test statistic using PyTorch Lightning.
    A custom loss function is used to find the maximizing parameters for hypothesis.
    """

    def __init__(
        self,
        context: ExecutionContext,
        detector_effect: DetectorEffect,
        is_numerator: bool,
        name: str,
    ):
        super().__init__()
        self._name = name
        self._is_numerator = is_numerator
        self._context = context
        self._config: Union[TrainConfig, DetectorConfig] = context.config

        # Add layers by spec. We would add two NNs to express f, g separately.
        self._build_layers()

        # Add detector uncertainty nuisance parameters
        self._detector_effect = detector_effect
        self._build_eta()
        self._bins_of_events = None  # Set in context

        # Initialize NN parameters according to strategy
        self._initialize_parameters()

        # Store training data and weights for metrics computation
        self._train_data = None
        self._train_target_classifier = None
        self._train_weights = None
        self._norm_factor = None
        self._training_history = {}
        self._tensorboard_writer = None

    @property
    def _device(self) -> torch.device:
        return torch.device("cpu")

    def _build_layers(self):
        # Fully connected 2-layer network:
        input_dim = self._config.train__nn_input_dimension
        hidden_size = self._config.train__nn_inner_layer_nodes
        output_size = self._config.train__nn_output_dimension

        if not self._is_numerator:
            self.f_network = _ZeroEstimator(output_size)
            self.g_network = _ZeroEstimator(output_size)
            return

        self.f_network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
        )
        self.g_network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
        )

    def _initialize_tensorboard_writer(self) -> None:
        """Initialize TensorBoard writer for this model instance."""
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_log_dir = get_model_logging_dir(self._context, self._name)
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        self._tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    def _close_tensorboard_writer(self) -> None:
        """Close and clear TensorBoard writer if initialized."""
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()
            self._tensorboard_writer = None

    def _log_tensorboard_parameter(
        self, tag: str, parameter: torch.Tensor, epoch: int
    ) -> None:
        if self._tensorboard_writer is None:
            return
        detached_parameter = parameter.detach().cpu()
        self._tensorboard_writer.add_histogram(tag, detached_parameter, epoch)

    def _log_tensorboard_network_parameters(
        self,
        network_name: str,
        network: nn.Module,
        epoch: int,
    ) -> None:
        for parameter_name, parameter in network.named_parameters():
            self._log_tensorboard_parameter(
                f"parameters/{network_name}/{parameter_name}",
                parameter,
                epoch,
            )

    def _log_tensorboard_nuisance_parameters(self, epoch: int) -> None:
        for observable_name, parameter in self._detector_deltas.items():
            self._log_tensorboard_parameter(
                f"parameters/eta/nuisance_{observable_name}",
                parameter,
                epoch,
            )

    def _log_tensorboard_epoch(self, epoch: int, loss: torch.Tensor) -> None:
        if self._tensorboard_writer is None:
            return
        self._tensorboard_writer.add_scalar(
            "loss",
            float(loss.detach().cpu()),
            epoch,
        )
        if (epoch + 1) % self._config.train__number_of_epochs_for_checkpoint != 0:
            return
        self._log_tensorboard_network_parameters("f_network", self.f_network, epoch)
        self._log_tensorboard_network_parameters("g_network", self.g_network, epoch)
        self._log_tensorboard_nuisance_parameters(epoch)

    def _build_eta(self):
        self._detector_deltas = {}
        if not self._config.train__data_is_train_for_nuisances:
            self.eta = _ZeroEstimator(1)
            return

        for i, nbins in enumerate(self._detector_effect._numbers_of_bins):
            nuisance_var = nn.Parameter(
                torch.full((nbins,), 1e-3, dtype=torch.float32, device=self._device)
            )
            self.register_parameter(
                f"nuisance_{self._observable_names[i]}", nuisance_var
            )
            self._detector_deltas[self._observable_names[i]] = nuisance_var

        def eta(events: torch.Tensor) -> torch.Tensor:
            if self._bins_of_events is None:
                raise RuntimeError(
                    "eta can only be evaluated inside a binning_context."
                )
            dimensional_deltas = []
            for i, obs in enumerate(self._observable_names):
                dimensional_deltas.append(
                    self._detector_deltas[obs][self._bins_of_events[:, i]]
                )
            return torch.stack(dimensional_deltas, dim=1).prod(dim=1, keepdim=True)

        self.eta = eta

    def _initialize_parameters(self) -> None:
        """
        Create newly initialized weights matching the training strategy.
        This is the single source of truth for weight initialization.
        Assumes 2-layer network (1 hidden layer).
        """
        # Use Xavier uniform with configurable gain for weight initialization
        gain = self._config.train__nn_xavier_gain

        if self._is_numerator:
            for network in self.f_network, self.g_network:
                # Layer 0: Input to hidden
                hidden_layer = network[0]
                nn.init.xavier_uniform_(hidden_layer.weight, gain=gain)
                nn.init.uniform_(hidden_layer.bias, a=-0.3, b=0.3)

                # Layer 2: Hidden to output (skipping LeakyReLU at index 1)
                output_layer = network[2]
                nn.init.xavier_uniform_(output_layer.weight, gain=gain)
                nn.init.uniform_(output_layer.bias, a=-0.3, b=0.3)

        # Handle detector nuisances separately
        if self._config.train__data_is_train_for_nuisances:
            for var in self._detector_deltas.values():
                nn.init.constant_(var, 1e-3)

    def _clamp_nuisance_parameters(self) -> None:
        if not self._config.train__data_is_train_for_nuisances:
            return
        with torch.no_grad():
            for var in self._detector_deltas.values():
                var.clamp_(min=-1.0 + 1e-6, max=1.0 - 1e-6)

    def configure_optimizers(self) -> Optional[optim.Optimizer]:
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            return None
        optimizer = optim.Adam(
            trainable_parameters,
            lr=self._config.train__learning_rate,
        )
        return optimizer

    @property
    def _observable_names(self) -> List[str]:
        return self._detector_effect._observable_names

    @staticmethod
    def ddp_minimization_loss(
        f_a_sr: torch.Tensor,
        f_b_sr: torch.Tensor,
        g_a_sr: torch.Tensor,
        g_b_sr: torch.Tensor,
        eta_a_sr: torch.Tensor,
        eta_a_cr: torch.Tensor,
        eta_b_sr: torch.Tensor,
        eta_b_cr: torch.Tensor,
        z_eta: torch.Tensor,
    ) -> torch.Tensor:
        """
        The loss function for minimizing any expression in lfvddp.

        From the t values expression in the paper, use:
        - with or without f, g for numerator or denominator expression.
        - with or without nuisance parameters at will.
        Enter a zeros vector for each for it not to be use. This is the default
        behavior of DifferentiatingModel fit.

        Returns torch.Tensor of the same shape as the input tensors with NLL values to be minimized.
        """
        # SR sum term
        eta_sr = torch.cat([eta_a_sr, eta_b_sr])

        N_A_SR = len(eta_a_sr)
        e_to_the_f_sr = torch.exp(torch.cat([f_a_sr, f_b_sr]))
        eta_plus_term_sr = 1 + eta_sr
        N_B_SR = len(eta_b_sr)
        e_to_the_g_sr = torch.exp(torch.cat([g_a_sr, g_b_sr]))
        eta_minus_term_sr = 1 - eta_sr
        sr_sum_term = torch.sum(
            N_A_SR * e_to_the_f_sr * eta_plus_term_sr
            + N_B_SR * e_to_the_g_sr * eta_minus_term_sr
        )

        # CR sum term
        eta_cr = torch.cat([eta_a_cr, eta_b_cr])
        N_A_CR = len(eta_a_cr)
        eta_plus_term_cr = 1 + eta_cr
        N_B_CR = len(eta_b_cr)
        eta_minus_term_cr = 1 - eta_cr
        cr_sum_term = torch.sum(N_A_CR * eta_plus_term_cr + N_B_CR * eta_minus_term_cr)

        # z_eta log term
        z_eta_term = N_B_SR * torch.log(z_eta)

        # Optional f and g sum terms
        f_a_sr_sum_term = torch.sum(f_a_sr)
        g_b_sr_sum_term = torch.sum(g_b_sr)

        # eta log terms
        eta_a = torch.cat([eta_a_sr, eta_a_cr])
        eta_plus_a_sum_term = torch.sum(torch.log(1 + eta_a))

        eta_b = torch.cat([eta_b_sr, eta_b_cr])
        eta_minus_b_sum_term = torch.sum(torch.log(1 - eta_b))

        # Total loss is sum of log-likelihoods
        return (
            sr_sum_term
            + cr_sum_term
            - z_eta_term
            - f_a_sr_sum_term
            - g_b_sr_sum_term
            - eta_plus_a_sum_term
            - eta_minus_b_sum_term
        )

    @staticmethod
    def _loss_from_estimates(
        region_mask: torch.Tensor,
        f_x_sr_est: torch.Tensor,
        g_x_sr_est: torch.Tensor,
        eta_x_est: torch.Tensor,
    ) -> torch.Tensor:
        a_sr_mask = region_mask == DataSet.DataSetCategory.A_SR.value
        b_sr_mask = region_mask == DataSet.DataSetCategory.B_SR.value
        sr_map = region_mask[a_sr_mask | b_sr_mask]

        # Z_eta norm term
        z_eta = torch.clamp(
            torch.mean(
                1 - eta_x_est[region_mask == DataSet.DataSetCategory.B_SR.value]
            ),
            min=1e-6,
        )

        return DifferentiatingModel.ddp_minimization_loss(
            f_a_sr=f_x_sr_est[sr_map == DataSet.DataSetCategory.A_SR.value],
            f_b_sr=f_x_sr_est[sr_map == DataSet.DataSetCategory.B_SR.value],
            g_a_sr=g_x_sr_est[sr_map == DataSet.DataSetCategory.A_SR.value],
            g_b_sr=g_x_sr_est[sr_map == DataSet.DataSetCategory.B_SR.value],
            eta_a_sr=eta_x_est[region_mask == DataSet.DataSetCategory.A_SR.value],
            eta_a_cr=eta_x_est[region_mask == DataSet.DataSetCategory.A_CR.value],
            eta_b_sr=eta_x_est[region_mask == DataSet.DataSetCategory.B_SR.value],
            eta_b_cr=eta_x_est[region_mask == DataSet.DataSetCategory.B_CR.value],
            z_eta=z_eta,
        )

    def forward(
        self,
        data: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> torch.Tensor:
        a_sr_mask = region_mask == DataSet.DataSetCategory.A_SR.value
        b_sr_mask = region_mask == DataSet.DataSetCategory.B_SR.value
        sr_data = data[a_sr_mask | b_sr_mask]
        f_x_sr_est = self.f_network(sr_data)
        g_x_sr_est = self.g_network(sr_data)
        eta_x_est = self.eta(data)

        return self._loss_from_estimates(
            region_mask=region_mask,
            f_x_sr_est=f_x_sr_est,
            g_x_sr_est=g_x_sr_est,
            eta_x_est=eta_x_est,
        )

    @contextmanager
    def binning_context(self, data: DataSet):
        try:
            self._bins_of_events = torch.tensor(
                self._detector_effect.get_event_bin_centers(data, indexed=True),
                dtype=torch.long,
                device=self._device,
            )
            yield
        finally:
            self._bins_of_events = None

    @staticmethod
    def _prepare_training_tensors(
        data: DataBatch,
        weights: npt.NDArray,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert DataSet and target to tensors on the correct device."""
        data_parts = []
        mask_parts = []
        for ds, params in data:
            data_parts.append(ds.events)
            mask_parts.append(
                np.full(ds.n_samples, params.category.value, dtype=np.int64)
            )

        data_tensor = torch.tensor(
            np.concatenate(data_parts), dtype=torch.float32, device=device
        )
        mask_tensor = torch.from_numpy(np.concatenate(mask_parts)).to(device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        return data_tensor, mask_tensor, weights_tensor

    def _prepare_training_data(
        self,
        data: DataBatch,
        weights: npt.NDArray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._prepare_training_tensors(data, weights, self._device)

    def _log_epoch(self, epoch: int, loss: torch.Tensor) -> None:
        self._training_history.setdefault(HistoryKeys.LOSS.value, []).append(
            float(loss.detach().cpu())
        )
        self._training_history.setdefault(HistoryKeys.EPOCH.value, []).append(epoch)
        self._log_tensorboard_epoch(epoch, loss)

    def has_trainable_parameters(self) -> bool:
        return any(parameter.requires_grad for parameter in self.parameters())

    @staticmethod
    def has_configured_trainable_parameters(
        config: Union[TrainConfig, DetectorConfig],
        is_numerator: bool,
    ) -> bool:
        if not isinstance(config, TrainConfig):
            raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")
        return is_numerator or config.train__data_is_train_for_nuisances

    def _load_training_checkpoint_if_requested(
        self, optimizer: Optional[optim.Optimizer]
    ) -> int:
        checkpoint_result = find_latest_training_checkpoint(
            self._context, self._name, warn_missing=False
        )
        if checkpoint_result is None:
            return 0

        checkpoint_path, checkpoint = checkpoint_result
        self.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        if optimizer is not None and optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        self._training_history = {
            key: list(value)
            for key, value in checkpoint.get("training_history", {}).items()
        }
        if (
            HistoryKeys.LOSS.value in self._training_history
            and HistoryKeys.EPOCH.value not in self._training_history
        ):
            self._training_history[HistoryKeys.EPOCH.value] = list(
                range(len(self._training_history[HistoryKeys.LOSS.value]))
            )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        info(
            f"Loaded checkpoint for {self._name} from {checkpoint_path}; resuming at epoch {start_epoch}"
        )
        return start_epoch

    def _train_step(
        self,
        optimizer: Optional[optim.Optimizer],
        data: torch.Tensor,
        region_mask: torch.Tensor,
        batch_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run one optimization step and return the batch loss."""
        loss = self(data=data, region_mask=region_mask)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self._clamp_nuisance_parameters()

        return loss

    def fit(
        self,
        data: DataBatch,
        weights: npt.NDArray,
    ) -> Dict[str, List[float]]:
        """Pure PyTorch training loop."""
        if not self.has_trainable_parameters():
            raise RuntimeError(
                "Cannot fit DifferentiatingModel without trainable parameters; "
                "call calculate_loss_value for the static loss instead."
            )

        # Store training data for metrics computation
        normalized_data, self._norm_factor = data.get_normalized()
        data_tensor, region_mask_tensor, weights_tensor = self._prepare_training_data(
            normalized_data, weights
        )

        self.train()
        optimizer = self.configure_optimizers()

        target_epochs = self._config.train__epochs
        start_epoch = self._load_training_checkpoint_if_requested(optimizer)

        if start_epoch >= target_epochs:
            return self._training_history

        self._initialize_tensorboard_writer()

        try:
            epoch_iterator = range(start_epoch, target_epochs)

            if self._config.train__enable_progress_bar:
                epoch_iterator = tqdm(epoch_iterator, desc=f"{self._name} training")

            # Training with binning context
            with self.binning_context(data.unified_data):
                for epoch in epoch_iterator:
                    epoch_last_predictions = self._train_step(
                        optimizer=optimizer,
                        data=data_tensor,
                        region_mask=region_mask_tensor,
                        batch_weights=weights_tensor,
                    )
                    self._log_epoch(epoch, epoch_last_predictions)

                    if (
                        (epoch + 1)
                        % self._config.train__number_of_epochs_for_checkpoint
                        == 0
                        or epoch == target_epochs - 1
                    ):
                        save_training_checkpoint(
                            context=self._context,
                            model_name=self._name,
                            model=self,
                            optimizer=optimizer,
                            epoch=epoch,
                            training_history=self._training_history,
                        )
        finally:
            self._close_tensorboard_writer()

        # Collect history from training
        return self._training_history

    def calculate_loss_value(
        self,
        data: DataBatch,
        weights: npt.NDArray,
    ) -> float:
        normalized_data, self._norm_factor = data.get_normalized()
        data_tensor, region_mask_tensor, _ = self._prepare_training_data(
            normalized_data, weights
        )
        self.eval()
        with torch.no_grad():
            with self.binning_context(data.unified_data):
                loss = self(data=data_tensor, region_mask=region_mask_tensor)
        return float(loss.detach().cpu())

    @classmethod
    def calculate_loss_statically(
        cls,
        context: ExecutionContext,
        data: DataBatch,
        detector_effect: DetectorEffect,
        is_numerator: bool,
        name: str,
    ) -> float:
        model = cls(
            context=context,
            detector_effect=detector_effect,
            is_numerator=is_numerator,
            name=name,
        )
        if model.has_trainable_parameters():
            raise RuntimeError(
                "Static loss calculation is only valid when the LFVNN model has no trainable parameters."
            )
        weights = _calculate_loss_weights(data)
        return model.calculate_loss_value(data=data, weights=weights)

    def _predict_ndf(
        self,
        data: DataSet,
        network: nn.Module,
        eta_sign: float,
    ) -> npt.NDArray:
        if self._norm_factor is None:
            raise RuntimeError("Cannot predict before the model has been fitted.")
        normalized_data = data / self._norm_factor
        x_tensor = torch.tensor(
            normalized_data.events, dtype=torch.float32, device=self._device
        )
        self.eval()
        with torch.no_grad():
            with self.binning_context(data):
                eta_term = torch.clamp(1 + eta_sign * self.eta(x_tensor), min=1e-12)
                predictions = torch.exp(network(x_tensor)) * eta_term
        return predictions.detach().cpu().numpy()

    def predict(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, self.f_network, eta_sign=1.0)

    def predict_secondary(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, self.g_network, eta_sign=-1.0)

    def predict_eta(self, data: DataSet) -> npt.NDArray:
        x_tensor = torch.tensor(data.events, dtype=torch.float32, device=self._device)
        self.eval()
        with torch.no_grad():
            with self.binning_context(data):
                predictions = self.eta(x_tensor)
        return predictions.detach().cpu().numpy()

    def save_parameters(self, file_path) -> None:
        """Save PyTorch model parameters to file."""
        torch.save(self.state_dict(), file_path)


def calc_min_LFVNN(
    context: ExecutionContext,
    data: DataBatch,
    detector_effect: DetectorEffect,
    is_numerator: bool,
    name: str,
) -> Tuple[ContextedModel, float]:
    loss_weights = _calculate_loss_weights(data)

    # Train
    info("Starting training")
    t0 = time()

    model = DifferentiatingModel(
        context=context,
        detector_effect=detector_effect,
        is_numerator=is_numerator,
        name=name,
    )
    if not model.has_trainable_parameters():
        raise RuntimeError(
            "Cannot train LFVNN without trainable parameters; "
            "call DifferentiatingModel.calculate_loss_statically instead."
        )

    model_history = model.fit(
        data=data,
        weights=loss_weights,
    )

    info(f"Training time (seconds): {time() - t0}")

    # Calculate minimum loss from training history
    final_loss = model_history[HistoryKeys.LOSS.value][-1]
    info(f"Minimum weighted loss achieved: {final_loss:.6f}")

    save_training_outcomes(
        context,
        model_history=model_history,
        tau_model=model,
    )

    return model, final_loss
