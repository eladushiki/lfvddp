"""Full-batch Adam warmup and budgeted L-BFGS continuation."""

from copy import deepcopy

import torch

MAX_ADAM_WARMUP = 2000


class EvaluationBudgetExhausted(Exception):
    """Stop before a closure would exceed the full-data evaluation budget."""


def adam_warmup_epochs(total_epochs: int) -> int:
    return min(total_epochs // 20, MAX_ADAM_WARMUP)


def validate_optimizer_continuation(saved_schedule, strategy: str, total_epochs: int):
    """Validate optimizer compatibility before restoring or skipping a checkpoint."""
    saved_strategy = (
        saved_schedule["strategy"] if saved_schedule is not None else "adam"
    )
    if saved_strategy != strategy:
        raise ValueError("Continuation optimizer strategy differs from checkpoint.")
    if saved_schedule is not None and total_epochs < saved_schedule["total_epochs"]:
        raise ValueError("Adam/L-BFGS continuation cannot reduce the epoch budget.")


class HybridOptimizer(torch.optim.Optimizer):
    """An Optimizer owning phase transitions, budget, and resumable state."""

    def __init__(self, parameters, learning_rate: float, total_epochs: int):
        super().__init__(parameters, {"lr": learning_rate})
        self.parameters = self.param_groups[0]["params"]
        self.learning_rate = self.param_groups[0]["lr"]
        self.total_epochs = total_epochs
        self.warmup_epochs = adam_warmup_epochs(total_epochs)
        self.evaluations = 0
        self.phase = "adam" if self.warmup_epochs else "lbfgs"
        self._adopt_optimizer(self._make_optimizer())

        self.register_state_dict_post_hook(self._save_schedule)
        self.register_load_state_dict_pre_hook(self._restore_schedule)
        self.register_load_state_dict_post_hook(self._bind_loaded_optimizer)

    def add_param_group(self, param_group):
        if self.param_groups:
            raise ValueError(
                "HybridOptimizer supports one parameter group, as required by L-BFGS."
            )
        super().add_param_group(param_group)

    def _bind_delegate(self):
        self.optimizer.state = self.state
        self.optimizer.param_groups = self.param_groups

    def _adopt_optimizer(self, optimizer):
        """Keep canonical group/state objects stable when changing algorithms."""
        group = self.param_groups[0]
        metadata = {
            key: group[key] for key in ("initial_lr", "param_names") if key in group
        }
        self.state.clear()
        self.state.update(optimizer.state)
        group.clear()
        group.update(optimizer.param_groups[0])
        group.update(metadata)
        self.optimizer = optimizer
        self.defaults = optimizer.defaults.copy()
        self._bind_delegate()

    def _save_schedule(self, _optimizer, state):
        state["optimizer_schedule"] = self.schedule_state_dict()

    def _restore_schedule(self, _optimizer, state):
        self.restore(state["optimizer_schedule"])
        return {
            key: value for key, value in state.items() if key != "optimizer_schedule"
        }

    def _bind_loaded_optimizer(self, _optimizer):
        # Optimizer.load_state_dict replaces state and param_groups with cast
        # copies. Rebind the delegate before user post-load hooks observe it.
        self._bind_delegate()

    def _make_optimizer(self):
        if self.phase == "adam":
            return torch.optim.Adam(self.parameters, lr=self.learning_rate)
        return torch.optim.LBFGS(
            self.parameters,
            lr=1.0,
            max_iter=1,
            max_eval=50,
            history_size=50,
            tolerance_grad=0.0,
            tolerance_change=0.0,
            line_search_fn="strong_wolfe",
        )

    def schedule_state_dict(self):
        return {
            "strategy": "adam_lbfgs",
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "evaluations": self.evaluations,
            "phase": self.phase,
        }

    def restore(self, state):
        validate_optimizer_continuation(state, "adam_lbfgs", self.total_epochs)
        if not 0 <= state["evaluations"] <= state["total_epochs"]:
            raise ValueError("Invalid checkpoint evaluation count.")
        if state["phase"] not in {"adam", "lbfgs"}:
            raise ValueError("Invalid checkpoint optimizer phase.")
        self.warmup_epochs = state["warmup_epochs"]
        if not 0 <= self.warmup_epochs <= min(state["total_epochs"], MAX_ADAM_WARMUP):
            raise ValueError("Invalid checkpoint warmup boundary.")
        if state["phase"] == "adam" and state["evaluations"] > self.warmup_epochs:
            raise ValueError("Adam checkpoint exceeds its warmup budget.")
        if state["phase"] == "lbfgs" and state["evaluations"] < self.warmup_epochs:
            raise ValueError("L-BFGS checkpoint precedes its warmup boundary.")
        self.evaluations = state["evaluations"]
        self.phase = state["phase"]
        self._adopt_optimizer(self._make_optimizer())

    def __iter__(self):
        while self.evaluations < self.total_epochs:
            yield self.evaluations

    def step(self, closure):
        """Swap optimizers at the warmup boundary and budget every closure call."""
        if self.phase == "adam" and self.evaluations >= self.warmup_epochs:
            self.phase = "lbfgs"
            self._adopt_optimizer(self._make_optimizer())
        last_loss = None

        def budgeted_closure():
            nonlocal last_loss
            if self.evaluations >= self.total_epochs:
                raise EvaluationBudgetExhausted
            last_loss = closure()
            self.evaluations += 1
            return last_loss

        if self.phase == "adam":
            self.optimizer.step(budgeted_closure)
            return last_loss
        # Interrupted line searches leave temporary parameters and partial state.
        # Roll both back; the common training closure retains evaluated candidates.
        saved_parameters = [p.detach().clone() for p in self.parameters]
        saved_optimizer = deepcopy(self.optimizer.state_dict())
        try:
            self.optimizer.step(budgeted_closure)
        except EvaluationBudgetExhausted:
            with torch.no_grad():
                for parameter, saved in zip(self.parameters, saved_parameters):
                    parameter.copy_(saved)
            self.optimizer.load_state_dict(saved_optimizer)
            self._adopt_optimizer(self.optimizer)
        return last_loss


def training_epochs(optimizer, start_epoch: int, target_epochs: int):
    """Use the optimizer's evaluation counter when steps contain multiple probes."""
    return (
        iter(optimizer)
        if isinstance(optimizer, HybridOptimizer)
        else range(start_epoch, target_epochs)
    )


def evaluation_epoch(optimizer, epoch: int) -> int:
    return optimizer.evaluations if isinstance(optimizer, HybridOptimizer) else epoch


def optimizer_schedule(optimizer):
    return (
        optimizer.schedule_state_dict()
        if isinstance(optimizer, HybridOptimizer)
        else None
    )


def load_optimizer_state(optimizer, state, schedule):
    """Allow old checkpoints with a separate schedule alongside raw Torch state."""
    if isinstance(optimizer, HybridOptimizer):
        optimizer.load_state_dict({"optimizer_schedule": schedule, **state})
    else:
        optimizer.load_state_dict(state)
