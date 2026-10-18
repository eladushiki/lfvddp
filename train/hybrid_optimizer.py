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


class HybridOptimizer:
    def __init__(self, parameters, learning_rate: float, total_epochs: int):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.total_epochs = total_epochs
        self.warmup_epochs = adam_warmup_epochs(total_epochs)
        self.evaluations = 0
        self.phase = "adam" if self.warmup_epochs else "lbfgs"
        self.optimizer = self._make_optimizer()

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

    def state_dict(self):
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
        self.optimizer = self._make_optimizer()

    def step(self, evaluate, after_evaluation=lambda: None):
        if self.phase == "adam" and self.evaluations >= self.warmup_epochs:
            self.phase = "lbfgs"
            self.optimizer = self._make_optimizer()

        def closure():
            if self.evaluations >= self.total_epochs:
                raise EvaluationBudgetExhausted
            self.optimizer.zero_grad(set_to_none=True)
            loss = evaluate(self.evaluations)
            loss.backward()
            self.evaluations += 1
            after_evaluation()
            return loss

        if self.phase == "adam":
            closure()
            self.optimizer.step()
            return
        # Interrupted line searches leave temporary parameters and partial state.
        # Roll both back, retaining all evaluated best candidates in the caller.
        saved_parameters = [p.detach().clone() for p in self.parameters]
        saved_optimizer = deepcopy(self.optimizer.state_dict())
        try:
            self.optimizer.step(closure)
        except EvaluationBudgetExhausted:
            with torch.no_grad():
                for parameter, saved in zip(self.parameters, saved_parameters):
                    parameter.copy_(saved)
            self.optimizer.load_state_dict(saved_optimizer)
