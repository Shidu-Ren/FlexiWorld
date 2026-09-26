from __future__ import annotations
from collections.abc import Sequence
import torch
from flexiworld.planning.base import DirectSolver
from flexiworld.planning.history import PrimitiveTailLedger
from flexiworld.models.world_model import normalize_schedule


class VarKDirectSolver(DirectSolver):
    evaluation_mode = "direct_vark"

    def __init__(
        self,
        *args,
        schedule: Sequence[int] | int = 5,
        replan_previous_mode: str = "current",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.schedule = normalize_schedule(schedule)
        self.replan_previous_mode = replan_previous_mode
        self._replan_history = PrimitiveTailLedger(replan_previous_mode)
        self._replans = 0
        self.schedule_log: list[list[int]] = []

    def configure(self, *args, **kwargs):
        out = super().configure(*args, **kwargs)
        self._check_config()
        return out

    def _check_config(self) -> None:
        cfg = self._config
        if cfg.action_block != 1:
            raise ValueError(
                "variable-k plans must run at action_block=1: the policy multiplies receding_horizon by action_block when it reshapes the plan (policy.py:420-422), which a mixed-size schedule cannot satisfy."
            )
        if cfg.warm_start:
            raise ValueError(
                "warm_start must be False: DirectSolver ignores init_action (direct_solver.py:24), so a warm start would only allocate a fixed-shape _next_init buffer (policy.py:411-416) that a shrinking plan later fails to fill."
            )
        if cfg.horizon < cfg.receding_horizon:
            raise ValueError("horizon must be >= receding_horizon")

    def _plan_length(self) -> int:
        return sum(self.schedule)

    def schedule_for(self, n_envs: int) -> list[int]:
        return list(self.schedule)

    def _validate_primitive_plan(self, actions: torch.Tensor) -> None:
        expected_dim = int(self.model.primitive_dim)
        if actions.ndim != 3 or actions.shape[-1] != expected_dim:
            raise ValueError(
                f"expected a primitive-level plan (B, n, {expected_dim}), got {tuple(actions.shape)}. A block-width plan means stock get_action leaked through; at action_block=1 that would reach policy.py:433 and corrupt the executed actions."
            )

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        del init_action
        import time

        if not bool(getattr(self.model, "has_intent_actor", lambda: False)()):
            raise RuntimeError("Direct evaluation requires a trained intent actor")
        sched = self.schedule_for(self._n_envs if hasattr(self, "_n_envs") else 1)
        previous_override = self._replan_history.override(
            info_dict, is_replan=self._replans > 0, primitive_dim=int(self.model.primitive_dim)
        )
        start = time.perf_counter()
        if previous_override is None:
            actions = self.model.get_action_vark(info_dict, sched)
        else:
            actions = self.model.get_action_vark(
                info_dict, sched, previous_block_override=previous_override
            )
        elapsed = time.perf_counter() - start
        self._validate_primitive_plan(actions)
        actions = actions.detach().cpu()
        if self._replans == 0:
            self._replan_history.remember(info_dict, actions)
        actions = self._pad_to_receding(actions)
        stats = {
            "solve_time": elapsed,
            "get_cost_calls": 0.0,
            "candidate_sequences": 0.0,
            "reach": float(sum(sched)),
            "depth": float(len(sched)),
        }
        stats.update(
            {
                k: float(v)
                for k, v in getattr(self.model, "last_direct_diagnostics", {}).items()
                if isinstance(v, (int, float))
            }
        )
        self.timing_history.append(stats)
        self.schedule_log.append(list(sched))
        self._replans += 1
        return {
            "actions": actions,
            "mean": [actions],
            "var": [torch.zeros_like(actions)],
            "costs": [],
            "timing": stats,
        }

    def _pad_to_receding(self, actions: torch.Tensor) -> torch.Tensor:
        need = int(self._config.receding_horizon)
        if actions.shape[1] >= need:
            return actions
        tail = actions[:, -1:].expand(-1, need - actions.shape[1], -1)
        return torch.cat([actions, tail], dim=1)
