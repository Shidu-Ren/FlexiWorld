"""Track the actually executed action tail across a two-stage evaluation."""

from __future__ import annotations


import hashlib


import re


from typing import Any


import numpy as np


import torch


class PrimitiveTailLedger:
    """Build the previous primitive block used at a re-observation boundary."""

    def __init__(self, mode: str = "current") -> None:
        match = re.fullmatch(r"(tile|carry|zero)(10|[1-9])", mode)
        if mode != "current" and match is None:
            raise ValueError(
                "replan_previous_mode must be current or tileN/carryN/zeroN for N in [1,10]"
            )
        self.mode = mode
        self.kind = "current" if match is None else match.group(1)
        self.length = 1 if match is None else int(match.group(2))
        self._tails: dict[str, torch.Tensor] = {}

    @staticmethod
    def _goal_keys(info: dict[str, Any]) -> list[str]:
        """Return stable per-environment keys, preferring compact goal state."""
        name = "goal_qpos" if info.get("goal_qpos") is not None else "goal"
        value = info.get(name)
        if value is None:
            raise ValueError("carryN evaluation requires goal_qpos or goal in policy info")
        value = torch.as_tensor(value).detach().cpu().contiguous()
        if value.ndim < 1:
            raise ValueError(f"{name} must have a batch dimension")
        keys = []
        for row in value:
            array = row.float().numpy() if row.dtype == torch.bfloat16 else row.numpy()
            digest = hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()
            keys.append(f"{name}:{tuple(row.shape)}:{row.dtype}:{digest}")
        return keys

    def remember(self, info: dict[str, Any], actions: torch.Tensor) -> None:
        """Store the planned tail; it is exactly executed when receding horizon is D."""
        if self.kind != "carry":
            return
        if actions.ndim != 3 or actions.size(1) < self.length:
            raise ValueError(
                f"carry{self.length} requires a primitive plan at least {self.length} steps long"
            )
        keys = self._goal_keys(info)
        if len(keys) != actions.size(0):
            raise ValueError("goal batch and action-plan batch have different sizes")
        if len(set(keys)) != len(keys):
            raise ValueError(
                "duplicate goal identities: carry history requires unique episode/env IDs "
                "when multiple environments share a goal"
            )
        for key, tail in zip(keys, actions[:, -self.length :]):
            self._tails[key] = tail.detach().cpu().clone()

    def override(
        self,
        info: dict[str, Any],
        *,
        is_replan: bool,
        primitive_dim: int,
    ) -> torch.Tensor | None:
        if not is_replan:
            block = info.get("_expert_previous_block")
            if block is None:
                raise ValueError("initial planning requires the normalized five-action history")
            if block.ndim != 3 or block.shape[1:] != (5, primitive_dim):
                raise ValueError("initial history must have shape (batch, 5, action_dim)")
            return block
        if self.kind == "current":
            return None
        action = torch.as_tensor(info["action"])
        if action.ndim == 2:
            action = action[:, None]
        last = action[:, -1:, :primitive_dim]
        if self.kind == "tile":
            return last.repeat(1, self.length, 1)
        if self.kind == "zero":
            return last.new_zeros(last.size(0), self.length, last.size(2))
        tails = []
        for key in self._goal_keys(info):
            if key not in self._tails:
                raise KeyError("second-stage goal did not match an initial planned trajectory")
            tails.append(self._tails[key])
        return torch.stack(tails, dim=0).to(last)
