from __future__ import annotations
import time
from collections import defaultdict
from typing import Any
import numpy as np
import torch
from stable_worldmodel.solver import CEMSolver
from flexiworld.planning.history import PrimitiveTailLedger
from flexiworld.models.action import sinusoidal


class ActorPathCEMSolver(CEMSolver):
    VALID_SCORE_MODES = {"terminal", "arrival_min"}
    evaluation_mode = "odyssey_actor_path_cem"

    def __init__(
        self,
        *args: Any,
        temperature: float = 0.2,
        score_mode: str = "arrival_min",
        update_alpha: float = 1.0,
        noise_std_floor: float = 0.05,
        noise_std_cap: float = 2.0,
        residual_penalty: float = 0.0,
        block_k: int = 5,
        replan_previous_mode: str = "current",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if self.n_steps < 1:
            raise ValueError("n_steps must be positive")
        if not 1 <= self.topk <= self.num_samples:
            raise ValueError("topk must be in [1, num_samples]")
        if self.var_scale <= 0:
            raise ValueError("var_scale must be positive")
        if self.callbacks:
            raise ValueError("ActorPathCEMSolver does not support CEM callbacks")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if score_mode not in self.VALID_SCORE_MODES:
            raise ValueError(f"unknown score_mode: {score_mode}")
        if not 0 < update_alpha <= 1:
            raise ValueError("update_alpha must be in (0, 1]")
        if not 0 < noise_std_floor <= noise_std_cap:
            raise ValueError("require 0 < noise_std_floor <= noise_std_cap")
        if residual_penalty < 0:
            raise ValueError("residual_penalty must be non-negative")
        if block_k < 1:
            raise ValueError("block_k must be positive")
        self.temperature = float(temperature)
        self.score_mode = str(score_mode)
        self.update_alpha = float(update_alpha)
        self.noise_std_floor = float(noise_std_floor)
        self.noise_std_cap = float(noise_std_cap)
        self.residual_penalty = float(residual_penalty)
        self.block_k = int(block_k)
        self.replan_previous_mode = replan_previous_mode
        self._replan_history = PrimitiveTailLedger(replan_previous_mode)
        self._replans = 0
        self.timing_history: list[dict[str, float]] = []

    def configure(self, *, action_space, n_envs: int, config) -> None:
        if int(config.action_block) != self.block_k:
            raise ValueError(f"plan_config.action_block must equal block_k={self.block_k}")
        if bool(getattr(config, "warm_start", False)):
            raise ValueError("warm_start must be false: the exact Direct plan is injected")
        if int(config.horizon) < 1:
            raise ValueError("plan_config.horizon must be positive")
        if int(config.receding_horizon) > int(config.horizon):
            raise ValueError("receding_horizon must not exceed horizon")
        super().configure(action_space=action_space, n_envs=n_envs, config=config)

    def _check_model_contract(self) -> None:
        model_methods = (
            "_advance",
            "_encode_block",
            "get_action_vark",
            "goal_intent",
            "predicted_steps_remaining",
        )
        missing_model = [name for name in model_methods if not hasattr(self.model, name)]
        if missing_model:
            raise TypeError("actor-path CEM requires VarKJEPA methods: " + ", ".join(missing_model))
        actor = self.model.intent_actor
        actor_methods = ("_conditioning_prefix", "_params", "_trunk", "action_embed")
        missing_actor = [name for name in actor_methods if not hasattr(actor, name)]
        if missing_actor:
            raise TypeError(
                "actor-path CEM requires ARPrimitiveActor methods: " + ", ".join(missing_actor)
            )

    @staticmethod
    def _slice_info(info: dict[str, Any], start: int, end: int) -> dict[str, Any]:
        sliced = {}
        for key, value in info.items():
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def _actor_action_from_noise(
        self,
        current: torch.Tensor,
        intent: torch.Tensor,
        previous_embedding: torch.Tensor,
        noise: torch.Tensor,
        steps_remaining: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor = self.model.intent_actor
        primitive_dim = int(actor.action_dim)
        if noise.shape[-1] % primitive_dim:
            raise ValueError("actor noise width is not a primitive multiple")
        block_k = noise.shape[-1] // primitive_dim
        if block_k != self.block_k:
            raise ValueError(f"expected k={self.block_k}, got k={block_k}")
        primitive_noise = noise.reshape(noise.shape[0], block_k, primitive_dim)
        k_tensor = torch.full((noise.shape[0],), block_k, dtype=torch.long, device=noise.device)
        sequence = actor._conditioning_prefix(
            current, intent, previous_embedding, k_tensor, steps_remaining=steps_remaining
        )
        emitted = []
        emitted_log_stds = []
        for primitive in range(block_k):
            positions = torch.arange(sequence.size(1), device=sequence.device)
            hidden = actor._trunk(sequence + sinusoidal(positions, actor.d_model).unsqueeze(0))
            mean, log_std = actor._params(hidden[:, -1])
            action = mean + self.temperature * primitive_noise[:, primitive]
            emitted.append(action)
            emitted_log_stds.append(log_std)
            sequence = torch.cat([sequence, actor.action_embed(action).unsqueeze(1)], dim=1)
        return (
            torch.stack(emitted, dim=1).reshape(noise.shape[0], -1),
            torch.stack(emitted_log_stds, dim=1).reshape(noise.shape[0], -1),
        )

    def _prepare_context(
        self, info: dict[str, Any], previous_block_override: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        tensors = {
            key: value.to(
                device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype
            )
            for key, value in info.items()
            if torch.is_tensor(value)
        }
        previous = tensors.get("action")
        if previous is None:
            raise ValueError("action history is required")
        if previous.ndim == 2:
            previous = previous[:, None]
        if previous.ndim != 3 or previous.shape[-1] != self.model.primitive_dim:
            raise ValueError(f"invalid primitive action history: {tuple(previous.shape)}")
        if not torch.isfinite(previous).all():
            raise ValueError("action history contains non-finite values")
        initial = dict(tensors)
        initial.pop("action", None)
        embeddings = self.model.encode(initial)["emb"]
        goal = self.model._encode_goal(tensors)
        width = max(self.block_k, int(self.model.k_max))
        if previous_block_override is None:
            previous_block = previous[:, -1:, : self.model.primitive_dim]
        else:
            previous_block = previous_block_override.to(device=self.device, dtype=self.dtype)
        previous_k = int(previous_block.size(1))
        previous_embedding = self.model._encode_block(previous_block, k=previous_k, width=width)
        action_embeddings = [previous_embedding] * embeddings.size(1)
        return (embeddings, action_embeddings, goal)

    def _rollout_noise(
        self,
        embedding_history: torch.Tensor,
        action_embedding_history: list[torch.Tensor],
        goal: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, samples, horizon, action_dim = noise.shape
        primitive_dim = int(self.model.primitive_dim)
        if action_dim != self.block_k * primitive_dim:
            raise ValueError(
                f"expected action width {self.block_k * primitive_dim}, got {action_dim}"
            )
        history_steps, latent_dim = embedding_history.shape[1:]
        embeddings = (
            embedding_history[:, None]
            .expand(batch, samples, history_steps, latent_dim)
            .reshape(batch * samples, history_steps, latent_dim)
            .clone()
        )
        flat_goal = (
            goal[:, None].expand(batch, samples, latent_dim).reshape(batch * samples, latent_dim)
        )
        action_embeddings = [
            value.repeat_interleave(samples, dim=0) for value in action_embedding_history
        ]
        flat_noise = noise.reshape(batch * samples, horizon, action_dim)
        history_size = int(self.model.predictor.pos_embedding.shape[1])
        width = max(self.block_k, int(self.model.k_max))
        planned_actions = []
        predicted_states = []
        actor_log_stds = []
        for step in range(horizon):
            current = embeddings[:, -1]
            intent = self.model.goal_intent(
                current, flat_goal, horizon - step, block_k=self.block_k
            )
            fallback = current.new_full((current.size(0),), self.block_k * (horizon - step))
            temporal_steps = self.model.predicted_steps_remaining(
                current, flat_goal, fallback=fallback
            )
            action, log_std = self._actor_action_from_noise(
                current,
                intent,
                action_embeddings[-1],
                flat_noise[:, step],
                steps_remaining=temporal_steps,
            )
            block = action.reshape(batch * samples, self.block_k, primitive_dim)
            new_action_embedding = self.model._encode_block(block, k=self.block_k, width=width)
            embeddings = self.model._advance(
                embeddings,
                action_embeddings,
                new_action_embedding,
                history_size,
                align=self.model.align,
            )
            planned_actions.append(action)
            predicted_states.append(embeddings[:, -1])
            actor_log_stds.append(log_std)
        actions = torch.stack(planned_actions, dim=1).reshape(batch, samples, horizon, action_dim)
        states = torch.stack(predicted_states, dim=1).reshape(batch, samples, horizon, latent_dim)
        log_stds = torch.stack(actor_log_stds, dim=1).reshape(batch, samples, horizon, action_dim)
        return (actions, states, log_stds)

    def _trajectory_cost(
        self, states: torch.Tensor, goal: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        distance = (states - goal[:, None, None]).square().sum(dim=-1).float()
        if self.score_mode == "terminal":
            cost = distance[:, :, -1]
        else:
            cost = distance.min(dim=-1).values
        if self.residual_penalty:
            cost = cost + self.residual_penalty * noise.square().mean(dim=(-1, -2))
        return cost

    def _rollout_actions(
        self,
        embedding_history: torch.Tensor,
        action_embedding_history: list[torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = embedding_history.clone()
        action_embeddings = list(action_embedding_history)
        history_size = int(self.model.predictor.pos_embedding.shape[1])
        width = max(self.block_k, int(self.model.k_max))
        states = []
        for step in range(actions.size(1)):
            block = actions[:, step].reshape(
                actions.size(0), self.block_k, self.model.primitive_dim
            )
            new_action_embedding = self.model._encode_block(block, k=self.block_k, width=width)
            embeddings = self.model._advance(
                embeddings,
                action_embeddings,
                new_action_embedding,
                history_size,
                align=self.model.align,
            )
            states.append(embeddings[:, -1])
        return torch.stack(states, dim=1)

    @staticmethod
    def _gather(sequence: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(sequence.shape[0], device=sequence.device)
        return sequence[rows, index]

    @torch.inference_mode()
    def solve(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        del init_action
        if not bool(getattr(self.model, "has_intent_actor", lambda: False)()):
            raise RuntimeError("actor-path CEM requires a trained intent actor")
        self._check_model_contract()
        expected_width = self.block_k * int(self.model.primitive_dim)
        if self.action_dim != expected_width:
            raise ValueError(
                f"plan_config.action_block must be {self.block_k}: solver action width is {self.action_dim}, expected {expected_width}"
            )
        started = time.perf_counter()
        pixels = info_dict.get("pixels")
        if not (torch.is_tensor(pixels) or isinstance(pixels, np.ndarray)):
            raise ValueError("info_dict must contain batched pixels")
        total_envs = int(pixels.shape[0])
        if total_envs < 1:
            raise ValueError("cannot solve an empty environment batch")
        output_actions = torch.empty(
            total_envs, self.horizon, self.action_dim, device=self.device, dtype=self.dtype
        )
        output_costs = torch.empty(total_envs, device=self.device)
        stats = defaultdict(float)
        stats.update(
            {
                "configured_num_samples": float(self.num_samples),
                "configured_iterations": float(self.n_steps),
                "configured_topk": float(self.topk),
                "configured_temperature": self.temperature,
                "configured_initial_noise_std": float(self.var_scale),
                "configured_block_k": float(self.block_k),
                "configured_residual_penalty": self.residual_penalty,
                "score_arrival_min": float(self.score_mode == "arrival_min"),
                "score_terminal": float(self.score_mode == "terminal"),
                "reference_forced_every_round": 1.0,
                "exact_direct_reference_forced": 1.0,
                "global_best_preserved": 1.0,
                "previous_action_k": float(
                    self._replan_history.length
                    if self._replans > 0 and self._replan_history.kind != "current"
                    else 1
                ),
            }
        )
        for start in range(0, total_envs, self.batch_size):
            end = min(start + self.batch_size, total_envs)
            info = self._slice_info(info_dict, start, end)
            previous_override = self._replan_history.override(
                info, is_replan=self._replans > 0, primitive_dim=int(self.model.primitive_dim)
            )
            embeddings, action_embeddings, goal = self._prepare_context(
                info, previous_block_override=previous_override
            )
            batch = end - start
            direct_kwargs = {}
            if previous_override is not None:
                direct_kwargs["previous_block_override"] = previous_override
            direct_reference = self.model.get_action_vark(
                info, [self.block_k] * self.horizon, **direct_kwargs
            ).reshape(batch, self.horizon, self.action_dim)
            direct_states = self._rollout_actions(embeddings, action_embeddings, direct_reference)
            noise_mean = torch.zeros(
                batch, self.horizon, self.action_dim, device=self.device, dtype=self.dtype
            )
            noise_std = torch.full_like(noise_mean, float(self.var_scale))
            global_noise = torch.zeros_like(noise_mean)
            global_actions = torch.zeros_like(noise_mean)
            global_cost = torch.full(
                (batch,), float("inf"), device=self.device, dtype=torch.float32
            )
            for _ in range(self.n_steps):
                noise = torch.randn(
                    batch,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    device=self.device,
                    dtype=self.dtype,
                    generator=self.torch_gen,
                )
                noise = noise * noise_std[:, None] + noise_mean[:, None]
                noise[:, 0] = 0
                if self.num_samples > 1:
                    noise[:, 1] = global_noise
                actions, states, log_stds = self._rollout_noise(
                    embeddings, action_embeddings, goal, noise
                )
                reference_error = float((actions[:, 0] - direct_reference).abs().max())
                stats["direct_reference_max_abs_error"] = max(
                    stats["direct_reference_max_abs_error"], reference_error
                )
                actions[:, 0] = direct_reference
                states[:, 0] = direct_states
                costs = self._trajectory_cost(states, goal, noise)
                best_cost, best_index = costs.min(dim=1)
                best_actions = self._gather(actions, best_index)
                best_noise = self._gather(noise, best_index)
                improved = best_cost < global_cost
                global_cost = torch.where(improved, best_cost, global_cost)
                global_actions = torch.where(improved[:, None, None], best_actions, global_actions)
                global_noise = torch.where(improved[:, None, None], best_noise, global_noise)
                elite_index = torch.topk(costs, k=self.topk, dim=1, largest=False).indices
                rows = torch.arange(batch, device=self.device)[:, None]
                elites = noise[rows, elite_index]
                elite_mean = elites.mean(dim=1)
                elite_std = elites.std(dim=1, unbiased=False)
                alpha = self.update_alpha
                noise_mean = (1 - alpha) * noise_mean + alpha * elite_mean
                noise_std = ((1 - alpha) * noise_std + alpha * elite_std).clamp(
                    min=self.noise_std_floor, max=self.noise_std_cap
                )
                stats["candidate_action_sequences"] += float(batch * self.num_samples)
                stats["candidate_action_steps"] += float(batch * self.num_samples * self.horizon)
                stats["actor_log_std_sum"] += float(log_stds.mean())
                stats["actor_log_std_observations"] += 1.0
            output_actions[start:end] = global_actions
            output_costs[start:end] = global_cost
        primitive_actions = output_actions.reshape(total_envs, -1, int(self.model.primitive_dim))
        if self._replans == 0:
            self._replan_history.remember(info_dict, primitive_actions)
        self._replans += 1
        stats["solve_time"] = time.perf_counter() - started
        stats["solved_envs"] = float(total_envs)
        observations = stats["actor_log_std_observations"]
        if observations:
            stats["actor_log_std_mean"] = stats["actor_log_std_sum"] / observations
        self.timing_history.append(dict(stats))
        actions = output_actions.detach().cpu()
        return {
            "actions": actions,
            "costs": output_costs.detach().cpu().tolist(),
            "mean": [actions],
            "var": [torch.zeros_like(actions)],
            "timing": dict(stats),
            "evaluation_mode": self.evaluation_mode,
        }

    def timing_summary(self) -> dict[str, float]:
        if not self.timing_history:
            return {}
        keys = sorted({key for row in self.timing_history for key in row})
        summary = {}
        for key in keys:
            values = [float(row.get(key, 0.0)) for row in self.timing_history]
            summary[f"{key}_sum"] = sum(values)
            summary[f"{key}_mean"] = sum(values) / len(values)
        summary["num_solves"] = float(len(self.timing_history))
        return summary
