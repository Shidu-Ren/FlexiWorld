from __future__ import annotations
from collections.abc import Sequence
import torch
from flexiworld.models.jepa import JEPA


def normalize_schedule(schedule: Sequence[int] | int) -> list[int]:
    if isinstance(schedule, int):
        schedule = [schedule]
    out = [int(k) for k in schedule]
    if not out:
        raise ValueError("schedule must be non-empty")
    if any((k < 1 for k in out)):
        raise ValueError(f"every block must have k >= 1, got {out}")
    return out


def uniform_schedule(total: int, k: int) -> list[int]:
    if total < 1:
        raise ValueError("total must be positive")
    full, rem = divmod(int(total), int(k))
    sched = [int(k)] * full + ([rem] if rem else [])
    return sched or [int(total)]


class VarKJEPA(JEPA):
    def __init__(
        self,
        *args,
        primitive_dim: int = 2,
        k_max: int = 10,
        align: str = "stock",
        prev_mode: str = "k1",
        remaining_steps_head=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.primitive_dim = primitive_dim
        self.k_max = k_max
        self.align = align
        self.prev_mode = prev_mode
        self.remaining_steps_head = remaining_steps_head

    def predicted_steps_remaining(self, current, goal, fallback=None):
        if self.remaining_steps_head is not None:
            return self.remaining_steps_head(current, goal)
        return fallback

    @staticmethod
    def _temporal_kwargs(actor, steps_remaining):
        if bool(getattr(actor, "use_temporal_conditioning", False)):
            return {"steps_remaining": steps_remaining}
        return {}

    def goal_intent(self, current, goal, steps_remaining, block_k=None):
        return super().goal_intent(current, goal, steps_remaining)

    @torch.inference_mode()
    def get_action(
        self, info: dict, horizon: int = 1, prefix_actions: torch.Tensor | None = None
    ) -> torch.Tensor:
        if prefix_actions is not None:
            raise NotImplementedError(
                "VarKJEPA.get_action does not support prefix_actions; all solvers here run warm_start=false"
            )
        plan = self.get_action_vark(info, [5] * int(horizon))
        return plan.reshape(plan.size(0), int(horizon), 5 * self.primitive_dim)

    @property
    def accepts_k(self) -> bool:
        return bool(getattr(self.action_encoder, "accepts_k", False))

    def native_block_size(self) -> int | None:
        patch = getattr(self.action_encoder, "patch_embed", None)
        if patch is None:
            return None
        return int(patch.in_channels) // self.primitive_dim

    def _pad_block(self, block: torch.Tensor, width: int) -> torch.Tensor:
        b, k, d = block.shape
        if k > width:
            raise ValueError(f"block of {k} primitives exceeds pad width {width}")
        if k < width:
            block = torch.cat([block, block.new_zeros(b, width - k, d)], dim=1)
        return block.reshape(b, width * d)

    def _decode_block(
        self,
        z: torch.Tensor,
        intent: torch.Tensor,
        prev_emb: torch.Tensor,
        k: int,
        steps_remaining: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actor = self.intent_actor
        if hasattr(actor, "decode"):
            return actor.decode(
                z, intent, prev_emb, k, **self._temporal_kwargs(actor, steps_remaining)
            )
        flat = actor.action_mean(z, intent, prev_emb)
        block = flat.reshape(z.size(0), -1, self.primitive_dim)
        if block.size(1) != k:
            raise RuntimeError(
                f"fixed-width actor emits {block.size(1)} primitives but the schedule asked for {k}"
            )
        return block

    def _encode_block(self, block: torch.Tensor, k: int, width: int) -> torch.Tensor:
        flat = self._pad_block(block, width).unsqueeze(1)
        if not self.accepts_k:
            return self.action_encoder(flat)[:, 0]
        k_t = torch.full((flat.size(0), 1), k, dtype=torch.long, device=flat.device)
        return self.action_encoder(flat, k=k_t)[:, 0]

    def _advance(
        self,
        embeddings: torch.Tensor,
        act_hist: list[torch.Tensor],
        new_act_emb: torch.Tensor,
        history_size: int,
        align: str = "stock",
    ) -> torch.Tensor:
        size = min(history_size, embeddings.size(1), len(act_hist))
        ctx_emb = embeddings[:, -size:]
        if align == "stock":
            ctx = list(act_hist[-size:])
            ctx[-1] = new_act_emb
        elif align == "aligned":
            ctx = list(act_hist[-(size - 1) :]) + [new_act_emb] if size > 1 else [new_act_emb]
        else:
            raise ValueError(f"unknown align mode: {align}")
        prediction = self.predict(ctx_emb, torch.stack(ctx, dim=1))[:, -1:]
        act_hist.append(new_act_emb)
        return torch.cat([embeddings, prediction], dim=1)

    @torch.inference_mode()
    def get_action_vark(
        self,
        info: dict,
        schedule: Sequence[int] | int,
        pad_width: int | None = None,
        previous_block_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sched = normalize_schedule(schedule)
        if self.intent_actor is None:
            raise RuntimeError("variable-k Direct requires an intent actor")
        if not self.accepts_k:
            native = self.native_block_size()
            if native is None or set(sched) != {native}:
                raise RuntimeError(
                    f"action encoder is fixed at k={native} but the schedule is {sched}; use a VarKActionEncoder for anything else"
                )
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        info = {k: v.to(device) if torch.is_tensor(v) else v for k, v in info.items()}
        batch = info["pixels"].size(0)
        native = self.native_block_size() if not self.accepts_k else None
        width = int(pad_width or native or max(max(sched), self.k_max))
        if not self.actor_warmstart:
            return torch.zeros(batch, sum(sched), self.primitive_dim, device=device, dtype=dtype)
        if previous_block_override is not None:
            prev_block = previous_block_override.to(device=device, dtype=dtype)
            if prev_block.ndim != 3 or prev_block.size(0) != batch:
                raise ValueError(
                    f"previous_block_override must have shape (B,k,{self.primitive_dim}), got {tuple(prev_block.shape)}"
                )
            if prev_block.size(-1) != self.primitive_dim:
                raise ValueError(
                    f"previous_block_override has the wrong primitive dimension: expected {self.primitive_dim}, got {prev_block.size(-1)}"
                )
            prev_k = int(prev_block.size(1))
            if not 1 <= prev_k <= width:
                raise ValueError(f"previous block length {prev_k} is outside [1,{width}]")
        else:
            prev = info.get("action")
            if prev is None:
                raise ValueError("action history is required")
            prev = prev.to(device=device, dtype=dtype)
            if prev.ndim == 2:
                prev = prev[:, None]
            prev_block = prev[:, -1:, : self.primitive_dim]
            prev_k = 1
            if self.prev_mode == "tile":
                repeats = 5
                prev_block = prev_block.repeat(1, repeats, 1)
                prev_k = repeats
            elif self.prev_mode != "k1":
                raise ValueError(f"unknown prev_mode: {self.prev_mode}")
        initial = {k: v for k, v in info.items() if torch.is_tensor(v)}
        initial.pop("action", None)
        embeddings = self.encode(initial)["emb"]
        goal = self._encode_goal(info)
        history_size = self.predictor.pos_embedding.size(1)
        prev_emb = self._encode_block(prev_block, prev_k, width)
        act_embs = [prev_emb] * embeddings.size(1)
        plan, intent_norms = ([], [])
        for step, k in enumerate(sched):
            current = embeddings[:, -1]
            intent = self.goal_intent(current, goal, len(sched) - step, block_k=k)
            temporal_steps = self.predicted_steps_remaining(
                current, goal, fallback=current.new_full((batch,), sum(sched[step:]))
            )
            block = self._decode_block(
                current, intent, act_embs[-1], k, steps_remaining=temporal_steps
            )
            plan.append(block)
            intent_norms.append(intent.norm(dim=-1).mean())
            embeddings = self._advance(
                embeddings,
                act_embs,
                self._encode_block(block, k, width),
                history_size,
                align=self.align,
            )
        self.last_direct_diagnostics = {
            "intent_norm": float(torch.stack(intent_norms).mean()),
            "terminal_latent_error": float((embeddings[:, -1] - goal).norm(dim=-1).mean()),
            "forward_calls": float(len(sched)),
            "candidate_sequences": 0.0,
            "reach": float(sum(sched)),
            "depth": float(len(sched)),
        }
        self.last_direct_endpoint = embeddings[:, -1].detach()
        return torch.cat(plan, dim=1)
