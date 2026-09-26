from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import nn
from flexiworld.models import layers as intact_module


def sinusoidal(positions: torch.Tensor, dim: int, base: float = 10_000.0) -> torch.Tensor:
    """Continuous sinusoidal embedding.

    Unlike a learned lookup table, this representation accepts arbitrary positions.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(base) * torch.arange(half, device=positions.device, dtype=torch.float32) / half
    )
    ang = positions.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class VarKActionEncoder(nn.Module):
    """Embed a block of ``k`` primitives into one 192-d vector, for any ``k``.

    Each primitive becomes a token; a small causal transformer mixes them; the output is
    read at the last *real* token, index ``k-1``.

    **Prefix property.**  Causal attention plus a readout at ``k-1`` means the output
    provably cannot depend on any slot ``>= k``.  That is what lets blocks of different
    ``k`` sit in one padded batch with no attention mask -- and it is asserted bitwise in
    the tests at several pad values, because it is the assumption every mixed-k batch
    rests on.

    ``k_max`` is metadata only and is never enforced: ``k > k_max`` must produce finite
    output so extrapolation stays measurable.
    """

    accepts_k = True

    def __init__(
        self,
        action_dim: int = 2,
        emb_dim: int = 192,
        d_tok: int = 64,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 16,
        mlp_dim: int = 256,
        k_max: int = 10,
        use_dt: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.d_tok = d_tok
        self.k_max = k_max
        self.use_dt = use_dt

        self.tokenise = nn.Linear(action_dim, d_tok)
        self.blocks = nn.ModuleList(
            CausalBlock(d_tok, heads, dim_head, mlp_dim) for _ in range(depth)
        )
        self.project = nn.Sequential(
            nn.Linear(d_tok, 4 * emb_dim), nn.SiLU(), nn.Linear(4 * emb_dim, emb_dim)
        )
        if use_dt:
            self.dt_mlp = nn.Sequential(
                nn.Linear(d_tok, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
            )

    def encode_blocks(
        self, x: torch.Tensor, k: torch.Tensor, dt_override: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``x (N, L, action_dim)``, ``k (N,)`` -> ``(N, emb_dim)``."""
        n, length, _ = x.shape
        if int(k.min()) < 1 or int(k.max()) > length:
            raise ValueError(f"k must lie in [1, {length}], got [{int(k.min())}, {int(k.max())}]")
        h = self.tokenise(x.float())
        pos = torch.arange(length, device=x.device)
        h = h + sinusoidal(pos, self.d_tok).unsqueeze(0)
        for blk in self.blocks:
            h = blk(h)
        out = self.project(h[torch.arange(n, device=x.device), k - 1])
        if self.use_dt:
            dt = k if dt_override is None else dt_override
            out = out + self.dt_mlp(sinusoidal(dt.to(x.device), self.d_tok))
        return out

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor | int | None = None,
        dt_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x`` is ``(B, T, L*action_dim)`` or ``(N, L*action_dim)``.

        ``k=None`` infers the block width from the tensor itself, so a fixed-width caller
        (stock ``rollout``/``get_cost``) needs no change at all.
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)
        b, t, flat = x.shape
        if flat % self.action_dim:
            raise ValueError(f"width {flat} is not a multiple of action_dim {self.action_dim}")
        length = flat // self.action_dim
        tokens = x.reshape(b * t, length, self.action_dim)

        if k is None:
            k_flat = torch.full((b * t,), length, dtype=torch.long, device=x.device)
        elif torch.is_tensor(k):
            k_flat = k.to(x.device, torch.long).reshape(-1)
            if k_flat.numel() == 1:
                k_flat = k_flat.expand(b * t)
            elif k_flat.numel() != b * t:
                raise ValueError(f"k has {k_flat.numel()} entries, expected {b * t}")
        else:
            k_flat = torch.full((b * t,), int(k), dtype=torch.long, device=x.device)

        dt_flat = None
        if dt_override is not None:
            dt_flat = torch.as_tensor(dt_override, device=x.device).reshape(-1)
            if dt_flat.numel() == 1:
                dt_flat = dt_flat.expand(b * t)

        out = self.encode_blocks(tokens, k_flat, dt_flat).reshape(b, t, self.emb_dim)
        return out.squeeze(1) if squeeze else out


class CausalBlock(nn.Module):
    """Pre-norm causal transformer block built from INTACT's own primitives.

    ``module.Attention`` and ``module.FeedForward`` each apply their own LayerNorm on
    entry, so this adds none.  Causality is not decorative: the prefix property of
    :class:`VarKActionEncoder` and the autoregressive factorisation of
    :class:`ARPrimitiveActor` both depend on it.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.attn = intact_module.Attention(dim, heads=heads, dim_head=dim_head)
        self.mlp = intact_module.FeedForward(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x, causal=True)
        return x + self.mlp(x)


class ARPrimitiveActor(nn.Module):
    """Autoregressive Gaussian actor conditioned on state, intent, and action history.

    The full method uses four prefix tokens [z, intent, z*intent, previous_action_embedding].
    Requested chunk length controls the number of emitted primitives, not an input token.
    Length-token and temporal conditioning are optional and disabled by default.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        action_emb_dim: int = 192,
        action_dim: int = 2,
        d_model: int = 192,
        depth: int = 3,
        heads: int = 4,
        dim_head: int = 48,
        mlp_dim: int = 768,
        k_max: int = 10,
        use_k_token: bool = False,
        use_temporal_conditioning: bool = False,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model
        self.k_max = k_max
        self.use_k_token = use_k_token
        self.use_temporal_conditioning = bool(use_temporal_conditioning)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.slot_z = nn.Linear(embed_dim, d_model)
        self.slot_m = nn.Linear(embed_dim, d_model)
        self.slot_zm = nn.Linear(embed_dim, d_model)
        self.slot_a = nn.Linear(action_emb_dim, d_model)
        self.n_prefix = 5 if use_k_token else 4
        if use_k_token:
            self.k_mlp = nn.Sequential(
                nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
            )
        if self.use_temporal_conditioning:
            self.temporal_mlp = nn.Sequential(
                nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
            )
            # Initialize temporal conditioning with zero contribution to the prefix.
            nn.init.zeros_(self.temporal_mlp[-1].weight)
            nn.init.zeros_(self.temporal_mlp[-1].bias)
        self.prefix_type = nn.Parameter(torch.zeros(self.n_prefix, d_model))
        self.action_embed = nn.Linear(action_dim, d_model)
        self.blocks = nn.ModuleList(
            CausalBlock(d_model, heads, dim_head, mlp_dim) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2 * action_dim)

    # -- conditioning -----------------------------------------------------------------

    def prefix(
        self,
        z: torch.Tensor,
        intent: torch.Tensor,
        prev_emb: torch.Tensor,
        k: torch.Tensor,
        steps_remaining: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(N, n_prefix, d_model)`` conditioning tokens."""
        if z.shape != intent.shape:
            raise ValueError(f"z {tuple(z.shape)} and intent {tuple(intent.shape)} must match")
        toks = [
            self.slot_z(z),
            self.slot_m(intent),
            self.slot_zm(z * intent),
            self.slot_a(prev_emb),
        ]
        if self.use_temporal_conditioning:
            if steps_remaining is None:
                steps_remaining = k
            steps = torch.as_tensor(steps_remaining, device=z.device, dtype=z.dtype)
            steps = steps.reshape(-1)
            if steps.numel() == 1:
                steps = steps.expand(z.size(0))
            if steps.numel() != z.size(0):
                raise ValueError(
                    f"steps_remaining has {steps.numel()} entries, expected {z.size(0)}"
                )
            # log1p keeps 1--150 primitive steps numerically compact while remaining
            # continuous, so a variable-k schedule never changes the unit of time.
            temporal = sinusoidal(torch.log1p(steps.clamp_min(0)), self.d_model)
            toks[1] = toks[1] + self.temporal_mlp(temporal.to(z.dtype))
        if self.use_k_token:
            toks.append(self.k_mlp(sinusoidal(k.to(z.device), self.d_model)))
        return torch.stack(toks, dim=1) + self.prefix_type.unsqueeze(0)

    def _trunk(self, tokens: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    def _conditioning_prefix(self, z, intent, prev_emb, k, steps_remaining=None) -> torch.Tensor:
        """Pass temporal arguments to the prefix when temporal conditioning is enabled."""
        if self.use_temporal_conditioning:
            return self.prefix(z, intent, prev_emb, k, steps_remaining=steps_remaining)
        return self.prefix(z, intent, prev_emb, k)

    def _params(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.head(hidden).chunk(2, dim=-1)
        return mean, log_std.clamp(self.min_log_std, self.max_log_std)

    # -- training ---------------------------------------------------------------------

    def forward(
        self,
        z,
        intent,
        prev_emb,
        actions,
        k,
        prefix_noise: float = 0.0,
        conditioning_actions: torch.Tensor | None = None,
        steps_remaining: torch.Tensor | None = None,
    ):
        """Teacher-forced pass.

        ``actions (N, L, action_dim)`` are the demonstrated primitives.  Returns
        ``mean``/``log_std`` of the same shape: position ``j`` predicts ``actions[:, j]``
        from the prefix and ``actions[:, :j]`` only.

        The full objective supplies generated or expert conditioning prefixes and
        keeps expert actions as targets. ``prefix_noise`` is disabled in the full method.
        """
        pre = self._conditioning_prefix(z, intent, prev_emb, k, steps_remaining=steps_remaining)
        # Shift right: position j must not see actions[:, j].  The prefix already
        # occupies the slot that would otherwise hold a BOS token.
        shifted = None
        if actions.size(1) > 1:
            conditioning = (conditioning_actions if conditioning_actions is not None else actions)[
                :, :-1
            ].float()
            if prefix_noise > 0 and self.training:
                conditioning = conditioning + prefix_noise * torch.randn_like(conditioning)
            shifted = self.action_embed(conditioning)
        seq = pre if shifted is None else torch.cat([pre, shifted], dim=1)
        pos = torch.arange(seq.size(1), device=seq.device)
        seq = seq + sinusoidal(pos, self.d_model).unsqueeze(0)
        hidden = self._trunk(seq)[:, self.n_prefix - 1 :]
        return self._params(hidden)

    def nll(
        self,
        z,
        intent,
        prev_emb,
        actions,
        k,
        reduction: str = "mean",
        prefix_noise: float = 0.0,
        student_p: float = 0.0,
        steps_remaining: torch.Tensor | None = None,
    ):
        """Gaussian NLL over the first ``k`` positions only.

        Masking rather than truncating keeps mixed-``k`` batches in one tensor; the value
        is by construction independent of anything stored at positions ``>= k``.

        ``student_p`` is the per-sample probability of conditioning on the actor's OWN
        greedy prefix (detached) instead of the expert prefix -- targets stay expert
        either way.  0.0 is classic teacher forcing; 0.5 trains half of every batch under
        exactly the conditioning distribution deployment produces.
        """
        conditioning = None
        if student_p > 0 and self.training:
            with torch.no_grad():
                own = self.decode(
                    z.detach(),
                    intent.detach(),
                    prev_emb.detach(),
                    actions.size(1),
                    steps_remaining=(
                        steps_remaining.detach()
                        if torch.is_tensor(steps_remaining)
                        else steps_remaining
                    ),
                )
            pick = (torch.rand(actions.size(0), 1, 1, device=actions.device) < student_p).float()
            conditioning = pick * own + (1 - pick) * actions.float()
        mean, log_std = self.forward(
            z,
            intent,
            prev_emb,
            actions,
            k,
            prefix_noise=prefix_noise,
            conditioning_actions=conditioning,
            steps_remaining=steps_remaining,
        )
        per_dim = 0.5 * ((actions - mean).square() * torch.exp(-2 * log_std) + 2 * log_std)
        per_step = per_dim.mean(dim=-1)
        mask = (
            torch.arange(actions.size(1), device=actions.device).unsqueeze(0)
            < k.to(actions.device).unsqueeze(1)
        ).float()
        per_sample = (per_step * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        if reduction == "none":
            return {"loss": per_sample, "mean": mean, "log_std": log_std}
        if reduction != "mean":
            raise ValueError(f"unsupported reduction: {reduction}")
        return {
            "loss": per_sample.mean(),
            "mean": mean,
            "log_std": log_std,
            "mae": ((mean - actions).abs().mean(dim=-1) * mask).sum() / mask.sum(),
        }

    # -- deployment -------------------------------------------------------------------

    @torch.no_grad()
    def action_mean(
        self, z, intent, prev_emb, k: int = 5, steps_remaining: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Stock-JEPA Actionable interface: one greedy k-block, flattened to ``(N, k*2)``.

        ``jepa.JEPA.get_action`` -- reached through ``prepare_init_action`` by the
        Guarded/CEM warm-start path -- calls ``intent_actor.action_mean``; without this
        the AR arms cannot serve as a search's Direct reference plan.  The vark
        controllers never call it (they use ``decode``/``sample_decode`` directly).
        """
        return self.decode(z, intent, prev_emb, k, steps_remaining=steps_remaining).reshape(
            z.size(0), -1
        )

    @torch.no_grad()
    def sample_decode(
        self,
        z,
        intent,
        prev_emb,
        k: int,
        n: int,
        temperature: float = 1.0,
        steps_remaining: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``n`` coherent action blocks per condition -> ``(N, n, k, action_dim)``.

        This is the capability the autoregressive factorisation actually buys and greedy
        mean decoding throws away.  With a multimodal ``p(a_1)``, the mean is *between*
        modes -- not itself a valid action -- and every later position then conditions on
        that off-manifold prefix.  Sampling keeps each sequence on one mode: position
        ``j+1`` sees the action actually taken at ``j``.  The one-shot MLP cannot do this
        at all; its slots are conditionally independent, so its "samples" are incoherent
        by construction.  A world model then scores the ``n`` candidates (one predictor
        step each) and control executes the best -- the paper's own Guarded pattern, at
        block level, driven by a real sequence distribution.
        """
        if k < 1 or n < 1:
            raise ValueError("k and n must be positive")
        big_z = z.repeat_interleave(n, dim=0)
        big_m = intent.repeat_interleave(n, dim=0)
        big_p = prev_emb.repeat_interleave(n, dim=0)
        rows = big_z.size(0)
        k_t = torch.full((rows,), k, dtype=torch.long, device=z.device)
        big_steps = None
        if steps_remaining is not None:
            base_steps = torch.as_tensor(steps_remaining, device=z.device).reshape(-1)
            if base_steps.numel() == 1:
                base_steps = base_steps.expand(z.size(0))
            big_steps = base_steps.repeat_interleave(n)
        seq = self._conditioning_prefix(big_z, big_m, big_p, k_t, steps_remaining=big_steps)
        emitted = []
        for _ in range(k):
            pos = torch.arange(seq.size(1), device=seq.device)
            hidden = self._trunk(seq + sinusoidal(pos, self.d_model).unsqueeze(0))
            mean, log_std = self._params(hidden[:, -1])
            action = mean + temperature * torch.exp(log_std) * torch.randn_like(mean)
            emitted.append(action)
            seq = torch.cat([seq, self.action_embed(action).unsqueeze(1)], dim=1)
        out = torch.stack(emitted, dim=1)  # (N*n, k, action_dim)
        return out.reshape(z.size(0), n, k, self.action_dim)

    @torch.no_grad()
    def decode_with_stats(
        self,
        z,
        intent,
        prev_emb,
        k: int,
        steps_remaining: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy decode plus conditional ``log_std`` for every emitted primitive.

        Returning the distribution statistics from the same autoregressive pass avoids
        a second decode when inference needs to compare nested action prefixes.
        """
        if k < 1:
            raise ValueError("k must be positive")
        n = z.size(0)
        k_t = torch.full((n,), k, dtype=torch.long, device=z.device)
        seq = self._conditioning_prefix(z, intent, prev_emb, k_t, steps_remaining=steps_remaining)
        emitted, log_stds = [], []
        for _ in range(k):
            pos = torch.arange(seq.size(1), device=seq.device)
            hidden = self._trunk(seq + sinusoidal(pos, self.d_model).unsqueeze(0))
            mean, log_std = self._params(hidden[:, -1])
            emitted.append(mean)
            log_stds.append(log_std)
            seq = torch.cat([seq, self.action_embed(mean).unsqueeze(1)], dim=1)
        return torch.stack(emitted, dim=1), torch.stack(log_stds, dim=1)

    @torch.no_grad()
    def decode(
        self, z, intent, prev_emb, k: int, steps_remaining: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Greedy (mean) decode of ``k`` primitives -> ``(N, k, action_dim)``.

        With the full-method config (no length token), shorter decodes are prefixes
        of longer decodes under the same conditioning context.
        """
        emitted, _ = self.decode_with_stats(z, intent, prev_emb, k, steps_remaining=steps_remaining)
        return emitted
