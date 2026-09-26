from __future__ import annotations
from dataclasses import dataclass
import numpy as np

DEFAULT_K_LO = 1
DEFAULT_K_HI = 10
DEFAULT_N_BLOCKS = 7
DEFAULT_P_UNIFORM = 0.0


@dataclass(frozen=True)
class AnchorPool:
    """Allowed local starts, grouped by episode for weighted random sampling."""

    counts: np.ndarray
    offsets: np.ndarray
    starts: np.ndarray

    @classmethod
    def from_ids(cls, anchor_counts: np.ndarray, ids: np.ndarray) -> AnchorPool:
        anchor_counts = np.asarray(anchor_counts, dtype=np.int64)
        ids = np.sort(np.asarray(ids, dtype=np.int64))
        if anchor_counts.ndim != 1 or np.any(anchor_counts < 0):
            raise ValueError("anchor counts must be a non-negative vector")
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("anchor pool must be a non-empty vector")
        if ids[0] < 0 or ids[-1] >= anchor_counts.sum() or np.any(np.diff(ids) == 0):
            raise ValueError("anchor IDs must be unique and within the window range")
        ends = anchor_counts.cumsum()
        episodes = np.searchsorted(ends, ids, side="right")
        counts = np.bincount(episodes, minlength=len(anchor_counts))
        offsets = np.concatenate(([0], counts.cumsum()[:-1]))
        starts = ids - np.concatenate(([0], ends[:-1]))[episodes]
        return cls(counts=counts, offsets=offsets, starts=starts)


@dataclass(frozen=True)
class Windows:
    ep: np.ndarray
    start: np.ndarray
    lat_idx: np.ndarray
    k: np.ndarray
    act_idx: np.ndarray
    act_mask: np.ndarray
    k_prev: np.ndarray
    prev_idx: np.ndarray
    prev_mask: np.ndarray

    @property
    def n(self) -> int:
        return int(self.lat_idx.shape[0])

    @property
    def n_blocks(self) -> int:
        return int(self.k.shape[1])


def _sample_bounded_compositions(
    rng: np.random.Generator,
    n: int,
    n_blocks: int,
    total: int,
    k_lo: int,
    k_hi: int,
    exclude_uniform: bool = False,
) -> np.ndarray:
    if not n_blocks * k_lo <= total <= n_blocks * k_hi:
        raise ValueError(f"total={total} is infeasible for {n_blocks} blocks in [{k_lo}, {k_hi}]")
    if n == 0:
        return np.empty((0, n_blocks), dtype=np.int64)
    if n_blocks == 1:
        return np.full((n, 1), total, dtype=np.int64)
    out = np.empty((n, n_blocks), dtype=np.int64)
    pending = np.arange(n, dtype=np.int64)
    while pending.size:
        prefix = rng.integers(k_lo, k_hi + 1, size=(pending.size, n_blocks - 1), dtype=np.int64)
        final = total - prefix.sum(axis=1)
        accepted = (final >= k_lo) & (final <= k_hi)
        if exclude_uniform and total % n_blocks == 0:
            shared_k = total // n_blocks
            is_uniform = (prefix == shared_k).all(axis=1) & (final == shared_k)
            accepted &= ~is_uniform
        rows = pending[accepted]
        out[rows, :-1] = prefix[accepted]
        out[rows, -1] = final[accepted]
        pending = pending[~accepted]
    return out


def sample_windows(
    rng: np.random.Generator,
    ep_len: np.ndarray,
    ep_offset: np.ndarray,
    n: int,
    k_lo: int = DEFAULT_K_LO,
    k_hi: int = DEFAULT_K_HI,
    n_blocks: int = DEFAULT_N_BLOCKS,
    p_uniform: float = DEFAULT_P_UNIFORM,
    eligible: np.ndarray | None = None,
    span_k_hi: int | None = None,
    k_prev_fixed: int | None = None,
    total_span: int | None = None,
    exclude_uniform_schedule: bool = False,
    anchor_pool: AnchorPool | None = None,
) -> Windows:
    if not 1 <= k_lo <= k_hi:
        raise ValueError(f"need 1 <= k_lo <= k_hi, got {k_lo}, {k_hi}")
    if n_blocks < 1:
        raise ValueError("n_blocks must be positive")
    ep_len = np.asarray(ep_len, dtype=np.int64)
    ep_offset = np.asarray(ep_offset, dtype=np.int64)
    if total_span is not None:
        total_span = int(total_span)
        if not n_blocks * k_lo <= total_span <= n_blocks * k_hi:
            raise ValueError(
                f"total_span={total_span} is infeasible for n_blocks={n_blocks} and k in [{k_lo}, {k_hi}]"
            )
        span = total_span
    else:
        span = n_blocks * int(span_k_hi if span_k_hi is not None else k_hi)
    anchors = ep_len - span
    ok = anchors > 0
    if eligible is not None:
        ok = ok & np.asarray(eligible, dtype=bool)
    if not ok.any():
        raise ValueError(
            f"no episode is long enough for n_blocks={n_blocks} k_hi={k_hi} (need ep_len >= {span + 1}; longest is {int(ep_len.max())})"
        )
    counts = anchors if anchor_pool is None else anchor_pool.counts
    if counts.shape != anchors.shape:
        raise ValueError("anchor pool must match the episode count")
    weights = np.where(ok, counts, 0).astype(np.float64)
    if weights.sum() <= 0:
        raise ValueError("no allowed starts in eligible episodes")
    weights /= weights.sum()
    # Draw episodes, starts, and chunk lengths in this order for reproducibility.
    ep = rng.choice(len(ep_len), size=n, p=weights)
    f_local = (rng.random(n) * counts[ep]).astype(np.int64)
    f_local = np.minimum(f_local, counts[ep] - 1)
    if anchor_pool is not None:
        f_local = anchor_pool.starts[anchor_pool.offsets[ep] + f_local]
    umask = rng.random(n) < p_uniform
    if total_span is None:
        k = rng.integers(k_lo, k_hi + 1, size=(n, n_blocks), dtype=np.int64)
        k_uniform = rng.integers(k_lo, k_hi + 1, size=n, dtype=np.int64)
        k = np.where(umask[:, None], k_uniform[:, None], k)
    else:
        shared_k, remainder = divmod(total_span, n_blocks)
        if p_uniform > 0 and (remainder or not k_lo <= shared_k <= k_hi):
            raise ValueError("p_uniform > 0 requires total_span / n_blocks to be a valid integer k")
        k = np.empty((n, n_blocks), dtype=np.int64)
        k[umask] = shared_k
        variable_rows = np.flatnonzero(~umask)
        k[variable_rows] = _sample_bounded_compositions(
            rng,
            variable_rows.size,
            n_blocks,
            total_span,
            k_lo,
            k_hi,
            exclude_uniform=exclude_uniform_schedule,
        )
    if k_prev_fixed is not None:
        k_prev = np.full(n, int(k_prev_fixed), dtype=np.int64)
        _ = rng.integers(k_lo, k_hi + 1, size=n)
    else:
        k_prev = rng.integers(k_lo, k_hi + 1, size=n, dtype=np.int64)
    start = ep_offset[ep] + f_local
    lat_idx = start[:, None] + np.concatenate(
        [np.zeros((n, 1), dtype=np.int64), np.cumsum(k, axis=1)], axis=1
    )
    slot = np.arange(k_hi, dtype=np.int64)[None, None, :]
    act_mask = slot < k[:, :, None]
    act_idx = np.where(act_mask, lat_idx[:, :-1, None] + slot, lat_idx[:, :-1, None])
    prev_width = max(int(k_hi), int(k_prev.max()))
    flat_slot = np.arange(prev_width, dtype=np.int64)[None, :]
    prev_raw = start[:, None] - k_prev[:, None] + flat_slot
    prev_mask = (flat_slot < k_prev[:, None]) & (prev_raw >= ep_offset[ep][:, None])
    prev_idx = np.where(prev_mask, prev_raw, start[:, None])
    return Windows(
        ep=ep,
        start=start,
        lat_idx=lat_idx,
        k=k,
        act_idx=act_idx,
        act_mask=act_mask,
        k_prev=k_prev,
        prev_idx=prev_idx,
        prev_mask=prev_mask,
    )


def zero_pad_value(action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return -np.asarray(action_mean, dtype=np.float64) / np.asarray(action_std, dtype=np.float64)


def episode_split(n_episodes: int, val_frac: float = 0.02, seed: int = 3072):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_val = max(1, int(round(val_frac * n_episodes)))
    val = np.zeros(n_episodes, dtype=bool)
    val[perm[:n_val]] = True
    return (~val, val)
