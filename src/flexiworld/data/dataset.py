from __future__ import annotations
import hashlib
import numpy as np
import torch
import stable_worldmodel as swm
from flexiworld.data import windows as vark_data
from flexiworld.data.splits import window_split

N_BLOCKS = 7
K_LO = 1
K_HI = 10
P_UNIFORM = 0.0
PRIMITIVE_DIM = 2
REFERENCE_CHUNK_LENGTH = 5


class VariableKPixelDataset(torch.utils.data.Dataset):
    """Randomly draw windows from a fixed split over a virtual training epoch."""

    def __init__(
        self,
        *,
        split: str,
        seed: int,
        length: int | None,
        cache_dir: str,
        dataset_name: str = "pusht_expert_train.lance",
        action_dim: int = PRIMITIVE_DIM,
        n_blocks: int = N_BLOCKS,
        fixed_total_span: int | None = None,
        k_lo: int = K_LO,
        k_hi: int = K_HI,
        p_uniform: float = P_UNIFORM,
        exclude_uniform_schedule: bool = False,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"unknown split {split!r}")
        self.dataset_name = str(dataset_name)
        self.action_dim = int(action_dim)
        self.cache_dir = str(cache_dir)
        self._pixels = None
        self._pixels = swm.data.load_dataset(
            self.dataset_name,
            transform=None,
            cache_dir=self.cache_dir,
            frameskip=1,
            num_steps=1,
            keys_to_load=["pixels", "action"],
            keys_to_cache=[],
        )
        actions = np.asarray(self._pixels.get_col_data("action"))
        finite = actions[np.isfinite(actions).all(axis=1)]
        mean = finite.mean(axis=0, keepdims=True).astype(np.float32)
        std = finite.std(axis=0, ddof=1, keepdims=True).astype(np.float32)
        if not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("action normalization requires positive finite standard deviations")
        ep_len = np.asarray(self._pixels.lengths, dtype=np.int64)
        ep_offset = np.asarray(self._pixels.offsets, dtype=np.int64)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(
                f"{self.dataset_name} action shape {actions.shape} does not match "
                f"action_dim={self.action_dim}"
            )
        actions = np.nan_to_num(actions, nan=0.0)
        eligible = np.ones(len(ep_len), dtype=bool)
        n_blocks = int(n_blocks)
        k_lo = int(k_lo)
        k_hi = int(k_hi)
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        if not K_LO <= k_lo <= k_hi <= K_HI:
            raise ValueError(f"k range must satisfy {K_LO} <= k_lo <= k_hi <= {K_HI}")
        if fixed_total_span is not None:
            fixed_total_span = int(fixed_total_span)
            if not n_blocks * k_lo <= fixed_total_span <= n_blocks * k_hi:
                raise ValueError(
                    f"fixed_total_span={fixed_total_span} is infeasible for n_blocks={n_blocks}"
                )
        required_span = fixed_total_span or n_blocks * k_hi
        # Match the fixed loader's full final block when enumerating split windows.
        window_steps = required_span + REFERENCE_CHUNK_LENGTH
        eligible = eligible & (ep_len >= window_steps)

        self.split = split
        self.seed = int(seed) + (0 if split == "train" else 1_000_000)
        self.anchor_counts = np.maximum(ep_len - window_steps + 1, 0)
        anchor_count = int(self.anchor_counts.sum())
        train_ids, val_ids = window_split(anchor_count, seed)
        self.anchor_ids = train_ids if split == "train" else val_ids
        if len(self.anchor_ids) == 0:
            raise ValueError("empty training/validation window split")
        self.anchor_pool = vark_data.AnchorPool.from_ids(self.anchor_counts, self.anchor_ids)
        if length is None:
            # The 98% episode mask determines epoch length only; windows use the 90/10 split.
            reference_masks = vark_data.episode_split(len(ep_len), val_frac=0.02, seed=seed)
            reference_mask = reference_masks[0 if split == "train" else 1]
            reference_counts = np.maximum(ep_len - required_span, 0)
            self.length = int(reference_counts[reference_mask].sum())
        else:
            self.length = int(length)
        if self.length < 1:
            raise ValueError("virtual epoch length must be positive")
        self.split_metadata = {
            "scheme": "official_window_90_10",
            "seed": int(seed),
            "split": split,
            "span": required_span,
            "window_steps": window_steps,
            "total_windows": anchor_count,
            "selected_windows": len(self.anchor_ids),
            "samples_per_epoch": self.length,
            "sampling": "with_replacement",
            "indices_sha256": hashlib.sha256(self.anchor_ids.tobytes()).hexdigest(),
        }
        self.n_blocks = n_blocks
        self.fixed_total_span = fixed_total_span
        self.k_lo = k_lo
        self.k_hi = k_hi
        self.p_uniform = float(p_uniform)
        self.exclude_uniform_schedule = bool(exclude_uniform_schedule)
        self.actions_z = ((actions - mean) / std).astype(np.float32)
        self.pad = vark_data.zero_pad_value(mean, std).astype(np.float32)
        self.ep_len = ep_len
        self.ep_offset = ep_offset
        self.eligible = eligible

    def __len__(self) -> int:
        return self.length

    def _pixel_dataset(self):
        if self._pixels is None:
            self._pixels = swm.data.load_dataset(
                self.dataset_name,
                transform=None,
                cache_dir=self.cache_dir,
                num_steps=1,
                frameskip=1,
                keys_to_load=["pixels"],
                keys_to_cache=[],
            )
        return self._pixels

    def _pixel_rows(self, indices: np.ndarray) -> torch.Tensor:
        dataset = self._pixel_dataset()
        batched = getattr(dataset, "__getitems__", None)
        if batched is not None:
            rows = batched(indices.tolist())
            return torch.cat([row["pixels"] for row in rows], dim=0)
        pixels = np.asarray(dataset.get_row_data(indices.tolist())["pixels"])
        tensor = torch.from_numpy(pixels)
        if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def _hdf5_pixel_windows(self, indices: np.ndarray) -> torch.Tensor:
        """Fetch sparse temporal windows through official-style dense slices."""
        dataset = self._pixel_dataset()
        dataset._open()
        source = dataset.h5_file["pixels"]
        output = []
        for requested in np.asarray(indices, dtype=np.int64):
            start = int(requested[0])
            stop = int(requested[-1]) + 1
            dense = np.asarray(source[start:stop])
            selected = dense[requested - start]
            tensor = torch.from_numpy(np.ascontiguousarray(selected))
            if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2)
            output.append(tensor)
        return torch.stack(output)

    def _rng(self, indices: list[int]) -> np.random.Generator:
        digest = hashlib.blake2b(digest_size=8)
        digest.update(self.seed.to_bytes(8, "little", signed=False))
        for index in indices:
            digest.update(int(index).to_bytes(8, "little", signed=False))
        return np.random.default_rng(int.from_bytes(digest.digest(), "little"))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        rng = self._rng(indices)
        windows = vark_data.sample_windows(
            rng,
            self.ep_len,
            self.ep_offset,
            len(indices),
            k_lo=self.k_lo,
            k_hi=self.k_hi,
            n_blocks=self.n_blocks,
            p_uniform=self.p_uniform,
            eligible=self.eligible,
            # Padding width is fixed; the causal encoder reads only the real prefix.
            span_k_hi=K_HI,
            k_prev_fixed=None,
            total_span=self.fixed_total_span,
            exclude_uniform_schedule=self.exclude_uniform_schedule,
            anchor_pool=self.anchor_pool,
        )

        pad = self.pad.reshape(1, 1, 1, self.action_dim)
        sampled_actions = np.where(
            windows.act_mask[..., None],
            self.actions_z[windows.act_idx],
            pad,
        )
        actions = np.broadcast_to(pad, (len(indices), self.n_blocks, K_HI, self.action_dim)).copy()
        actions[:, :, : sampled_actions.shape[2]] = sampled_actions
        actions = actions.reshape(len(indices), self.n_blocks, K_HI * self.action_dim)
        previous = np.where(
            windows.prev_mask[..., None],
            self.actions_z[windows.prev_idx],
            self.pad.reshape(1, 1, self.action_dim),
        ).reshape(len(indices), -1)

        if self.dataset_name.endswith(".h5"):
            pixels = self._hdf5_pixel_windows(windows.lat_idx)
        else:
            flat_rows = windows.lat_idx.reshape(-1)
            unique_rows, inverse = np.unique(flat_rows, return_inverse=True)
            unique_pixels = self._pixel_rows(unique_rows)
            pixels = unique_pixels[torch.from_numpy(inverse)].reshape(
                len(indices), self.n_blocks + 1, *unique_pixels.shape[1:]
            )

        batch = {
            "pixels": pixels.contiguous(),
            "a_pad": torch.from_numpy(np.ascontiguousarray(actions)),
            "k": torch.from_numpy(windows.k.copy()),
            "prev_pad": torch.from_numpy(np.ascontiguousarray(previous)),
            "k_prev": torch.from_numpy(windows.k_prev.copy()),
        }
        return [{key: value[row] for key, value in batch.items()} for row in range(len(indices))]
