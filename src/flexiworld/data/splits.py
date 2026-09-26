"""INTACT/LeWM's seeded window-level 90/10 random-split semantics."""

import math
import numpy as np
import torch


def window_split(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("at least two windows are required")
    lengths = [math.floor(size * 0.9), math.floor(size * (1 - 0.9))]
    for index in range(size - sum(lengths)):
        lengths[index % 2] += 1
    ids = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).numpy()
    return ids[: lengths[0]].copy(), ids[lengths[0] :].copy()
