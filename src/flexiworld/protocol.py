"""Shared sampling, history, and provenance for paired main evaluations."""

import hashlib
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_eval_anchors(dataset, distance, seed, count=100):
    name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode = np.asarray(dataset.get_col_data(name))
    steps = np.asarray(dataset.get_col_data("step_idx"))
    max_start = {int(ep): int(steps[episode == ep].max()) - distance for ep in np.unique(episode)}
    valid = np.flatnonzero(steps <= np.array([max_start[int(ep)] for ep in episode]))
    # Both planners sample the same start rows, excluding the final eligible row.
    if len(valid) - 1 < count:
        raise ValueError("not enough eligible start rows")
    rng = np.random.default_rng(seed)
    selected = np.sort(valid[rng.choice(len(valid) - 1, count, replace=False)])
    return episode[selected], steps[selected]


def expert_history(dataset, episodes, starts, processor):
    blocks = []
    dim = len(processor.mean_)
    for episode, start in zip(episodes, starts):
        start = int(start)
        block = np.zeros((5, dim), dtype=np.float32)
        lo = max(0, start - 5)
        if start > lo:
            chunk = dataset.load_chunk([int(episode)], [lo], [start])[0]
            actions = np.asarray(chunk["action"])
            if not np.array_equal(np.asarray(chunk["step_idx"]), np.arange(lo, start)):
                raise ValueError("history rows do not match the requested episode/start")
            block[-len(actions) :] = actions
        blocks.append(processor.transform(block))
    return np.asarray(blocks, dtype=np.float32)
