import dataclasses

import h5py
import numpy as np
import pytest
import torch
from stable_worldmodel.data.dataset import Dataset

from flexiworld.data.dataset import VariableKPixelDataset
from flexiworld.data.sampler import HomogeneousDistributedBatchSampler
from flexiworld.data.splits import window_split
from flexiworld.data.windows import AnchorPool, sample_windows
from train import PUSHT_TRAIN_WINDOWS, VAL_WINDOWS


@pytest.mark.parametrize("span", [35, 55, 75])
@pytest.mark.parametrize("seed", [0, 42, 3072])
def test_full_anchor_pool_preserves_original_draws(span, seed):
    lengths = np.array([20, 100, 130, 160])
    counts = np.maximum(lengths - span, 0)
    pool = AnchorPool.from_ids(counts, np.arange(counts.sum()))
    kwargs = dict(
        ep_len=lengths, ep_offset=np.array([0, 20, 120, 250]), n=64,
        n_blocks=span // 5, total_span=span, exclude_uniform_schedule=True,
    )
    original_rng, pool_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    original = sample_windows(original_rng, **kwargs)
    actual = sample_windows(pool_rng, anchor_pool=pool, **kwargs)
    for field in dataclasses.fields(original):
        np.testing.assert_array_equal(getattr(original, field.name), getattr(actual, field.name))
    assert original_rng.bit_generator.state == pool_rng.bit_generator.state


def test_pool_draws_with_replacement_and_weights_episodes_by_allowed_starts():
    counts = np.array([10, 10])
    ids = np.array([0, 2, 3, 7, 9, 15])
    pool = AnchorPool.from_ids(counts, ids)
    windows = sample_windows(
        np.random.default_rng(42), np.array([45, 45]), np.array([0, 45]),
        20_000, k_lo=5, k_hi=5, n_blocks=7, total_span=35, anchor_pool=pool,
    )
    sampled_ids = windows.ep * 10 + windows.start - windows.ep * 45
    assert set(sampled_ids) == set(ids)
    assert len(np.unique(sampled_ids)) < len(sampled_ids)
    np.testing.assert_allclose(
        [(sampled_ids == i).mean() for i in ids], np.full(6, 1 / 6), atol=0.015, rtol=0,
    )


@pytest.mark.parametrize("ids", [[], [-1], [20], [1, 1]])
def test_pool_rejects_invalid_anchor_ids(ids):
    with pytest.raises(ValueError):
        AnchorPool.from_ids(np.array([10, 10]), np.array(ids))


def make_dataset_file(path, lengths):
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    with h5py.File(path, "w") as f:
        f["ep_len"] = lengths
        f["ep_offset"] = offsets
        f["action"] = np.random.default_rng(0).normal(size=(sum(lengths), 2)).astype("float32")
        f["pixels"] = np.zeros((sum(lengths), 16, 16, 3), dtype="uint8")


@pytest.mark.parametrize("span", [35, 55, 75])
@pytest.mark.parametrize("seed", [0, 42, 3072])
def test_split_windows_match_fixed_loader_boundaries(tmp_path, span, seed):
    lengths = span + np.array([0, 1, 4, 5, 6, 12, 35])
    offsets = np.concatenate(([0], lengths.cumsum()[:-1]))
    path = tmp_path / "boundaries.h5"
    make_dataset_file(path, lengths)
    reference = Dataset(lengths, offsets, num_steps=span // 5 + 1, frameskip=5)
    reference_splits = torch.utils.data.random_split(
        reference, [0.9, 1 - 0.9], generator=torch.Generator().manual_seed(seed),
    )
    all_pairs = np.asarray(reference.clip_indices)
    for split, expected in zip(("train", "val"), reference_splits):
        ds = VariableKPixelDataset(
            split=split, seed=seed, length=512, cache_dir=str(tmp_path),
            dataset_name=str(path), fixed_total_span=span, n_blocks=span // 5,
            exclude_uniform_schedule=True,
        )
        np.testing.assert_array_equal(ds.anchor_ids, expected.indices)
        np.testing.assert_array_equal(ds.anchor_counts, [0, 0, 0, 1, 2, 8, 31])
        pool_pairs = np.column_stack((
            np.repeat(np.arange(len(lengths)), ds.anchor_pool.counts),
            ds.anchor_pool.starts,
        ))
        np.testing.assert_array_equal(pool_pairs, all_pairs[np.sort(expected.indices)])
        assert ds.split_metadata["window_steps"] == span + 5
        assert ds.split_metadata["total_windows"] == len(reference)
        assert len(ds) == 512
        windows = sample_windows(
            ds._rng(list(range(512))), lengths, offsets, 512,
            n_blocks=span // 5, total_span=span, exclude_uniform_schedule=True,
            eligible=ds.eligible, anchor_pool=ds.anchor_pool,
        )
        sampled_pairs = np.column_stack((windows.ep, windows.start - offsets[windows.ep]))
        assert set(map(tuple, sampled_pairs)) <= set(map(tuple, pool_pairs))
        assert np.all(sampled_pairs[:, 1] + span + 5 <= lengths[windows.ep])
        np.testing.assert_array_equal(windows.k.sum(axis=1), np.full(512, span))


def test_virtual_indices_resample_only_the_requested_split(tmp_path, monkeypatch):
    path = tmp_path / "tiny.h5"
    make_dataset_file(path, [100, 100, 100])
    seen = []

    def capture(*args, **kwargs):
        result = sample_windows(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr("flexiworld.data.dataset.vark_data.sample_windows", capture)
    split_draws = []
    for split in ("train", "val"):
        ds = VariableKPixelDataset(
            split=split, seed=42, length=512, cache_dir=str(tmp_path),
            dataset_name=str(path), fixed_total_span=35, exclude_uniform_schedule=True,
        )
        expected_ids = window_split(183, 42)[0 if split == "train" else 1]
        np.testing.assert_array_equal(ds.anchor_ids, expected_ids)
        assert len(ds) == 512
        assert ds.split_metadata["samples_per_epoch"] == 512
        assert ds.split_metadata["selected_windows"] == len(expected_ids)
        indices = list(range(256, 384))
        ds.__getitems__(indices)
        first = seen[-1]
        flat_ids = first.ep * 61 + first.start - first.ep * 100
        assert set(flat_ids) <= set(expected_ids)
        assert len(set(flat_ids)) < len(flat_ids)
        ds.__getitems__(indices)
        np.testing.assert_array_equal(first.start, seen[-1].start)
        np.testing.assert_array_equal(first.k, seen[-1].k)
        ds.__getitems__(list(reversed(indices)))
        assert not np.array_equal(first.start, seen[-1].start)
        split_draws.append(set(flat_ids))
    assert not split_draws[0] & split_draws[1]


def test_short_virtual_epoch_does_not_truncate_anchor_pool(tmp_path):
    path = tmp_path / "tiny.h5"
    make_dataset_file(path, [100, 100])
    ds = VariableKPixelDataset(
        split="train", seed=0, length=2, cache_dir=str(tmp_path),
        dataset_name=str(path), fixed_total_span=35,
    )
    assert len(ds) == 2
    np.testing.assert_array_equal(ds.anchor_ids, window_split(122, 0)[0])


@pytest.mark.parametrize("seed", [0, 42, 3072])
def test_reference_hdf5_epoch_lengths(tmp_path, monkeypatch, seed):
    lengths = np.full(10_000, 201)

    class MetadataDataset:
        def __init__(self):
            self.lengths = lengths
            self.offsets = np.arange(10_000) * 201

        def get_col_data(self, name):
            assert name == "action"
            return np.array([[0, 1], [2, 3]], dtype=np.float32)

    monkeypatch.setattr("stable_worldmodel.data.load_dataset", lambda *a, **k: MetadataDataset())
    counts = []
    for span, expected in zip((35, 55, 75), (1_626_800, 1_430_800, 1_234_800)):
        ds = VariableKPixelDataset(
            split="train", seed=seed, length=None, cache_dir=str(tmp_path),
            dataset_name="fixture.h5", fixed_total_span=span, n_blocks=span // 5,
        )
        assert len(ds) == expected
        assert len(ds) != len(ds.anchor_ids)
        counts.append(len(ds))
    assert len(HomogeneousDistributedBatchSampler(counts, 128, 2, 0, seed)) == 16_766


def test_reference_pusht_epoch_and_validation_budgets():
    assert PUSHT_TRAIN_WINDOWS == {35: 1_447_219, 55: 1_112_071, 75: 794_830}
    assert VAL_WINDOWS == 25_600
    for rank in (0, 1):
        sampler = HomogeneousDistributedBatchSampler(
            list(PUSHT_TRAIN_WINDOWS.values()), 128, 2, rank, 3072,
        )
        assert len(sampler) == 13_101
