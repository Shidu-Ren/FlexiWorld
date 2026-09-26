import h5py
import numpy as np
import stable_worldmodel as swm
from flexiworld.data.dataset import VariableKPixelDataset
from flexiworld.data.splits import window_split


def test_lance_conversion_and_training_reader(tmp_path):
    source, target = tmp_path / "source.h5", tmp_path / "copy.lance"
    with h5py.File(source, "w") as f:
        f["ep_len"] = np.array([44, 44])
        f["ep_offset"] = np.array([0, 44])
        f["episode_idx"] = np.repeat([0, 1], 44)
        f["step_idx"] = np.tile(np.arange(44), 2)
        f["action"] = np.random.default_rng(0).normal(size=(88, 2)).astype("float32")
        f["pixels"] = np.zeros((88, 16, 16, 3), dtype="uint8")
    swm.data.convert(source, target, dest_format="lance", progress=False)
    ds = VariableKPixelDataset(
        split="train",
        seed=0,
        length=None,
        cache_dir=str(tmp_path),
        dataset_name=str(target),
        n_blocks=7,
        fixed_total_span=35,
        exclude_uniform_schedule=True,
    )
    np.testing.assert_array_equal(ds.anchor_ids, window_split(10, 0)[0])
    assert len(ds) == 9
    item = ds[0]
    assert item["pixels"].shape == (8, 3, 16, 16)
    assert item["k"].sum().item() == 35
