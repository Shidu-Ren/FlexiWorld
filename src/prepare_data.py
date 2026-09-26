"""Convert official PushT HDF5 to Lance and verify row identity; no resampling."""

import argparse
from pathlib import Path
import numpy as np
import stable_worldmodel as swm


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    args = parser.parse_args()
    root = args.cache_dir.resolve() / "datasets"
    source = root / "pusht_expert_train.h5"
    target = root / "pusht_expert_train.lance"
    if not source.is_file():
        raise FileNotFoundError(f"download the official dataset first: {source}")
    if not target.exists():
        swm.data.convert(source, target, dest_format="lance")
    a = swm.data.load_dataset(str(source), keys_to_load=["action"])
    b = swm.data.load_dataset(str(target), keys_to_load=["action"])
    for field in ("lengths", "offsets"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    np.testing.assert_allclose(
        np.asarray(a.get_col_data("action"), dtype=np.float32),
        b.get_col_data("action"),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    print("PushT episode layout and action rows verified.")


if __name__ == "__main__":
    run()
