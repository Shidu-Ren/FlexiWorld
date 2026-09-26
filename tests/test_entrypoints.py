"""Check documented entry points without launching training or simulation."""

from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task", ["pusht", "cube", "reacher", "tworoom"])
def test_training_entrypoint_resolves_configs(task):
    result = subprocess.run(
        [sys.executable, "src/train.py", "--cfg", "job", f"task={task}", "seed=42"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    cfg = yaml.safe_load(result.stdout)
    assert cfg["task"] == task
    assert cfg["seed"] == 42
    assert cfg["model"]["_target_"] == "flexiworld.models.world_model.VarKJEPA"
    assert cfg["trainer"]["devices"] == 2
    assert cfg["trainer"]["strategy"] == "ddp"
    assert cfg["loader"]["batch_size"] == 128


@pytest.mark.parametrize("args", [["src/eval.py", "--help"], ["src/prepare_data.py", "--help"]])
def test_documented_cli_help(args):
    result = subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
        check=True, timeout=60,
    )
    assert "--cache-dir" in result.stdout
