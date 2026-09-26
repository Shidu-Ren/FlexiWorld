from types import SimpleNamespace
import h5py
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from flexiworld.data.dataset import VariableKPixelDataset
from flexiworld.models.layers import SIGReg
from train_step import forward


def test_hdf5_loader_keeps_split_separate_from_epoch_length(tmp_path):
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as f:
        f["ep_len"] = np.array([100, 100])
        f["ep_offset"] = np.array([0, 100])
        f["action"] = np.random.default_rng(0).normal(size=(200, 2)).astype("float32")
        f["pixels"] = np.zeros((200, 16, 16, 3), dtype="uint8")
    kwargs = dict(
        seed=0,
        length=None,
        cache_dir=str(tmp_path),
        dataset_name=str(path),
        n_blocks=7,
        fixed_total_span=35,
        exclude_uniform_schedule=True,
    )
    train = VariableKPixelDataset(split="train", **kwargs)
    val = VariableKPixelDataset(split="val", **kwargs)
    assert not set(train.anchor_ids) & set(val.anchor_ids)
    assert len(train) + len(val) == 130
    assert len(train.anchor_ids) + len(val.anchor_ids) == 122
    assert len(train) != len(train.anchor_ids)
    batch = train.__getitems__([0, 1])
    assert batch[0]["pixels"].shape == (8, 3, 16, 16)
    assert batch[0]["k"].sum().item() == 35
    assert torch.equal(batch[0]["k"], train.__getitems__([0, 1])[0]["k"])


def test_complete_training_objective_backward():
    cfg = OmegaConf.load("src/configs/train/flexiworld.yaml")
    cfg.model = OmegaConf.load("src/configs/train/model/flexiworld.yaml")
    # Use the same model constructors, with a small image and shallower network.
    cfg.img_size = 28
    cfg.model.predictor.depth = 1
    model = hydra.utils.instantiate(cfg.model)
    runner = SimpleNamespace(
        model=model, sigreg=SIGReg(knots=5, num_proj=8), log=lambda *args, **kwargs: None
    )
    batch = {
        "pixels": torch.randint(0, 256, (2, 3, 3, 28, 28), dtype=torch.uint8),
        "a_pad": torch.randn(2, 2, 20),
        "prev_pad": torch.randn(2, 20),
        "k": torch.tensor([[3, 7], [4, 6]]),
        "k_prev": torch.tensor([5, 5]),
    }
    loss = forward(runner, batch, "train", cfg)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    for component in ("encoder", "action_encoder", "intent_actor", "predictor"):
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in getattr(model, component).parameters()
        )


def test_two_stage_direct_and_arcem_interfaces():
    import gymnasium as gym
    import stable_worldmodel as swm
    from flexiworld.planning.arcem import ActorPathCEMSolver
    from flexiworld.planning.direct import VarKDirectSolver

    cfg = OmegaConf.load("src/configs/train/flexiworld.yaml")
    cfg.model = OmegaConf.load("src/configs/train/model/flexiworld.yaml")
    cfg.img_size = 28
    cfg.model.predictor.depth = 1
    model = hydra.utils.instantiate(cfg.model).eval()
    info = {
        "pixels": torch.randn(2, 1, 3, 28, 28),
        "goal": torch.randn(2, 1, 3, 28, 28),
        "action": torch.zeros(2, 1, 2),
        "_expert_previous_block": torch.zeros(2, 5, 2),
    }
    for search in (False, True):
        kwargs = dict(
            model=model,
            num_samples=4,
            topk=2,
            n_steps=2,
            replan_previous_mode="carry5",
            device="cpu",
        )
        solver = (
            ActorPathCEMSolver(**kwargs) if search else VarKDirectSolver(**kwargs, schedule=[5, 5])
        )
        plan = swm.PlanConfig(
            horizon=2 if search else 10,
            receding_horizon=2 if search else 10,
            action_block=5 if search else 1,
            warm_start=False,
        )
        solver.configure(action_space=gym.spaces.Box(-1, 1, (2, 2)), n_envs=2, config=plan)
        first = solver.solve(info)["actions"]
        assert first.reshape(2, -1, 2).shape == (2, 10, 2)
        assert torch.isfinite(first).all()
        active = {key: value[1:] for key, value in info.items() if not key.startswith("_")}
        second = solver.solve(active)["actions"]
        assert second.reshape(1, -1, 2).shape == (1, 10, 2)
        assert torch.isfinite(second).all()
