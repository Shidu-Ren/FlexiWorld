import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from stable_worldmodel.world import world as world_impl
from flexiworld.utils import RawZeroActionProcessor
from flexiworld.runtime import configure_deterministic_math
from flexiworld.protocol import select_eval_anchors, expert_history, sha256_file

EVAL_SEEDS = (0, 1, 42)
DISTANCES = (25, 50, 75, 100)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=cfg.dataset.keys_to_cache, cache_dir=dataset_path
    )
    return dataset


def evaluate_from_dataset_with_midpoint(
    world,
    *,
    dataset,
    episodes_idx,
    start_steps,
    goal_offset,
    eval_budget,
    midpoint_steps,
    callables=None,
    video=None,
    reset_mode="wait",
):
    """Evaluate once for ``eval_budget`` steps and snapshot success at midpoint."""
    n = len(episodes_idx)
    if n != world.num_envs:
        raise ValueError(f"expected {world.num_envs} episodes, got {n}")
    if not 0 < midpoint_steps < eval_budget:
        raise ValueError("midpoint_steps must lie strictly inside eval_budget")
    init_state, goal_state, dataset_videos = world_impl._extract_init_goal(
        dataset, episodes_idx, start_steps, goal_offset
    )
    world.reset(seed=init_state.get("seed"))
    if callables:
        merged = {**init_state, **goal_state}
        for i in range(n):
            env_init = {key: value[i] for key, value in merged.items()}
            world_impl._apply_callables(world.envs.envs[i].unwrapped, callables, env_init)
    shape_prefix = world.infos["pixels"].shape[:2]
    for state in (init_state, goal_state):
        for key, value in state.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    value[:, None, ...], shape_prefix + value.shape[1:]
                ).copy()
    goal_snapshot = {key: world.infos[key].copy() for key in goal_state}
    successes = np.zeros(n, dtype=bool)
    midpoint_successes = None
    step_count = 0
    frames = defaultdict(list) if video else None

    def on_step(active_world):
        nonlocal midpoint_successes, step_count
        step_count += 1
        active_world.infos.update(deepcopy(goal_snapshot))
        successes[:] |= active_world.terminateds
        if step_count == midpoint_steps:
            midpoint_successes = successes.copy()
        if frames is not None:
            for i in range(active_world.num_envs):
                frame = active_world.infos["pixels"][i]
                frame = frame[-1] if frame.ndim > 3 else frame
                frames[i].append(np.asarray(frame).copy())

    world._run(max_steps=eval_budget, mode=reset_mode, on_step=on_step)
    if midpoint_successes is None:
        if successes.all():
            midpoint_successes = successes.copy()
        else:
            raise RuntimeError(
                f"evaluation ended after {step_count} steps before midpoint {midpoint_steps}"
            )
    if frames:
        world_impl.save_panel_videos(
            Path(video), {"agent": frames, "dataset": dataset_videos, "goal": goal_state["goal"]}
        )
    return {
        "success_rate": float(successes.sum()) / n * 100.0,
        "episode_successes": successes,
        "first_segment_success_rate": float(midpoint_successes.sum()) / n * 100.0,
        "first_segment_episode_successes": midpoint_successes,
        "first_segment_steps": midpoint_steps,
        "seeds": init_state.get("seed"),
    }


class HistoryPolicy(swm.policy.WorldModelPolicy):
    def __init__(self, *, initial_history, **kwargs):
        super().__init__(**kwargs)
        self.initial_history = torch.as_tensor(initial_history, dtype=torch.float32)
        self.initialized = False

    def _prepare_info(self, info_dict):
        info = super()._prepare_info(info_dict)
        if not self.initialized:
            if len(info["pixels"]) != len(self.initial_history):
                raise ValueError("initial history and initial evaluation batch differ")
            info["_expert_previous_block"] = self.initial_history
            self.initialized = True
        return info


def evaluate(args):
    from flexiworld.planning.arcem import ActorPathCEMSolver
    from flexiworld.planning.direct import VarKDirectSolver

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    configure_deterministic_math(args.seed)
    cfg = OmegaConf.load(Path(__file__).resolve().parent / "configs" / "eval" / f"{args.task}.yaml")
    cfg.cache_dir = args.cache_dir
    cfg.eval.num_eval = 100
    cfg.eval.goal_offset_steps = args.distance
    cfg.eval.eval_budget = 2 * args.distance
    cfg.world.max_episode_steps = 4 * args.distance
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    episodes, starts = select_eval_anchors(dataset, args.distance, args.seed, 100)
    process = {}
    for col in cfg.dataset.keys_to_cache:
        values = dataset.get_col_data(col)
        values = values[np.isfinite(values).all(axis=1)]
        process[col] = (
            RawZeroActionProcessor.fit(values)
            if col == "action"
            else preprocessing.StandardScaler().fit(values)
        )
        if col != "action":
            process[f"goal_{col}"] = process[col]
    history = expert_history(dataset, episodes, starts, process["action"])
    directory = args.checkpoint.resolve()
    weights = directory / "weights.pt"
    model = hydra.utils.instantiate(OmegaConf.load(directory / "model.yaml"))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model = model.cuda().eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    common = dict(
        model=model,
        num_samples=128,
        n_steps=3,
        topk=16,
        batch_size=1,
        var_scale=1.0,
        replan_previous_mode="carry5",
        device="cuda",
        seed=args.seed,
    )
    if args.planner == "direct":
        solver = VarKDirectSolver(**common, schedule=[5] * (args.distance // 5))
        config = swm.PlanConfig(
            horizon=args.distance, receding_horizon=args.distance, action_block=1, warm_start=False
        )
    else:
        solver = ActorPathCEMSolver(**common, temperature=0.2, score_mode="arrival_min")
        config = swm.PlanConfig(
            horizon=args.distance // 5,
            receding_horizon=args.distance // 5,
            action_block=5,
            warm_start=False,
        )
    policy = HistoryPolicy(
        initial_history=history,
        solver=solver,
        config=config,
        process=process,
        transform={"pixels": img_transform(cfg), "goal": img_transform(cfg)},
    )
    world = swm.World(**cfg.world, image_shape=(224, 224))
    try:
        world.set_policy(policy)
        metrics = evaluate_from_dataset_with_midpoint(
            world,
            dataset=dataset,
            episodes_idx=episodes.tolist(),
            start_steps=starts.tolist(),
            goal_offset=args.distance,
            eval_budget=2 * args.distance,
            midpoint_steps=args.distance,
            callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
        )
        payload = dict(
            task=args.task,
            planner=args.planner,
            distance=args.distance,
            eval_seed=args.seed,
            training_seed=args.training_seed,
            checkpoint_sha256=sha256_file(weights),
            protocol="D+obs+D; k5; T0.2; expert5/carry5",
            eval_episodes=episodes,
            eval_start_steps=starts,
            metrics=metrics,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(_json_safe(payload), handle, indent=2)
        print(f"{args.task} {args.planner}: {metrics['success_rate']:.2f}%")
        return _json_safe(payload)
    finally:
        world.envs.close()


def summarize_results(rows, distances):
    """Report evaluation-seed variation for one checkpoint, not training-seed SD."""
    cells = {(r["distance"], r["eval_seed"]): r for r in rows}
    expected = {(d, s) for d in distances for s in EVAL_SEEDS}
    if len(cells) != len(rows) or set(cells) != expected:
        raise ValueError("incomplete or duplicate evaluation results")
    identity = ("task", "planner", "training_seed", "checkpoint_sha256", "protocol")
    for row in rows:
        if any(row[k] != rows[0][k] for k in identity):
            raise ValueError("results must come from one checkpoint and planner")
        successes = np.asarray(row["metrics"]["episode_successes"])
        if successes.shape != (100,) or not np.isin(successes, [0, 1]).all():
            raise ValueError("expected 100 binary episode outcomes per evaluation")
        if not np.isclose(successes.mean() * 100, row["metrics"]["success_rate"]):
            raise ValueError("success rate does not match episode outcomes")
    rates = np.array([
        [cells[d, s]["metrics"]["success_rate"] for s in EVAL_SEEDS] for d in distances
    ])

    def stats(values):
        return {
            "mean_success_rate": float(values.mean()),
            "eval_seed_sample_sd": float(values.std(ddof=1)),
            "per_eval_seed": dict(zip(map(str, EVAL_SEEDS), values.tolist())),
        }

    return {
        **{k: rows[0][k] for k in identity},
        "eval_seeds": list(EVAL_SEEDS),
        "distances": list(distances),
        "episodes_per_seed_and_distance": 100,
        "by_distance": {str(d): stats(rates[i]) for i, d in enumerate(distances)},
        "overall": stats(rates.mean(axis=0)),
    }


def run(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate one FlexiWorld checkpoint on seeds 0, 1, 42")
    parser.add_argument("--task", choices=["pusht", "cube", "reacher", "tworoom"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--planner", choices=["direct", "arcem"], required=True)
    parser.add_argument("--distance", nargs="+", type=int, choices=DISTANCES, default=list(DISTANCES))
    parser.add_argument("--cache-dir", default=os.environ.get("SWM_DATA_DIR", "./data_cache"))
    parser.add_argument("--output", type=Path, default=Path("results"), help="Results root directory")
    args = parser.parse_args(argv)
    if len(set(args.distance)) != len(args.distance):
        parser.error("distances must not contain duplicates")
    trained = OmegaConf.load(args.checkpoint / "train_config.yaml")
    if trained.task != args.task:
        raise ValueError("checkpoint task does not match --task")
    training_seed = int(trained.seed)
    for name in ("weights.pt", "model.yaml"):
        if not (args.checkpoint / name).is_file():
            raise FileNotFoundError(args.checkpoint / name)
    output = args.output / f"{args.task}_s{training_seed}_{args.planner}"
    jobs = [
        SimpleNamespace(**{
            **vars(args), "training_seed": training_seed, "distance": distance, "seed": seed,
            "output": output / f"d{distance}_e{seed}.json",
        })
        for distance in args.distance for seed in EVAL_SEEDS
    ]
    summary_path = output / "summary.json"
    for path in [summary_path, *(job.output for job in jobs)]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}; use a new --output directory")
    rows = [evaluate(job) for job in jobs]
    summary = summarize_results(rows, args.distance)
    output.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x") as handle:
        json.dump(summary, handle, indent=2)
    for distance, result in summary["by_distance"].items():
        print(f"D={distance}: {result['mean_success_rate']:.2f}% success "
              f"(evaluation-seed SD {result['eval_seed_sample_sd']:.2f})")
    print(f"Overall: {summary['overall']['mean_success_rate']:.2f}% success; {summary_path}")


if __name__ == "__main__":
    run()
