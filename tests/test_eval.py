import copy
import json

import pytest

import eval as evaluation


def result(distance, seed, rate=80, training_seed=3072, task="pusht", planner="direct"):
    return dict(
        task=task, planner=planner, distance=distance, eval_seed=seed,
        training_seed=training_seed, checkpoint_sha256="test", protocol="test",
        metrics={"success_rate": rate, "episode_successes": [1] * rate + [0] * (100 - rate)},
    )


def test_summary_averages_distances_before_eval_seed_sd():
    rows = [result(d, s, rate) for d, rates in ((25, (70, 80, 90)), (50, (90, 80, 70)))
            for s, rate in zip(evaluation.EVAL_SEEDS, rates)]
    summary = evaluation.summarize_results(rows, [25, 50])
    assert summary["by_distance"]["25"]["eval_seed_sample_sd"] == 10
    assert summary["overall"]["mean_success_rate"] == 80
    assert summary["overall"]["eval_seed_sample_sd"] == 0
    assert "training_seed_sample_sd" not in summary["overall"]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "checkpoint", "rate", "outcomes"])
def test_summary_rejects_invalid_results(bad):
    rows = [result(25, s) for s in evaluation.EVAL_SEEDS]
    if bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif bad == "checkpoint":
        rows[0]["checkpoint_sha256"] = "other"
    elif bad == "rate":
        rows[0]["metrics"]["success_rate"] = 99
    else:
        rows[0]["metrics"]["episode_successes"] = [2] * 100
    with pytest.raises(ValueError):
        evaluation.summarize_results(rows, [25])


def checkpoint(tmp_path, task="pusht", seed=3072):
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "train_config.yaml").write_text(f"task: {task}\nseed: {seed}\n")
    (path / "model.yaml").write_text("{}")
    (path / "weights.pt").touch()
    return path


@pytest.mark.parametrize("task", ["pusht", "cube", "reacher", "tworoom"])
@pytest.mark.parametrize("planner", ["direct", "arcem"])
def test_cli_runs_three_seeds_and_reads_training_seed(tmp_path, monkeypatch, task, planner):
    path = checkpoint(tmp_path, task=task, seed=17)
    calls = []

    def fake_evaluate(args):
        calls.append(args)
        return result(args.distance, args.seed, training_seed=args.training_seed,
                      task=args.task, planner=args.planner)

    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    argv = ["--task", task, "--checkpoint", str(path), "--planner", planner,
            "--output", str(tmp_path / "results")]
    evaluation.run(argv)
    assert [(a.distance, a.seed) for a in calls] == [
        (d, s) for d in evaluation.DISTANCES for s in evaluation.EVAL_SEEDS
    ]
    assert {a.training_seed for a in calls} == {17}
    summary = json.loads((tmp_path / "results" / f"{task}_s17_{planner}" / "summary.json").read_text())
    assert summary["training_seed"] == 17
    assert summary["eval_seeds"] == [0, 1, 42]
    with pytest.raises(FileExistsError):
        evaluation.run(argv)
    assert len(calls) == 12


def test_single_distance_and_task_mismatch(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    calls = []

    def fake_evaluate(args):
        calls.append(args)
        return result(args.distance, args.seed)

    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    argv = ["--task", "pusht", "--checkpoint", str(path), "--planner", "direct",
            "--distance", "75", "--output", str(tmp_path / "results")]
    evaluation.run(argv)
    assert [(a.distance, a.seed) for a in calls] == [(75, s) for s in evaluation.EVAL_SEEDS]
    argv[1] = "cube"
    with pytest.raises(ValueError, match="checkpoint task"):
        evaluation.run(argv)


def test_failed_evaluation_does_not_write_summary(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)

    def fail(args):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(evaluation, "evaluate", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluation.run(["--task", "pusht", "--checkpoint", str(path), "--planner", "direct",
                        "--output", str(tmp_path / "results")])
    assert not list(tmp_path.rglob("summary.json"))
