# Evaluation

Run commands from the repository root after [installation and data preparation](../README.md#installation).

## Evaluate a Checkpoint

```bash
python src/eval.py --task pusht --checkpoint checkpoints/pusht/seed3072 \
  --planner direct --cache-dir ./data_cache

python src/eval.py --task pusht --checkpoint checkpoints/pusht/seed3072 \
  --planner arcem --cache-dir ./data_cache
```

Select `pusht`, `cube`, `reacher`, or `tworoom`, with a matching checkpoint.
The same weights support both planners. A checkpoint trained locally can be
passed as `--checkpoint outputs/pusht/seed0`.

Each command automatically runs evaluation seeds **0, 1, 42**, at distances
**25, 50, 75, 100**, with 100 episodes per seed and distance. To run one distance,
add `--distance 75`. The training seed is read from `train_config.yaml`.

Both planners use `k=5`. ARCEM uses residual scale 0.2, 128 candidates,
3 search iterations, and 16 elites. Only first-stage failures are replanned.
Task configurations live in [`configs/eval/`](../src/configs/eval/).

## Results

```text
results/<task>_s<training-seed>_<planner>/
  d25_e0.json
  ...
  summary.json
```

Individual files record episode outcomes and checkpoint hashes. `summary.json`
contains success rates and sample SD across the three evaluation seeds for this
checkpoint. Overall statistics first average distances within each evaluation seed.
The paper's main tables report SD across training seeds.

Existing results are not overwritten. Use `--output results/another-run` to
evaluate again. See the [protocol](../docs/PROTOCOL.md) for execution and history details.

## Implementation

| File | Purpose |
|:--|:--|
| [`eval.py`](../src/eval.py) | Checkpoint loading, environment evaluation, and summary |
| [`protocol.py`](../src/flexiworld/protocol.py) | Evaluation anchors, expert history, and hashes |

The planners live in [`flexiworld/planning/`](../src/flexiworld/planning/):
[`direct.py`](../src/flexiworld/planning/direct.py) and
[`arcem.py`](../src/flexiworld/planning/arcem.py).
