# Training and Evaluation Protocol

## Scope

FlexiWorld supports training and Direct/ARCEM evaluation on PushT, Cube, Reacher, and TwoRoom.

## Training Data

Use the benchmark trajectories from the [LeWM collection](https://huggingface.co/collections/quentinll/lewm). Following [LeWM's training setup](https://github.com/lucas-maes/le-wm/blob/main/train.py), dataset windows are randomly split into 90% training and 10% validation using the training seed. `split_manifest.json` records the split seed, window counts, and index hashes.

For goal span `S`, the split window covers `S + 5` trajectory rows, matching the fixed-five-action loader. An episode of length `L` contributes `max(L - S - 4, 0)` candidate starts. Each span initializes its split generator independently with the training seed.

PushT training uses a Lance dataset prepared from HDF5 with `src/prepare_data.py`. The converter stores images as JPEG (quality 95) and numeric columns as float32, and validates episode layout and action rows. Evaluation uses HDF5.

Training combines goal spans 35, 55, and 75 with 7, 11, and 15 action blocks respectively. Each block contains 1--10 primitive actions, and the partition sums to its requested span. The all-five partition is excluded. The model is trained for two epochs with Student Forcing probability 0.5. Configuration files are in `src/configs/train`.

Default training uses two GPUs on one node with DDP and batch size 128 per GPU (global batch 256). BatchNorm and SIGReg operate on local batches.

Within each split, windows are drawn with replacement: select an episode in proportion to its allowed start count, then select an allowed start uniformly. Batch indices seed the draws. Per-span epoch lengths are defined in `src/train.py` and `src/flexiworld/data/dataset.py` and recorded in `split_manifest.json`.

PushT uses 1,447,219 / 1,112,071 / 794,830 draws per epoch for spans 35 / 55 / 75. The homogeneous batch sampler drops each span's incomplete global batch, giving 13,101 updates per epoch. Validation uses a virtual length of 25,600 with one validation batch per epoch.

The [pretrained model package](https://huggingface.co/ryanren0330/FlexiWorld) contains one checkpoint per benchmark, all with training seed 3072. Each checkpoint includes task and training-seed metadata in `train_config.yaml`. New training runs save the full resolved configuration.

## Main Evaluation

- Each training command uses the requested seed; paper training seeds are 0, 42, 3072. Evaluation reads the training seed from the checkpoint and runs evaluation seeds 0, 1, 42 automatically.
- Four tasks; distances 25, 50, 75, 100; 100 episodes per cell.
- Same episode and start arrays for Direct and ARCEM at each task/seed/distance.
- First plan executes D primitive commands. Only failures reobserve and plan another D commands.
- Following [INTACT](https://github.com/zju3dv/INTACT-JEPA#evaluation), the actor receives the five dataset actions immediately preceding the sampled start `t` (`rows[t-5:t]`). Unavailable history at an episode boundary is left-padded with raw zeros before normalization. Subsequent history contains the controller's latest five executed commands (before environment clipping), without reading dataset actions at or after `t`.
- Direct uses conditional means. ARCEM uses `a = mean + 0.2 * residual`; 128 candidates, 3 iterations, top 16, arrival-min latent scoring.
- Evaluation uses benchmark trajectory start/goal states.

The evaluation entry point runs one task, one checkpoint, and one planner. It defaults to all four distances; `--distance 75` selects one distance. Each run saves per-episode outcomes and a `summary.json` with per-distance means and sample SD across evaluation seeds. For the overall result, first average distances within each evaluation seed, then compute the mean and sample SD of those three values.

The paper's reported results average distances and evaluation seeds within each checkpoint and compute sample SD across three training checkpoints.

## Tests

Run `pytest -q` for model/loss backward propagation, causal prefix properties, ARCEM zero-residual equivalence, unit residual scaling, data loading, action-history handling, evaluation entry points, and evaluation-seed aggregation checks.
