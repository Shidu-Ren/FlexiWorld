# Training

Run commands from the repository root after [installation and data preparation](../README.md#installation).

## Train a Model

```bash
python src/train.py task=pusht seed=0
```

Select `pusht`, `cube`, `reacher`, or `tworoom`. The training seed is configurable.
Training uses mixed goal spans 35/55/75, variable-length action chunks, Student
Forcing, and two epochs. One trained model supports both Direct and ARCEM.

Configuration: [`configs/train/flexiworld.yaml`](../src/configs/train/flexiworld.yaml).
Architecture: [`configs/train/model/flexiworld.yaml`](../src/configs/train/model/flexiworld.yaml).

The default uses DDP on two GPUs on one node, with batch size 128 per GPU
(global batch 256).
Launch the command once; Lightning starts the two training processes.
BatchNorm and SIGReg operate on each GPU's local batch of 128 samples.

Training randomly samples windows with replacement from the training split.
PushT runs 13,101 updates per epoch. See the
[protocol](PROTOCOL.md#training-data) for budget and sampling details.

## Outputs

`outputs/<task>/seed<seed>/` contains `weights.pt`, epoch checkpoints,
`model.yaml`, `train_config.yaml`, and `split_manifest.json`. Completed model
directories are not overwritten. Training logs and outputs are saved locally.

Use the resulting directory with the [evaluation workflow](EVALUATION.md).

## Implementation

| File | Purpose |
|:--|:--|
| [`train.py`](../src/train.py) | Training loop and checkpoint saving |
| [`train_step.py`](../src/train_step.py) | Joint world-model and actor objective |
| [`concat.py`](../src/flexiworld/data/concat.py) | Mixed-span dataset composition |
| [`sampler.py`](../src/flexiworld/data/sampler.py) | Homogeneous distributed batches |
| [`prepare_data.py`](../src/prepare_data.py) | PushT HDF5-to-Lance preparation |

Shared trajectory loaders live in [`flexiworld/data/`](../src/flexiworld/data/).
