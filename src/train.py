"""Train the full FlexiWorld model on mixed 35/55/75-step goal spans."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
from functools import partial
from pathlib import Path
import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from flexiworld.data.concat import HomogeneousConcatDataset
from flexiworld.data.dataset import VariableKPixelDataset
from flexiworld.models.layers import SIGReg
from train_step import forward
from flexiworld.data.sampler import HomogeneousDistributedBatchSampler, distributed_context
from flexiworld.runtime import configure_deterministic_math

TASKS = {
    "pusht": ("pusht_expert_train.lance", 2),
    "cube": ("ogbench/cube_single_expert.h5", 5),
    "reacher": ("dmc/reacher_random.h5", 2),
    "tworoom": ("tworoom.h5", 2),
}
PUSHT_TRAIN_WINDOWS = {35: 1_447_219, 55: 1_112_071, 75: 794_830}
VAL_WINDOWS = 25_600


class SaveModel(pl.Callback):
    def __init__(self, directory, cfg):
        self.directory, self.cfg = Path(directory), cfg

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            state = pl_module.model.state_dict()
            torch.save(state, self.directory / f"weights_epoch_{trainer.current_epoch + 1}.pt")
            torch.save(state, self.directory / "weights.pt")
            OmegaConf.save(self.cfg.model, self.directory / "model.yaml", resolve=True)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "configs" / "train"),
    config_name="flexiworld",
)
def run(cfg):
    if cfg.task not in TASKS:
        raise ValueError(f"task must be one of {tuple(TASKS)}")
    cfg.dataset_name, cfg.action_dim = TASKS[cfg.task]
    configure_deterministic_math(cfg.seed)
    world_size, rank = distributed_context(cfg)
    if cfg.loader.batch_size * world_size != 256:
        raise ValueError("the full-method global batch size must be 256")
    directory = Path(cfg.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "weights.pt").exists():
        raise FileExistsError(f"refusing to overwrite an existing model: {directory}")
    sets = [
        VariableKPixelDataset(
            split="train",
            seed=cfg.seed,
            length=PUSHT_TRAIN_WINDOWS[span] if cfg.task == "pusht" else None,
            cache_dir=cfg.cache_dir,
            dataset_name=cfg.dataset_name,
            action_dim=cfg.action_dim,
            n_blocks=span // 5,
            fixed_total_span=span,
            exclude_uniform_schedule=True,
        )
        for span in (35, 55, 75)
    ]
    validation = VariableKPixelDataset(
        split="val",
        seed=cfg.seed,
        length=VAL_WINDOWS,
        cache_dir=cfg.cache_dir,
        dataset_name=cfg.dataset_name,
        action_dim=cfg.action_dim,
        n_blocks=7,
        fixed_total_span=35,
        exclude_uniform_schedule=True,
    )
    sampler = HomogeneousDistributedBatchSampler(
        [len(x) for x in sets], cfg.loader.batch_size, world_size, rank, cfg.seed
    )
    loader_kwargs = dict(
        num_workers=cfg.loader.num_workers,
        pin_memory=True,
        persistent_workers=cfg.loader.num_workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(
        HomogeneousConcatDataset(sets), batch_sampler=sampler, **loader_kwargs
    )
    val_sampler = torch.utils.data.DistributedSampler(
        validation, num_replicas=world_size, rank=rank, shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        validation, batch_size=cfg.loader.batch_size, sampler=val_sampler, **loader_kwargs
    )
    model = hydra.utils.instantiate(cfg.model)
    module = spt.Module(
        model=model,
        sigreg=SIGReg(knots=17, num_proj=1024),
        forward=partial(forward, cfg=cfg),
        optim={
            "model_opt": {
                "modules": "model",
                "optimizer": dict(cfg.optimizer),
                "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                "interval": "epoch",
            }
        },
    )
    if rank == 0:
        OmegaConf.save(cfg, directory / "train_config.yaml", resolve=True)
        (directory / "split_manifest.json").write_text(
            json.dumps([x.split_metadata for x in sets] + [validation.split_metadata], indent=2)
            + "\n"
        )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[SaveModel(directory, cfg)],
        logger=CSVLogger(save_dir=directory, name="metrics"),
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        use_distributed_sampler=False,
    )
    trainer.fit(module, datamodule=spt.data.DataModule(train=train_loader, val=val_loader))


if __name__ == "__main__":
    run()
