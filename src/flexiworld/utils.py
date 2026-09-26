from __future__ import annotations
import numpy as np
from lightning.pytorch.callbacks import Callback


class RawZeroActionProcessor:
    """Z-score actions while mapping missing history to a raw zero command."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean_ = np.asarray(mean)
        self.scale_ = np.asarray(std)
        if not np.isfinite(self.mean_).all():
            raise ValueError("Action mean must be finite")
        if not np.isfinite(self.scale_).all() or np.any(self.scale_ <= 0):
            raise ValueError("Action standard deviation must be finite and positive")

    @classmethod
    def fit(cls, values: np.ndarray) -> "RawZeroActionProcessor":
        values = np.asarray(values)
        valid = values[~np.isnan(values).any(axis=1)]
        if valid.shape[0] < 2:
            raise ValueError("At least two finite actions are required")
        return cls(valid.mean(axis=0), valid.std(axis=0, ddof=1))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        values = np.where(np.isnan(values), 0.0, values)
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        return values * self.scale_ + self.mean_


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch + 1
        should_save = epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs
        if trainer.is_global_zero and should_save:
            self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained

        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f"weights_epoch_{epoch}.pt",
        )
