from __future__ import annotations
from bisect import bisect_right
import torch
from flexiworld.data.dataset import VariableKPixelDataset


class HomogeneousConcatDataset(torch.utils.data.Dataset):
    """Preserve vectorized Lance reads for homogeneous ConcatDataset batches."""

    def __init__(self, datasets: list[VariableKPixelDataset]) -> None:
        self.datasets = list(datasets)
        lengths = torch.tensor([len(dataset) for dataset in datasets], dtype=torch.long)
        self.cumulative = lengths.cumsum(0).tolist()

    def __len__(self) -> int:
        return self.cumulative[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        bucket = bisect_right(self.cumulative, index)
        offset = 0 if bucket == 0 else self.cumulative[bucket - 1]
        return bucket, index - offset

    def __getitem__(self, index: int):
        bucket, local = self._locate(int(index))
        return self.datasets[bucket][local]

    def __getitems__(self, indices: list[int]):
        located = [self._locate(int(index)) for index in indices]
        buckets = {bucket for bucket, _ in located}
        if len(buckets) != 1:
            raise ValueError("a mixed-frame batch must contain one frame length")
        bucket = located[0][0]
        return self.datasets[bucket].__getitems__([local for _, local in located])
