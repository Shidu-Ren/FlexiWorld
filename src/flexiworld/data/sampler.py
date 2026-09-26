import os
import torch


class HomogeneousDistributedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Shuffle fixed-length global batches, then shard each batch across ranks."""

    def __init__(
        self,
        bucket_lengths: list[int],
        local_batch_size: int,
        world_size: int,
        rank: int,
        seed: int,
    ) -> None:
        if not bucket_lengths or any(length < 0 for length in bucket_lengths):
            raise ValueError("bucket_lengths must be non-empty and non-negative")
        if local_batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed batch sampler configuration")
        self.bucket_lengths = list(bucket_lengths)
        self.local_batch_size = int(local_batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

    @property
    def sampler(self):
        """Expose this batch sampler to Lightning's per-epoch sampler hook."""
        return self

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def global_batch_size(self) -> int:
        return self.local_batch_size * self.world_size

    def __len__(self) -> int:
        return sum(length // self.global_batch_size for length in self.bucket_lengths)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_batches = []
        offset = 0
        for length in self.bucket_lengths:
            permutation = torch.randperm(length, generator=generator)
            usable = length - length % self.global_batch_size
            if usable:
                batches = permutation[:usable].reshape(-1, self.global_batch_size) + offset
                global_batches.extend(batches.unbind(0))
            offset += length

        order = torch.randperm(len(global_batches), generator=generator).tolist()
        start = self.rank * self.local_batch_size
        stop = start + self.local_batch_size
        for index in order:
            yield global_batches[index][start:stop].tolist()


def distributed_context(cfg) -> tuple[int, int]:
    devices = cfg.trainer.devices
    world_size = int(devices if isinstance(devices, int) else len(devices))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank
