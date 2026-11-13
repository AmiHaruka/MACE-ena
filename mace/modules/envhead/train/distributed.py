"""Single-host torchrun context, CPU loaders, and per-rank recovery state."""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from mace.tools.torch_geometric.dataloader import DataLoader


@dataclass
class DistributedContext:
    device: object = "cpu"
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0

    @classmethod
    def initialize(cls, device):
        size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(device)
        if device.type == "cuda":
            device = torch.device("cuda", local if size > 1 else device.index or 0)
            torch.cuda.set_device(device)
        if size > 1:
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        return cls(device, rank, size, local)

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def broadcast(self, value):
        if not self.enabled:
            return value
        objects = [value if self.is_main else None]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

    def capture_rng(self):
        state = dict(
            torch=torch.get_rng_state(),
            python=random.getstate(),
            numpy=np.random.get_state(),
        )
        if torch.device(self.device).type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(self.device)
        if not self.enabled:
            return [state]
        states = [None] * self.world_size
        dist.all_gather_object(states, state)
        return states

    def restore_rng(self, states):
        if len(states) != self.world_size:
            raise ValueError(
                "Exact training resume requires the checkpoint's world size"
            )
        state = states[self.rank]
        torch.set_rng_state(state["torch"].cpu())
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        if "cuda" in state and torch.device(self.device).type == "cuda":
            torch.cuda.set_rng_state(state["cuda"].cpu(), self.device)

    def close(self):
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_worker):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loaders(train, valid, args, context):
    if len(train) < context.world_size:
        raise ValueError("Training needs at least one structure per DDP rank")
    sampler = DistributedSampler(
        train,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=context.enabled,
    )
    validation = (
        Subset(valid, range(context.rank, len(valid), context.world_size))
        if context.enabled
        else valid
    )
    options = dict(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=_seed_worker,
    )
    if args.num_workers:
        options.update(
            multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2
        )
    # Separate generators keep loader/worker initialization out of model RNG.
    return (
        DataLoader(
            train,
            batch_size=args.batch_size,
            sampler=sampler,
            generator=torch.Generator().manual_seed(args.seed + context.rank),
            **options
        ),
        DataLoader(
            validation,
            batch_size=args.valid_batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(
                args.seed + context.rank + context.world_size
            ),
            **options
        ),
    )
