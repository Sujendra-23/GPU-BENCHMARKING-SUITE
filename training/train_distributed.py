"""Multi-GPU data-parallel training using PyTorch's DistributedDataParallel
with the NCCL backend for gradient synchronization across GPUs.

Requires a machine with 2+ NVIDIA GPUs and NCCL - not available in the
sandbox this repo was built in (single-process CPU-only), see README.

Launch (single node, 4 GPUs):

    torchrun --standalone --nproc_per_node=4 train_distributed.py \
        --epochs 5 --batch-size 64 --data-dir ./data

What this demonstrates for the "Deep Learning and HPC" bucket:
  - NCCL backend for cross-GPU gradient all-reduce
  - DistributedSampler so each GPU sees a distinct data shard per epoch
  - Gradient scaling for mixed-precision (fp16) training
  - Per-GPU throughput logging, the kind of data an HPC benchmarking report
    would aggregate across nodes
"""
from __future__ import annotations

import argparse
import os
import time


def setup_distributed():
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_rank(), dist.get_world_size()


def build_model_and_data(batch_size: int, data_dir: str):
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    # Fine-tune only the classifier head, matching the transfer-learning
    # setup this reuses from the existing image-classification project.
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # binary example task

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    return model, loader, sampler


def train(args):
    import torch
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    model, loader, sampler = build_model_and_data(args.batch_size, args.data_dir)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # reshuffle differently each epoch across ranks
        model.train()
        epoch_start = time.time()
        samples_seen = 0

        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # NCCL all-reduce happens inside DDP's backward hook
            scaler.step(optimizer)
            scaler.update()

            samples_seen += images.size(0)

        elapsed = time.time() - epoch_start
        throughput = samples_seen * world_size / elapsed
        if global_rank == 0:
            print(f"epoch {epoch}: {elapsed:.2f}s, "
                  f"~{throughput:.1f} samples/sec across {world_size} GPUs, "
                  f"loss={loss.item():.4f}")

    import torch.distributed as dist
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
