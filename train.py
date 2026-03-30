import datetime
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

INPUT_DIM = 32
NUM_CLASSES = 10
NUM_SAMPLES = 4096
BATCH_SIZE = 64
EPOCHS = 5000
LR = 1e-3


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def make_fake_data():
    X = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    return TensorDataset(X, y)


def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(minutes=30))
    torch.cuda.set_device(rank)

    dataset = make_fake_data()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    model = MLP().to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in loader:
            X, y = X.to(rank), y.to(rank)

            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += X.size(0)

        if rank == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/total:.4f}  acc={correct/total:.2%}")

    dist.destroy_process_group()
    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

    if os.environ.get("LOCAL_RANK") is not None:
        # Launched by torchrun — we're already inside a worker
        rank = int(os.environ["LOCAL_RANK"])
        train(rank, world_size)
    else:
        # Direct launch — use torch.multiprocessing.spawn for debuggability
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        if os.environ.get("DEBUG"):
            import debugpy
            debugpy.connect(("localhost", int(os.environ.get("DEBUG_PORT", 5678))))
            debugpy.wait_for_client()

        torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
