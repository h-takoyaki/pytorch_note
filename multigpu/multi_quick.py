"""
multigpu_torchrun
"""
import os # handle 
import torch
from torch import nn
import torch.distributed as dist
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import *
import random
import numpy as np

def ddp_setup():
    dist.init_process_group(backend='nccl')
    dist.barrier()

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def load_data(batch_size):
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def train(dataloader, model, loss_fn, optimizer):
    device = int(os.environ['LOCAL_RANK'])
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"GPU{device} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    device = int(os.environ['LOCAL_RANK'])
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main(epochs: int, batch_size: int):
    ddp_setup()
    # env_dict = {
    #     key: os.environ[key]
    #     for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "LOCAL_RANK")
    # }
    # print(env_dict)
    local_rank = int(os.environ['LOCAL_RANK'])
    init_seeds(0+local_rank)
    model = ToyNet().to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    train_dataloader, test_dataloader = load_data(batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    pbar = tqdm(range(epochs))
    for t in pbar:
        pbar.set_description(f"GPU{local_rank} epoch {t}/{epochs}")
        train(train_dataloader, model, loss_fn, optimizer)
    if int(os.environ['LOCAL_RANK']) == 0:
        test(test_dataloader, model, loss_fn)
    dist.barrier()
    dist.destroy_process_group()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="multigpu_torchrun quick start")
    parser.add_argument('--epochs', default=50, type=int, help="Total epochs (default: 30)")
    parser.add_argument('--batch_size', default=8, type=int, help="Input batch size on each device (default: 32)")
    args = parser.parse_args()

    main(args.epochs, args.batch_size)