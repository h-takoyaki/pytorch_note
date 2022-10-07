import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

import _init_path
from utils.comm import *
from utils.logger import setup_logger
from utils.miscellaneous import init_seeds



class ToyNet(nn.Module):
    """
    ToyNet: 28*28, 512 -> 10
    """
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
        """
        Args:
            x (_type_): 28*28, 512
        Returns:
            logits: can be treated as loss
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def load_data(batch_size):
    """_summary_

    Args:
        batch_size (int):

    Returns:
        train_dataloader: ddp
        test_dataloader: dataloader
    """
    train_data = datasets.FashionMNIST(
        root="~/data/datasets/",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="~/data/datasets/",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if is_distributed() else None
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def train(dataloader, model, loss_fn, optimizer, train_logger):
    """

    Args:
        dataloader (_type_): ddp
        model (_type_): ToyNet
        loss_fn (_type_): cross entropy
        optimizer (_type_): _description_
    """

    device = get_rank()
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X) * num_gpus()
            train_logger.parent = None
            train_logger.info(f"GPU{device} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, test_logger):
    """
    Args:
        dataloader (_type_): _description_
        model (_type_): ToyNet
        loss_fn (_type_): cross entropy
    """
    device = get_rank()
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
    test_logger.parent = None
    test_logger.info(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


def main(args):
    """
    Args:
        args (parser.parse_args())
        args.epochs (int)
        args.batch_size (int)
    """
    is_ddp = is_distributed()
    ddp_setup()
    local_rank = get_rank()
    init_seeds(0+local_rank)
    model = ToyNet().to(local_rank)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            )
    train_dataloader, test_dataloader = load_data(args.batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    save_path = Path('/home/huangzhanghao/data/codes/experiments/logger/outputs')
    main_logger = setup_logger(f'logger.main.{local_rank}', True, save_path, local_rank)
    train_logger = setup_logger(f'logger.train.{local_rank}',
                                True,
                                save_path,
                                local_rank)
    test_logger = setup_logger(f'logger.test.{local_rank}',
                               True,
                               save_path,
                               0)

    main_logger.info(f'Start Training')
    synchronize()
    for t in range(args.epochs):
        main_logger.info(f"Epoch {t+1} / {args.epochs}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, train_logger)
        if is_main_process():
            test(test_dataloader, model, loss_fn, test_logger)
        synchronize()
    cleanup()
    main_logger.info('Done')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="multigpu_torchrun quick start")
    parser.add_argument('--epochs',
                        default=5, type=int,
                        help="Total epochs (default: 30)")
    parser.add_argument('--batch_size',
                        default=16, type=int,
                        help="Input batch size on each device (default: 32)")
    args = parser.parse_args()

    main(args)
