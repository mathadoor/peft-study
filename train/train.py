import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import sys

sys.path.append('..')
from utils.metrics import AverageMeter


def train_step(model: Module, optimizer: Optimizer, x_train: torch.Tensor, y_train: torch.Tensor, criterion) -> (
        torch.Tensor, torch.Tensor):
    """
    Perform a single training step
    :param model: PyTorch model
    :param optimizer: PyTorch optimizer
    :param x_train: Input data
    :param y_train: Target data
    :param criterion: Loss function
    :return: Tuple of loss and predictions
    """
    # Reset optimizer
    optimizer.zero_grad()

    # Forward pass
    logits = model(x_train)
    loss = criterion(logits, y_train)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Compute Predictions
    y_pred = torch.argmax(logits, dim=1)

    return loss, y_pred


def train_epoch(model: Module,
                optimizer: Optimizer,
                train_loader: DataLoader,
                criterion,
                loss_meter: AverageMeter,
                acc_meter: AverageMeter) -> (float, float):
    model.train()
    loss_meter.reset()
    acc_meter.reset()

    for x_train, y_train in train_loader:
        loss, y_pred = train_step(model, optimizer, x_train, y_train, criterion)
        loss_meter.update(loss.item(), len(y_train))
        acc_meter.update(torch.sum(y_train == y_pred).item(), len(y_train))

    return loss_meter.average(), acc_meter.average()
