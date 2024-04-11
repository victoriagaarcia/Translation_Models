import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    model.train()
    total_loss = 0.0

    # for batch_idx, (inputs, targets) in enumerate(train_data):
    for inputs, targets in train_data:
        inputs = inputs.to(device) #.float()
        targets = targets.to(device) #.float()

        # Zero the gradients for this batch
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)

        # Denormalize the outputs and targets ¿?
        # outputs = outputs * std + mean
        # targets = targets * std + mean

        # Compute the loss (CrossEntropyLoss in training)
        loss_value = loss(outputs, targets)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item()

    avg_loss = total_loss / len(train_data)
    print(f"Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)



@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    model.eval()
    total_loss = 0.0

    for inputs, targets in val_data:
        inputs, targets = inputs.to(device).float(), targets.to(device).float()

        outputs = model(inputs)
        outputs = outputs * std + mean
        targets = targets * std + mean
        loss_value = loss(outputs, targets)

        total_loss += loss_value.item()

    avg_loss = total_loss / len(val_data)

    print(
            f"Epoch: {epoch + 1}, Val Loss: {avg_loss:.4f}"
        )


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        mean: mean of the target.
        std: std of the target.
        device: device for running operations.

    Returns:
        mae of the test data.
    """

    model.eval()
    maes_list = []

    for inputs, targets in test_data:
        inputs, targets = inputs.to(device).float(), targets.to(device).float()

        outputs = model(inputs)
        outputs = outputs * std + mean
        targets = targets * std + mean
        maes_list.append(torch.abs(outputs - targets).mean().item())

    mean_mae: float = float(np.mean(maes_list))
    return mean_mae


######################################################

import torch.optim as optim
import torch.nn as nn
import random as rn

# def train_step():
#     loss = nn.CrossEntropyLoss(ignore_index=eng_word2int[PAD_TOKEN])