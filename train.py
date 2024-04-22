from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')

SOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def asMinutes(s: float) -> str:
    """
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since: float, percent: float) -> str:
    """
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def val_epoch(dataloader: DataLoader, encoder: torch.nn.Module, decoder: torch.nn.Module,
              encoder_optimizer: torch.optim.Optimizer,
              decoder_optimizer: torch.optim.Optimizer,
              criterion: torch.nn.Module, device: torch.device) -> float:

    """
    This function computes the validation for each epoch.

    Args:

    """

    # set model to eval mode
    encoder.eval()
    decoder.eval()

    # define metric lists
    total_loss = 0

    with torch.no_grad():
        # iterate over the validation data
        for data in dataloader:
            # Load data
            input_tensor, target_tensor, _, _ = data

            # Adjust dimensions
            # input_tensor dimensions: [Batch Size, Sequence Length, 1]
            input_tensor = input_tensor.squeeze(-1)
            # target_tensor dimensions: [Batch Size, Sequence Length, 1]
            target_tensor = target_tensor.squeeze(-1)

            # Compute encoder forward
            encoder_outputs, encoder_hidden = encoder(input_tensor)

            # Compute decoder forward
            decoder_outputs, _, _ = decoder(encoder_outputs,
                                            encoder_hidden, target_tensor)

            # Compute loss
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )

            total_loss += loss.item()

        return total_loss / len(dataloader)

    # write metrics
    # writer.add_scalar("Loss/val", np.mean(losses), epoch)
    # writer.add_scalar("Accuracy/val", acc.compute(), epoch)


def train_epoch(dataloader: DataLoader, encoder: torch.nn.Module,
                decoder: torch.nn.Module, encoder_optimizer: torch.optim.Optimizer,
                decoder_optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                device: torch.device) -> float:
    """
    This function computes the training for each epoch.

    Args:
    """

    total_loss = 0

    # models to train
    encoder.train()
    decoder.train()

    for data in tqdm(dataloader):
        # Load data
        input_tensor, target_tensor, _, _ = data

        # Adjust dimensions
        # input_tensor dimensions: [Batch Size, Sequence Length, 1]
        input_tensor = input_tensor.squeeze(-1)
        # target_tensor dimensions: [Batch Size, Sequence Length, 1]
        target_tensor = target_tensor.squeeze(-1)

        # zero the parameter gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Compute encoder forward
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # Compute decoder forward
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # Compute loss
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        # backward pass
        loss.backward()

        # Optimize
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Compute loss
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(train_dataloader: DataLoader, val_dataloader: DataLoader,
          encoder: torch.nn.Module, decoder: torch.nn.Module, n_epochs: int,
          learning_rate: float, device: torch.device, print_every: int = 100) -> None:

    start = time.time()

    # Reset every print_every
    print_loss_total: float = 0.0
    print_loss_total_val: float = 0.0

    # Define optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # Define Loss Function
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, n_epochs + 1)):
        # Train loop
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer,
                           decoder_optimizer, criterion, device)
        print_loss_total += loss

        # Print train loss
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))

        # Val loop
        loss = val_epoch(val_dataloader, encoder, decoder, encoder_optimizer,
                         decoder_optimizer, criterion, device)
        print_loss_total_val += loss

        # Print val loss
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total_val / print_every
            print_loss_total_val = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))
