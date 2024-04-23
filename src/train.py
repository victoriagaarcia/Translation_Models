from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')

SOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def asMinutes(s: float) -> str:
    """
    Function to convert seconds to minutes and seconds.

    Args:
        s: seconds to convert.

    Returns:
        string with the minutes and seconds.
    """

    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since: float, percent: float) -> str:
    """
    Function to compute the time since the start of the training.

    Args:
        since: start time.
        percent: percentage of the training.

    Returns:
        string with the time since the start and the time remaining.
    """

    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def val_epoch(dataloader: DataLoader, encoder: torch.nn.Module, decoder: torch.nn.Module,
              criterion: torch.nn.Module, device: torch.device) -> float:

    """
    This function computes the validation for each epoch.

    Args:
        dataloader: DataLoader with the validation data.
        encoder: encoder model.
        decoder: decoder model.
        criterion: loss function.
        device: device to run the computation.

    Returns:
        average loss.
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


def train_epoch(dataloader: DataLoader, encoder: torch.nn.Module,
                decoder: torch.nn.Module, encoder_optimizer: torch.optim.Optimizer,
                decoder_optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                device: torch.device) -> float:
    """
    This function computes the training for each epoch.

    Args:
        dataloader: DataLoader with the training data.
        encoder: encoder model.
        decoder: decoder model.
        encoder_optimizer: encoder optimizer.
        decoder_optimizer: decoder optimizer.
        criterion: loss function.
        device: device to run the computation.

    Returns:
        average loss.
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
          encoder: torch.nn.Module, decoder: torch.nn.Module,
          n_epochs: int, learning_rate: float, device: torch.device,
          writer: SummaryWriter, print_every: int = 100) -> None:

    """
    This function trains the model.

    Args:
        train_dataloader: DataLoader with the training data.
        val_dataloader: DataLoader with the validation data.
        encoder: encoder model.
        decoder: decoder model.
        n_epochs: number of epochs.
        learning_rate: learning rate.
        device: device to run the computation.
        writer: tensorboard writer.
        print_every: print every n epochs.
    """

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
        writer.add_scalar("Loss/train", loss, epoch)

        # Print train loss
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))

        # Val loop
        loss_val = val_epoch(val_dataloader, encoder, decoder, criterion, device)
        print_loss_total_val += loss_val
        writer.add_scalar("Loss/val", loss_val, epoch)
        # Print val loss
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total_val / print_every
            print_loss_total_val = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))
