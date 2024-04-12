import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional


@torch.enable_grad()
# def train_step(
#     model: torch.nn.Module,
#     train_data: DataLoader,
#     mean: float,
#     std: float,
#     loss: torch.nn.Module, # nn.CrossEntropyLoss
#     optimizer: torch.optim.Optimizer,
#     writer: SummaryWriter,
#     epoch: int,
#     device: torch.device,
# ) -> None:
#     """
#     This function train the model.

#     Args:
#         model: model to train.
#         train_data: dataloader of train data.
#         mean: mean of the target.
#         std: std of the target.
#         loss: loss function.
#         optimizer: optimizer.
#         writer: writer for tensorboard.
#         epoch: epoch of the training.
#         device: device for running operations.
#     """

#     model.train()
#     train_loss = 0.0

#     # for batch_idx, (inputs, targets) in enumerate(train_data):
#     for inputs, targets in train_data:
#         inputs = inputs.to(device) #.float()
#         targets = targets.to(device) #.float()

#         # Zero the gradients for this batch
#         optimizer.zero_grad()
#         # Forward pass
#         outputs = model(inputs)

#         # Denormalize the outputs and targets ¿?
#         # outputs = outputs * std + mean
#         # targets = targets * std + mean

#         # Compute the loss (CrossEntropyLoss in training)
#         loss_value = loss(outputs, targets)
#         loss_value.backward()
#         optimizer.step()

#         train_loss += loss_value.item()

#     # Compute the average loss
#     avg_train_loss = train_loss / len(train_data)

#     print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}")
#     writer.add_scalar("Loss/train", avg_train_loss, epoch)

def train_step(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    train_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module, # nn.CrossEntropyLoss
    writer: SummaryWriter,
    epochs: int,
    batch_size: int,
    device: torch.device,
    lang1_word2int: dict, # Diccionario de palabras a índices en el vocabulario de la lengua 1
    lang2_word2int: dict, # Diccionario de palabras a índices en el vocabulario de la lengua 2
    start_token: str = '<SOS>',
    pad_token: int = 0
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

    train_loss = 0.0

    loss_function = loss(ignore_index=lang1_word2int[pad_token])
    encoder_optimizer = optim.AdamW(encoder.parameters())
    decoder_optimizer = optim.AdamW(decoder.parameters())

    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        for inputs, targets in train_data:
            inputs = inputs.to(device) #.float()
            targets = targets.to(device) #.float()

            # Zero the gradients for this batch
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Forward pass
            _, encoder_hidden, encoder_cell = encoder(inputs)

            decoder_input = torch.full((batch_size,1), lang1_word2int[start_token], dtype=torch.long).to(device)
            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell

            for i in range(rn.randint(0, len(targets)-1)):
                logits, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
                train_loss += loss_function(logits, targets[:, i])
                decoder_input = targets[:, i].reshape(batch_size, 1) # Teacher forcing

            
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        # Compute the average loss
        avg_train_loss = train_loss / len(targets)

        print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}")


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module, # BLEU score
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
    val_loss = 0.0

    for inputs, targets in val_data:
        inputs = inputs.to(device) #.float()
        targets = targets.to(device) #.float()

        # Forward pass
        outputs = model(inputs)

        # Denormalize the outputs and targets ¿?
        # outputs = outputs * std + mean
        # targets = targets * std + mean

        # Compute the loss (BLEU score in validation)
        loss_value = loss(outputs, targets)
        val_loss += loss_value.item()

    # Update the scheduler
    if scheduler is not None:
        scheduler.step()

    # Compute the average loss
    avg_val_loss = val_loss / len(val_data)

    print(f"Epoch: {epoch + 1}, Val Loss: {avg_val_loss:.4f}")
    writer.add_scalar("Loss/val", avg_val_loss, epoch)


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
    loss: torch.nn.Module, # BLEU score
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
    bleu_scores = []

    for inputs, targets in test_data:
        inputs = inputs.to(device) #.float()
        targets = targets.to(device) #.float()

        # Forward pass
        outputs = model(inputs)

        # Denormalize the outputs and targets ¿?
        # outputs = outputs * std + mean
        # targets = targets * std + mean

        # Compute the BLEU score
        bleu_score = loss(outputs, targets)
        bleu_scores.append(bleu_score.item())

    avg_bleu: float = float(np.mean(bleu_scores))
    return avg_bleu


######################################################

import torch.optim as optim
import torch.nn as nn
import random as rn

# def train_step():
#     loss = nn.CrossEntropyLoss(ignore_index=eng_word2int[PAD_TOKEN])