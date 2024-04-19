import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.metrics import bleu_score
from sacrebleu import corpus_bleu
from tqdm.auto import tqdm

import time

# other libraries
from typing import Optional


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

@torch.enable_grad()
def train_step(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    train_data: DataLoader,
    # mean: float,
    # std: float,
    loss: torch.nn.Module, # nn.CrossEntropyLoss
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    batch_size: int,
    device: torch.device,
    lang1_word2int: dict, # Diccionario de palabras a índices en el vocabulario de la lengua 1
    lang2_int2word: dict, # Diccionario de palabras a índices en el vocabulario de la lengua 2
    start_token: str = '<SOS>',
    # pad_token: int = 0
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
    train_acc = 0.0
    # loss_function = loss(ignore_index=lang1_word2int[pad_token])
    # encoder_optimizer = optim.AdamW(encoder.parameters())
    # decoder_optimizer = optim.AdamW(decoder.parameters())

    encoder.train()
    decoder.train()
    
    for inputs, targets, length_lang1, length_lang2 in tqdm(train_data):
        
        inputs = inputs.squeeze(-1)
        targets = targets.squeeze(-1)

        inputs = inputs.to(device) #.float()
        targets = targets.to(device) #.float()

        # Zero the gradients for this batch
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Forward pass
        output_encoder, encoder_hidden, encoder_cell = encoder(inputs, length_lang1)
        context = encoder_hidden.transpose(0, 1).reshape(1, batch_size, -1)
        decoder_input = torch.full((batch_size, 1), lang1_word2int[start_token]).to(device)
        # print('shape decoder input en training (pre-concat)', decoder_input.shape) # [64, 1]
        # print("shape encoder hidden en training", encoder_hidden.shape) # [2, 64, 128]
        # print(decoder_input)
        # decoder_input = torch.concat((decoder_input, context), dim=1) # o encoder_hidden
        # print('shape decoder input en training (post-concat)', decoder_input.shape) # [64, 257]
        decoder_hidden = context # encoder_hidden.view(1, batch_size, -1)
        decoder_cell = encoder_cell.view(1, batch_size, -1)
        
        loss_value = 0
        predicted_sentences = []
        
        # print(decoder_input[:10])
        # print(decoder_hidden[:10])
        
        for i in range(targets.size(1)):
            # print(targets[:,i])
            # time.sleep(0.5)
            logits, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
            # print(logits[:10])
            loss_value += loss(logits, targets[:, i])
            decoder_input = targets[:, i].reshape(batch_size, 1) # Teacher forcing
            # decoder_input = torch.concat((decoder_input, context), dim=1)
            # print('decoder input shape', decoder_input.shape)

            # # Obtener las palabras predichas para este paso de tiempo
            # top_values, top_indices = logits.topk(1)
            # predicted_words = [lang2_int2word[index.item()] for index in top_indices.squeeze(1)]
            
            # # Agregar las palabras predichas a la lista de la frase correspondiente
            # for j in range(batch_size):
            #     if i == 0:
            #         # Si es el primer paso de tiempo, inicializa la lista para esta frase
            #         predicted_sentences.append([])
            #     predicted_sentences[j].append(predicted_words[j])
        
        # print(predicted_sentences)
        loss_value.backward()
        
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        predicted_sentences = np.array(predicted_sentences)
        train_loss += loss_value.item()
        # train_acc += bleu_score(predicted_sentences, targets)
        
    # Compute the average loss
    avg_train_loss = train_loss / len(targets)
    # avg_train_acc = train_acc / len(targets)

    print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}")
    writer.add_scalar("Loss/train", avg_train_loss, epoch)

    # print(f"Epoch: {epoch + 1}, Train Acc: {avg_train_acc:.4f}")
    # writer.add_scalar("Acc/train", avg_train_acc, epoch)

@torch.no_grad()
def val_step(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    val_data: DataLoader,
    # mean: float,
    # std: float,
    writer: SummaryWriter,
    epoch: int,
    batch_size: int,
    device: torch.device,
    lang1_word2int: dict, # Diccionario de palabras a índices en el vocabulario de la lengua 1
    lang2_word2int: dict, # Diccionario de palabras a índices en el vocabulario de la lengua 2
    start_token: str = '<SOS>',
    # pad_token: int = 0
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

    encoder.eval()
    decoder.eval()
    
    
    val_acc = 0

    for inputs, targets in val_data:
        
        inputs = inputs.squeeze(-1)
        targets = targets.squeeze(-1)

        inputs = inputs.to(device) #.float()
        targets = targets.to(device) #.float()

        # Forward pass
        _, encoder_hidden, encoder_cell = encoder(inputs)

        decoder_input = torch.full((batch_size,1), lang1_word2int[start_token], dtype=torch.long).to(device)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        
        accuracy = 0
        
        for i in range(targets.size(1)):
            logits, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
            accuracy += bleu_score(logits.unsqueeze(1), targets[:, i].unsqueeze(1))
            decoder_input = targets[:, i].reshape(batch_size, 1) # Teacher forcing     
        
        val_acc += accuracy
        
    # Compute the average loss
    avg_val_acc = val_acc / len(targets)

    print(f"Epoch: {epoch + 1}, Val Accuracy: {avg_val_acc:.4f}")
    writer.add_scalar("Accuracy/val", avg_val_acc, epoch)



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