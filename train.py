from __future__ import unicode_literals, print_function, division

from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from evaluate import evaluateRandomly

import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

SOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def val_epoch(dataloader, encoder, decoder, encoder_optimizer, 
              decoder_optimizer, criterion, device):
    
    encoder.eval()
    decoder.eval()

    # define metric lists
    total_loss = 0

    with torch.no_grad():
        # iterate over the validation data
        for data in dataloader:
            
            input_tensor, target_tensor, length_input, length_output = data
            
            input_tensor = input_tensor.squeeze(-1)
            target_tensor = target_tensor.squeeze(-1)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )

            total_loss += loss.item()
        
        return total_loss / len(dataloader)

    # # write metrics
    # writer.add_scalar("Loss/val", np.mean(losses), epoch)
    # writer.add_scalar("Accuracy/val", acc.compute(), epoch)
    # evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang)

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, device):

    total_loss = 0

    encoder.train()
    decoder.train()

    for data in tqdm(dataloader):
        input_tensor, target_tensor, length_input, length_output = data
        
        input_tensor = input_tensor.squeeze(-1)
        target_tensor = target_tensor.squeeze(-1)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, val_dataloader, encoder, decoder, n_epochs, learning_rate, device,
               print_every=100, plot_every=100):
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    plot_losses_val = []
    print_loss_total_val = 0  # Reset every print_every
    plot_loss_total_val = 0  # Reset every plot_every


    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        loss = val_epoch(val_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device)
        print_loss_total_val += loss
        plot_loss_total_val += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total_val / print_every
            print_loss_total_val = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg_val = plot_loss_total_val / plot_every
            plot_losses_val.append(plot_loss_avg_val)
            plot_loss_total_val = 0
        
    showPlot(plot_losses)