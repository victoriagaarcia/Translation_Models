from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from data import tensorFromSentence

from data import get_dataloader, normalizeString
from utils import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

def evaluate(encoder, decoder, sentence, input_lang, output_lang, unk_token_str):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, EOS_token, unk_token_str)
        input_tensor = input_tensor.transpose(0, 1)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn  = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, input_lang, output_lang, unk_token_str):
    
    with open('data/evaluate.txt', 'r') as file:
        lines = file.readlines()
        
        for sentence in lines:
            sentence = normalizeString(sentence)
            output_words, _ = evaluate(encoder, decoder, sentence, input_lang, output_lang, unk_token_str)
            output_sentence = ' '.join(output_words)
            print('>', sentence)
            print('<', output_sentence)
            print('')

if __name__ == "__main__":
        
    # Hyperparameters
    batch_size: int = 1

    # Parameteres data
    SOS_token: int = 1
    EOS_token: int = 2
    unk_token: int = 3
    unk_token_str: str = "UNK"
    max_length: int = 10
    namelang_in: str = 'eng'
    namelang_out: str = 'fra'

    # Set device
    device: torch.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    input_lang, output_lang, train_dataloader, val_dataloader = get_dataloader(batch_size, unk_token_str, EOS_token, max_length, namelang_in, namelang_out)

    # load models
    encoder = load_model('models/best_encoder.pt')
    decoder = load_model('models/best_decoder.pt')

    evaluateRandomly(encoder, decoder, input_lang, output_lang, unk_token_str)
        