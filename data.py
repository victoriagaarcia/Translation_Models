
from __future__ import unicode_literals, print_function, division
import unicodedata
import re

import torch

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 1
EOS_token = 2

MAX_LENGTH = 15

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        self.word2count = {}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class DatasetTranslator(Dataset):

    def __init__(self, text_lang1, text_lang2):
        self.text_lang1 = text_lang1
        self.text_lang2 = text_lang2
    
    def __len__(self):
        return len(self.text_lang1)
    
    def __getitem__(self, idx):
        return self.text_lang1[idx], self.text_lang2[idx]
# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, filename):
    print("Reading lines...")
    # Read the file into a DataFrame
    df = pd.read_csv(filename)

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    text_lang1 = df[lang1]
    text_lang2 = df[lang2]
    
    # divide the data into train, validation and test
    train_size = round(0.8 * len(text_lang1))
    val_size = round(0.2 * len(text_lang1))
    
    tr_texts = text_lang1[:train_size]
    val_texts = text_lang1[train_size:train_size+val_size]

    tr_texts2 = text_lang2[:train_size]
    val_texts2 = text_lang2[train_size:train_size+val_size]   
    
    tokens_lang1_tr = [normalizeString(str(s)).split() for s in tr_texts]
    tokens_lang2_tr = [normalizeString(str(s)).split() for s in tr_texts2]
    tokens_lang1_val = [normalizeString(str(s)).split() for s in val_texts]
    tokens_lang2_val = [normalizeString(str(s)).split() for s in val_texts2]
    
    return input_lang, output_lang, tokens_lang1_tr, tokens_lang2_tr, tokens_lang1_val, tokens_lang2_val

def filterPair(p):
    return len(p[0]) < MAX_LENGTH and \
        len(p[1]) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse = False):

    filename = f'data/{lang1.lower()}_{lang2.lower()}.csv'
    
    input_lang, output_lang, text_tr, text_tr2, text_val, text_val2 = readLangs(lang1, lang2, filename)

    pairs = [[text_tr[i], text_tr2[i]] for i in range(len(text_tr))]
    pairs_val = [[text_val[i], text_val2[i]] for i in range(len(text_val))]
    
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    pairs_val = filterPairs(pairs_val)
    
    print("Trimmed to %s sentence pairs" % len(pairs))
    
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, pairs_val


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes= [SOS_token]
    indexes += indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def collate_fn(batch, input_lang, output_lang):

    # Sort the batch by the length of text sequences in descending order
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    
    # Unzip texts and labels from the sorted batch
    lan1_batch, lan2_batch = zip(*batch)
    
    # Convert texts to indices using the word2idx function and w2v_model
    lang1_indx = [tensorFromSentence(input_lang, sentence) for sentence in lan1_batch]
    lang2_indx = [tensorFromSentence(output_lang, sentence) for sentence in lan2_batch]
    
    # Calculate the lengths of each element of texts_indx.
    # The minimum length shall be 1, in order to avoid later problems when training the RNN
    lang1_lengths = [max(len(sentence), 1) for sentence in lang1_indx]
    lang2_lengths = [max(len(sentence), 1) for sentence in lang2_indx]
    
    # Pad the text sequences to have uniform length
    lan1_padded = pad_sequence(lang1_indx, batch_first=True)
    lan2_padded = pad_sequence(lang2_indx, batch_first=True)
    
    return lan1_padded, lan2_padded

def get_dataloader(batch_size, input_lang, output_lang):

    input_lang, output_lang, pairs, pairs_val = prepareData(input_lang, output_lang)
    
    tr_texts1 = [pair[0] for pair in pairs]
    tr_texts2 = [pair[1] for pair in pairs]

    val_texts1 = [pair[0] for pair in pairs_val]
    val_texts2 = [pair[1] for pair in pairs_val]

    train_data = DatasetTranslator(tr_texts1, tr_texts2)
    val_data = DatasetTranslator(val_texts1, val_texts2)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True,
                                  collate_fn=lambda batch: collate_fn(batch, input_lang, output_lang))
    
    val_dataloader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0,
                                drop_last=True,
                                collate_fn=lambda batch: collate_fn(batch, input_lang, output_lang))

    return train_dataloader, val_dataloader, input_lang, output_lang
