
from __future__ import unicode_literals, print_function, division
import re
from typing import Tuple, List, Dict, Any

import torch

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
from utils import normalizeString
import torchtext


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 1
EOS_token = 2

def load_data(file_path: str) -> Tuple[List[List[str]], List[int]]:
    """
    Load data from a specified file path, extract texts and targets, and tokenize the texts using the tokenize_tweet function.

    Parameters:
    file_path (str): The path to the dataset file.

    Returns:
    Tuple[List[str], List[int]]: Lists of texts and corresponding targets.
    """

    # TODO: Read the corresponding csv
    print('Reading lines...')
    data: pd.DataFrame = pd.read_csv(file_path)
    
    # TODO: Obtain text columns from data
    text_english= data['English']
    text_spanish = data['Spanish']
    
    # TODO: Return tokenized texts
    return [normalizeString(str(s)).split() for s in text_english], [normalizeString(str(s)).split() for s in text_spanish] 
      
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.word2count = {}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.n_words = 4

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
    
    def __getitem__(self, idx,):
        return self.text_lang1[idx], self.text_lang2[idx],

    
# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427

def readLangs(lang1, lang2, filename):
    
    text_tokenize_english, text_tokenize_spanish = load_data(filename)

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    # divide the data into train, validation and test
    train_size = round(0.8 * len(text_tokenize_english))
    val_size = round(0.2 * len(text_tokenize_english))
    
    tr_texts = text_tokenize_english[:train_size]
    val_texts = text_tokenize_english[train_size:train_size+val_size]

    tr_texts2 = text_tokenize_spanish[:train_size]
    val_texts2 = text_tokenize_spanish[train_size:train_size+val_size]   
    
    # tokenizer_lang1 = torchtext.data.get_tokenizer('basic_english', 'en')
    # tokenizer_lang2 = torchtext.data.get_tokenizer('moses', 'es')
    
    # tokens_lang1_tr = [tokenizer_lang1(str(s).lower()) for s in tr_texts]
    # tokens_lang2_tr = [tokenizer_lang2(str(s).lower()) for s in tr_texts2]
    # tokens_lang1_val = [tokenizer_lang1(str(s).lower()) for s in val_texts]
    # tokens_lang2_val = [tokenizer_lang2(str(s).lower()) for s in val_texts2]   
    
    # # Use function to normalize the strings
    # tokens_lang1_tr = [normalizeString(str(s)).split() for s in tr_texts]
    # tokens_lang2_tr = [normalizeString(str(s)).split() for s in tr_texts2]
    # tokens_lang1_val = [normalizeString(str(s)).split() for s in val_texts]
    # tokens_lang2_val = [normalizeString(str(s)).split() for s in val_texts2]
    
    return input_lang, output_lang, tr_texts, tr_texts2, val_texts, val_texts2

def filterPair(p, max_length):
    return len(p[0]) <= max_length and \
        len(p[1]) <= max_length


def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def prepareData(lang1, lang2, max_length):

    filename = f'data/{lang1.lower()}_{lang2.lower()}.csv'
    
    input_lang, output_lang, text_tr, text_tr2, text_val, text_val2 = readLangs(lang1, lang2, filename)

    pairs = [[text_tr[i], text_tr2[i]] for i in range(len(text_tr))]
    pairs_val = [[text_val[i], text_val2[i]] for i in range(len(text_val))]
    
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    pairs_val = filterPairs(pairs_val, max_length)
    
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
    
    return lan1_padded, lan2_padded, lang1_lengths, lang2_lengths

def get_dataloader(batch_size, input_lang, output_lang, max_length):

    input_lang, output_lang, pairs, pairs_val = prepareData(input_lang, output_lang, max_length)
    
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