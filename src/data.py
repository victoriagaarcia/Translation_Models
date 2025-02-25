from __future__ import unicode_literals, print_function, division
from typing import List, Tuple
from io import open
import unicodedata
import re
import pandas as pd
import csv

import torch

from typing import Dict

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    """
    This class is the Language
    """

    def __init__(self, name: str) -> None:
        """
        Constructor of Lang.

        Args:
            name: name of the language
        """
        self.name = name
        self.word2index: Dict[str, int] = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
        self.word2count: Dict[str, int] = {}
        self.index2word: Dict[int, str] = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.n_words: int = 4  # Count SOS and EOS

    def addSentence(self, sentence: str) -> None:
        """
        Function to add a sentence to the language
        Args:
            sentence: sentence to add to the language
        """

        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word: str) -> None:
        """
        Function to add a word to the language
        Args:
            word: word to add to the language
        """

        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class TranslatorDataset(Dataset):
    """
    This class is the Translator dataset
    """
    def __init__(self, textlang_in: List[str], textlang_out: List[str],
                 lang_in: Lang, lang_out: Lang, end_token: int) -> None:
        """
        Constructor of the TranslatorDataset

        Args:
            textlang_in: input language texts
            textlang_out: output language texts
            lang_in: input language class
            lang_out: output language class
            end_token: last token for every sentence
        """

        self.textlang_in = textlang_in
        self.textlang_out = textlang_out
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.end_token = end_token

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            length of dataset.
        """

        return len(self.textlang_in)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with a text in the input language and
            the same text in the output language
        """

        return self.textlang_in[idx], self.textlang_out[idx]


def tensorFromSentence(
        lang: Lang,
        sentence: str,
        end_token: int,
        unk_token_str: str
        ) -> torch.Tensor:

    """
    Function to convert a sentence to a tensor

    Args:
        lang: language class
        sentence: sentence to convert to tensor
        end_token: last token for every sentence
        unk_token_str: unknown token string

    Returns:
        tensor with the indexes of the words in the sentence
    """

    indexes = [lang.word2index[word] if word in lang.word2index
               else lang.word2index[unk_token_str] for word in sentence.split(' ')]
    indexes.append(end_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def filterPairs(pairs: List[List[str]], max_length: int) -> List[List[str]]:
    """
    Function to filter pairs with a maximum length

    Args:
        pairs: list of pairs
        max_length: maximum length of the sentences
    """

    return [pair for pair in pairs if len(pair[0].split(' ')) < max_length and
            len(pair[1].split(' ')) < max_length]


# Turn a Unicode string to plain ASCII
def unicodeToAscii(s: str) -> str:
    """
    Function to convert a unicode string to ASCII

    Args:
        s: unicode string

    Returns:
        ASCII string
    """

    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s: str) -> str:
    """
    Function to normalize a string

    Args:
        s: string

    Returns:
        normalized string
    """

    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def collate_fn(
        batch: List[Tuple[str, str]],
        lang_in: Lang,
        lang_out: Lang,
        unk_token_str: str,
        end_token: int
        ) -> Tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """
    Function to collate the batch

    Args:
        batch: batch of texts
        lang_in: input language class
        lang_out: output language class
        unk_token_str: unknown token string
        end_token: last token for every sentence

    Returns:
        tuple with the input and output tensors
    """

    # Sort the batch by the length of text sequences in descending order
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    # Unzip textslang_in and textslang_out from the sorted batch
    lan1_batch, lan2_batch = zip(*batch)

    # Convert texts to indices
    lang1_indx: List[torch.Tensor] = [tensorFromSentence(lang_in, sentence,
                                      end_token, unk_token_str)
                                      for sentence in lan1_batch]
    lang2_indx: List[torch.Tensor] = [tensorFromSentence(lang_out, sentence,
                                      end_token, unk_token_str)
                                      for sentence in lan2_batch]

    # Calculate the lengths of each element of texts_indx.
    # The minimum length shall be 1, in order to avoid
    # later problems when training the RNN
    lang1_lengths: List[int] = [max(len(sentence), 1)
                                for sentence in lang1_indx]
    lang2_lengths: List[int] = [max(len(sentence), 1)
                                for sentence in lang2_indx]

    # Pad the text sequences to have uniform length
    lan1_padded: torch.Tensor = pad_sequence(lang1_indx, batch_first=True)
    lan2_padded: torch.Tensor = pad_sequence(lang2_indx, batch_first=True)

    return lan1_padded, lan2_padded, lang1_lengths, lang2_lengths


def readLangs(langname_in: str, langname_out: str
              ) -> Tuple[Lang, Lang, List[List[str]]]:
    """
    Function to read the languages

    Args:
        langname_in: input language name
        langname_out: output language name

    Returns:
        input language, output language and pairs of sentences
    """

    print("Reading lines...")

    data: pd.DataFrame = pd.read_csv('data/%s-%s.csv' % (langname_in, langname_out))

    normalized_data = data.map(lambda x: normalizeString(x)
                               if pd.notna(x) and x.strip() != '' else None)
    pairs: List[List[str]] = [list(row.dropna())
                              for _, row in normalized_data.iterrows()]
    pairs = [pair for pair in pairs
             if len(pair) == 2 and pair[0] != '' and pair[1] != '']

    input_lang = Lang(langname_in)
    output_lang = Lang(langname_out)

    return input_lang, output_lang, pairs


def prepareData(langname_in: str, langname_out: str, max_length: int
                ) -> Tuple[Lang, Lang, List[List[str]]]:
    """
    Function to prepare the data

    Args:
        langname_in: input language name
        langname_out: output language name
        max_length: maximum length of the sentences

    Returns:
        input language, output language and pairs of sentences
    """

    # Read data
    input_lang, output_lang, pairs = readLangs(langname_in, langname_out)

    # Filter pairs with max length
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))

    # Create vocabulary
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs


def get_dataloader(
    batch_size: int, unk_token_str: str, end_token: int,
    max_length: int, namelang_in: str, namelang_out: str
) -> Tuple[Lang, Lang, DataLoader, DataLoader]:
    """
    Function to get the dataloader

    Args:
        batch_size: batch size
        unk_token_str: unknown token string
        end_token: last token for every sentence
        max_length: maximum length of the sentences
        namelang_in: input language name
        namelang_out: output language name

    Returns:
        input language, output language and dataloaders
    """

    # Load and preprocess data
    input_lang, output_lang, pairs = prepareData(namelang_in, namelang_out, max_length)

    # Create inputs and outputs
    # divide the data into train and validation
    train_size = round(0.7 * len(pairs))
    val_size = round(0.2 * len(pairs))

    train_lang_in = [pair[0] for pair in pairs[:train_size]]
    train_lang_out = [pair[1] for pair in pairs[:train_size]]

    val_lang_in = [pair[0] for pair in pairs[train_size:train_size+val_size]]
    val_lang_out = [pair[1] for pair in pairs[train_size:train_size+val_size]]

    # Save data for test
    test_lang_in = [pair[0] for pair in pairs[train_size+val_size:]]
    test_lang_out = [pair[1] for pair in pairs[train_size+val_size:]]

    # Save data for test
    with open('data/%s-%s_test.csv' % (namelang_in, namelang_out),
              'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for pair in zip(test_lang_in, test_lang_out):
            csv_writer.writerow(pair)

    train_data: Dataset = TranslatorDataset(train_lang_in, train_lang_out,
                                            input_lang, output_lang, end_token)
    val_data: Dataset = TranslatorDataset(val_lang_in, val_lang_out,
                                          input_lang, output_lang, end_token)

    train_dataloader: DataLoader = DataLoader(train_data, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              drop_last=True,
                                              collate_fn=lambda batch:
                                              collate_fn(batch, input_lang, output_lang,
                                                         unk_token_str, end_token))

    val_dataloader: DataLoader = DataLoader(val_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            drop_last=True,
                                            collate_fn=lambda batch:
                                            collate_fn(batch, input_lang, output_lang,
                                                       unk_token_str, end_token))

    return input_lang, output_lang, train_dataloader, val_dataloader
