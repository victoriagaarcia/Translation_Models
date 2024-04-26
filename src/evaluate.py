from __future__ import unicode_literals, print_function, division

import torch
import pandas as pd
from typing import List

import numpy as np
from data import normalizeString, filterPairs
from utils import load_model, load_vocab
from torch.jit import RecursiveScriptModule

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu


def tensorFromSentence(word2index: dict[str, int],
                       sentence: str,
                       end_token: int,
                       unk_token_str: str) -> torch.Tensor:
    """
    Convert a sentence to a tensor of indexes

    Args:
        word2index (dict): dictionary with the vocabulary
        sentence (str): sentence to convert
        end_token (int): end token
        unk_token_str (str): unknown token

    Returns:
        torch.Tensor: tensor with indexes
    """

    indexes = [word2index[word]
               if word in word2index else word2index[unk_token_str]
               for word in sentence.split(' ')]
    indexes.append(end_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def evaluate_step(encoder: RecursiveScriptModule, decoder: RecursiveScriptModule,
                  sentence: str, word2index_lang1: dict[str, int],
                  index2word_lang2: dict[int, str], unk_token_str: str):
    """
    Evaluate a sentence

    Args:
        encoder (RecursiveScriptModule): encoder model
        decoder (RecursiveScriptModule): decoder model
        sentence (str): sentence to evaluate
        word2index_lang1 (dict): dictionary with the vocabulary of the input language
        index2word_lang2 (dict): dictionary with the vocabulary of the output language
        unk_token_str (str): unknown token

    Returns:
        list: list with the words of the translated sentence
        torch.Tensor: tensor with the attention weights
    """

    with torch.no_grad():
        input_tensor = tensorFromSentence(word2index_lang1, sentence,
                                          EOS_token, unk_token_str)
        input_tensor = input_tensor.transpose(0, 1)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs,
                                                                encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(index2word_lang2[idx.item()])
    return decoded_words, decoder_attn


def evaluate(encoder: RecursiveScriptModule, decoder: RecursiveScriptModule,
             input_sentences: list, targets: list, word2index_lang1: dict[str, int],
             index2word_lang2: dict[int, str], unk_token_str: str):
    """
    Evaluate a list of sentences

    Args:
        encoder (RecursiveScriptModule): encoder model
        decoder (RecursiveScriptModule): decoder model
        input_sentences (list): list with the sentences to evaluate
        targets (list): list with the target sentences
        word2index_lang1 (dict): dictionary with the vocabulary of the input language
        index2word_lang2 (dict): dictionary with the vocabulary of the output language
        unk_token_str (str): unknown token
    """

    output_sentences = []

    for sentence in input_sentences:
        output_words, _ = evaluate_step(encoder, decoder, sentence,
                                        word2index_lang1, index2word_lang2,
                                        unk_token_str)
        output_sentence = ' '.join(output_words)
        print('>', sentence)
        print('<', output_sentence)
        print('')
        output_words = output_words[:-1]
        output_sentences.append(output_words)

    targets = [target.split() for target in targets]

    bleu = calculate_bleu(targets, output_sentences)
    print(bleu)


def calculate_bleu(refs: list, hypos: list) -> dict:
    """
    Calculate BLEU score for a single candidate caption against
    multiple reference captions.

    Args:
        refs Dict[str, List[str]]: A list of reference captions.
        hypos Dict[str, List[str]]: A list of candidate captions.

    Returns:
        dict: BLEU score for each n-gram.
    """

    bleu_dict: dict = {'1-gram': [],
                       '2-gram': [],
                       '3-gram': [],
                       '4-gram': []}

    smoothing = SmoothingFunction().method1
    # smoothing = SmoothingFunction().method7

    weights = {
            '1-gram': (1, 0, 0, 0),
            '2-gram': (0.5, 0.5, 0, 0),
            '3-gram': (0.33, 0.33, 0.33, 0),
            '4-gram': (0.25, 0.25, 0.25, 0.25)
        }

    for sentence in hypos:

        # The hypos[img_id] is a list with only one element
        # that is the caption predicted by the model
        hypo_tokens = sentence

        # The refs[img_id] is a list with possible
        # descriptions of the image
        refs_tokens = refs

        for key, value in weights.items():
            bleu = sentence_bleu(refs_tokens,
                                 hypo_tokens,
                                 weights=value,
                                 smoothing_function=smoothing)
            bleu_dict[key].append(bleu)

    bleu_means = {key: np.mean(value) for key, value in bleu_dict.items()}

    # Return the average BLEU score
    return bleu_means


if __name__ == "__main__":

    # Hyperparameters
    batch_size: int = 1

    # Parameteres data
    SOS_token: int = 1
    EOS_token: int = 2
    unk_token: int = 3
    unk_token_str: str = "UNK"
    max_length: int = 15
    namelang_in: str = 'eng'
    namelang_out: str = 'fra'
    # namelang_out: str = 'spa'

    # Set device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data: pd.DataFrame = pd.read_csv('data/%s-%s_test.csv' % (namelang_in, namelang_out))
    data = data.sample(n=100)
    normalized_data = data.map(lambda x: normalizeString(x)
                               if pd.notna(x) and x.strip() != '' else None)
    pairs: List[List[str]] = [list(row.dropna())
                              for _, row in normalized_data.iterrows()]
    pairs = [pair for pair in pairs
             if len(pair) == 2 and pair[0] != '' and pair[1] != '']
    pairs = filterPairs(pairs, max_length)

    word2index_lang1 = load_vocab(f"{namelang_in}")
    word2index_lang2 = load_vocab(f"{namelang_out}")

    # Crear un diccionario inverso
    index2word_lang2 = {valor: clave for clave, valor in word2index_lang2.items()}

    # load models
    encoder = load_model('models/best_encoder.pt', device)
    decoder = load_model('models/best_decoder.pt', device)

    sentences_input = [pair[0] for pair in pairs]
    targets = [pair[1] for pair in pairs]

    # Inputs with targets
    evaluate(encoder, decoder, sentences_input, targets,
             word2index_lang1, index2word_lang2, unk_token_str)

    # # just wanting to translate a
    # # sentence without evaluating
    # sentence = '"The computer is broken"'
    # output_words, _ = evaluate_step(encoder, decoder, sentence,
    #                                 word2index_lang1, index2word_lang2, unk_token_str)
    # output_sentence = ' '.join(output_words)
    # print('>', sentence)
    # print('<', output_sentence)
