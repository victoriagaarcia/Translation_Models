from __future__ import unicode_literals, print_function, division
from io import open

import torch
import numpy as np
from data import tensorFromSentence

from data import get_dataloader, normalizeString, Lang
from utils import load_model
from torch.jit import RecursiveScriptModule

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu


def evaluate(encoder: RecursiveScriptModule, decoder: RecursiveScriptModule,
             sentence: str, input_lang: Lang, output_lang: Lang, unk_token_str: str):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, EOS_token, unk_token_str)
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
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluateRandomly(encoder: RecursiveScriptModule, decoder: RecursiveScriptModule,
                     input_lang: Lang, output_lang: Lang, unk_token_str: str):

    output_sentences = []

    with open('data/evaluate.txt', 'r') as file:
        lines = file.readlines()

        for sentence in lines:
            sentence = normalizeString(sentence)
            output_words, _ = evaluate(encoder, decoder, sentence, input_lang,
                                       output_lang, unk_token_str)
            output_sentence = ' '.join(output_words)
            print('>', sentence)
            print('<', output_sentence)
            print('')
            output_words = output_words[:-1]
            output_sentences.append(output_words)

    return output_sentences


def evaluate_targets():
    with open('data/evaluate_targets.txt', 'r') as file:
        lines = file.readlines()
        targets = []
        for sentence in lines:
            targets.append(sentence.split())
    return targets


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
    max_length: int = 10
    namelang_in: str = 'eng'
    namelang_out: str = 'fra'

    # Set device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    input_lang, output_lang, train_dataloader, val_dataloader = get_dataloader(
        batch_size, unk_token_str, EOS_token, max_length, namelang_in, namelang_out)

    # load models
    encoder = load_model('models/best_encoder.pt', device)
    decoder = load_model('models/best_decoder.pt', device)

    output_sentences = evaluateRandomly(encoder, decoder, input_lang,
                                        output_lang, unk_token_str)
    targets = evaluate_targets()

    print(calculate_bleu(targets, output_sentences))
