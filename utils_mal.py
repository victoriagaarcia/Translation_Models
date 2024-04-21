# deep learning libraries
import torch
import numpy as np
from torch.jit import RecursiveScriptModule
import unicodedata
import re

# other libraries
import os
import random
import json


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # # save scripted model
    # model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    # model_scripted.save(f"models/{name}.pt")
    torch.save(model, f"models/{name}.pt" )

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    # model: RecursiveScriptModule = torch.jit.load(f"{name}")
    model = torch.load(f"{name}")
    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None

def save_vocab(vocab, name):
    
    # create folder if it does not exist
    if not os.path.isdir("vocab"):
        os.makedirs("vocab")
        
    # save vocab como un json para guardar el diccionario
    with open(f'vocab/vocab_{name}.json', 'w') as f:
        json.dump(vocab, f)

def load_vocab(name):
    
    # load vocab
    with open(f'vocab/vocab_{name}.json', 'r') as f:
        vocab = json.load(f)
        
    return vocab

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

# def collate_fn(batch) :
#     """
#     Prepares and returns a batch for training/testing in a torch model.

#     This function sorts the batch by the length of the text sequences in descending order,
#     tokenizes the text using a pre-defined word-to-index mapping, pads the sequences to have
#     uniform length, and converts labels to tensor.

#     Args:
#         batch (List[Tuple[List[str], int]]): A list of tuples, where each tuple contains a
#                                              list of words (representing a text) and an integer label.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three elements:
#             - texts_padded (torch.Tensor): A tensor of padded word indices of the text.
#             - labels (torch.Tensor): A tensor of labels.
#             - lengths (torch.Tensor): A tensor representing the lengths of each text sequence.
#     """
#     # TODO: Sort the batch by the length of text sequences in descending order
#     batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

#     # TODO: Unzip texts and labels from the sorted batch
#     texts, labels = zip(*batch)

#     # TODO: Convert texts to indices using the word2idx function and w2v_model
#     texts_indx = [word2idx(w2v_model, text) for text in texts]

#     # TODO: Calculate the lengths of each element of texts_indx.
#     # The minimum length shall be 1, in order to avoid later problems when training the RNN
#     lengths = [max(len(text), 1) for text in texts_indx]

#     # TODO: Pad the text sequences to have uniform length
#     texts_padded: torch.Tensor = pad_sequence(texts_indx, batch_first=True)

#     # TODO: Convert labels to tensor
#     labels: torch.Tensor = torch.tensor(labels, dtype=torch.float32)

#     return texts_padded, labels, lengths