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
    torch.save(model, f"models/{name}.pt")

    return None


def load_model(name: str, device) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    # model: RecursiveScriptModule = torch.jit.load(f"{name}")
    model = torch.load(f"{name}", map_location=torch.device(device))
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
    """
    """
    # create folder if it does not exist
    if not os.path.isdir("vocab"):
        os.makedirs("vocab")

    # save vocab como un json para guardar el diccionario
    with open(f'vocab/vocab_{name}.json', 'w') as f:
        json.dump(vocab, f)


def load_vocab(name):
    """
    """
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
