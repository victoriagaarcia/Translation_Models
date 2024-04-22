# deep learning libraries
import torch
from flair.embeddings import WordEmbeddings

# own modules
from data import get_dataloader
from models import EncoderRNN, AttnDecoderRNN
from train import train
# from evaluate_pytorch import evaluateRandomly
from utils import save_model, set_seed

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)


def main() -> None:
    """
    This function is the main program for the training
    """


# Hyperparameters
hidden_size: int = 256
batch_size: int = 128
lr: float = 0.0001
n_epochs: int = 60

# Parameters plot
print_every: int = 5

# Parameteres data
PAD_token: int = 0
SOS_token: int = 1
EOS_token: int = 2
max_length: int = 10
namelang_in: str = 'eng'
namelang_out: str = 'fra'

# Set device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define encoder and decoder names
name_encoder: str = f"encoder_lr_{lr}_hs_{hidden_size}_b_{batch_size}_nep_{n_epochs}" + \
                    f"_maxl_{max_length}_lang1_{namelang_in}_lang2_{namelang_out}"
name_decoder: str = f"decoder_lr_{lr}_hs_{hidden_size}_b_{batch_size}_nep_{n_epochs}" + \
                    f"_maxl_{max_length}_lang1_{namelang_in}_lang2_{namelang_out}"

# load embeddings
input_lang_embeddings = WordEmbeddings('en')
output_lang_embeddings = WordEmbeddings('fr')

# load data
input_lang, output_lang, train_dataloader, val_dataloader = get_dataloader(
    batch_size, SOS_token, EOS_token, max_length, namelang_in, namelang_out)

# Define model: encoder + decoder with attention
encoder: torch.nn.Module = EncoderRNN(input_lang.n_words, hidden_size).to(device)
# decoder: torch.nn.Module = DecoderRNN(hidden_size, output_lang.n_words).to(device)
decoder: torch.nn.Module = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

# Train the model
train(train_dataloader, val_dataloader, encoder, decoder,
      n_epochs, lr, device, print_every)

# Save modes (encoder and decoder)
save_model(encoder, name_encoder)
save_model(decoder, name_decoder)
