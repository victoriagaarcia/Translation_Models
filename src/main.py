# deep learning libraries
import torch
# from flair.embeddings import WordEmbeddings

# own modules
from data import get_dataloader
from models import EncoderRNN, AttnDecoderRNN
# from models import DecoderRNN, EncoderRNN_Embed
from train import train
from utils import save_model, set_seed, save_vocab
from torch.utils.tensorboard import SummaryWriter


# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)


def main() -> None:
    """
    This function is the main program for the training
    """


# Hyperparameters
hidden_size: int = 512
batch_size: int = 64
lr: float = 0.0001
n_epochs: int = 60

# Parameters plot
print_every: int = 5

# Parameteres data
PAD_token: int = 0
SOS_token: int = 1
EOS_token: int = 2

max_length: int = 15
unk_token_str: str = "UNK"

namelang_in: str = 'eng'
namelang_out: str = 'fra'
# namelang_out: str = 'spa'

# Set device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define encoder and decoder names and write
name_encoder: str = f"encoder_lr_{lr}_hs_{hidden_size}_b_{batch_size}_nep_{n_epochs}" + \
                    f"_maxl_{max_length}_lang1_{namelang_in}_lang2_{namelang_out}"
name_decoder: str = f"decoder_lr_{lr}_hs_{hidden_size}_b_{batch_size}_nep_{n_epochs}" + \
                    f"_maxl_{max_length}_lang1_{namelang_in}_lang2_{namelang_out}"
writer: SummaryWriter = SummaryWriter(f"runs/{name_encoder}")

# load embeddings -- we can use embeddings pretrained
# input_lang_embeddings = WordEmbeddings('en')
# output_lang_embeddings = WordEmbeddings('es')

# load data
input_lang, output_lang, train_dataloader, val_dataloader = get_dataloader(
  batch_size, unk_token_str, EOS_token, max_length, namelang_in, namelang_out)

# Define model: encoder + decoder with attention
encoder: torch.nn.Module = EncoderRNN(input_lang.n_words, hidden_size).to(device)
# with embeddings pretrained
# encoder: torch.nn.Module = EncoderRNN_Embed(hidden_size,
# input_lang_embeddings).to(device)

# without attention
# decoder: torch.nn.Module = DecoderRNN(hidden_size, output_lang.n_words).to(device)
decoder: torch.nn.Module = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
# with embeddings pretrained
# decoder: torch.nn.Module = AttnDecoderRNN_Embed(hidden_size,
# output_lang.n_words, output_lang_embeddings).to(device)

# Save vocab
save_vocab(input_lang.word2index, f"{namelang_in}")
save_vocab(output_lang.word2index, f"{namelang_out}")

# Train the model
train(train_dataloader, val_dataloader, encoder, decoder,
      n_epochs, lr, device, writer, print_every)

# Save models (encoder and decoder)
save_model(encoder, name_encoder)
save_model(decoder, name_decoder)
