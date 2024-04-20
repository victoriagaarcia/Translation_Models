import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from flair.embeddings import WordEmbeddings
# Other BLEU source?

from tqdm.auto import tqdm
from typing import Final

# Our own libraries
from models import Encoder, Decoder
from train_functions import train_step, val_step
from data import get_dataloader
from utils import set_seed, save_model, save_vocab
from evaluate import translator

# save_model functions...

# DATA_PATH: Final[str] = 'data/'

if torch.cuda.is_available():
    device: Final[torch.device] = torch.device('cuda')
elif torch.backends.mkl.is_available():
    device: Final[torch.device] = torch.device('mps')
else:
    device: Final[torch.device] = torch.device('cpu')

# set the seed...
# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

def main():
    # training parameters... 
    epochs = 400
    lr = 0.03
    batch_size = 128 # 32
    
    # model parameters...
    # vocab_size = 0
    embed_size = 300
    hidden_size = 128
    num_layers = 2

    step_size = 25
    gamma = 0.5
    
    input_lang = 'English'
    output_lang = 'Spanish'

    start_token = '<SOS>'
    end_token = '<EOS>'
    pad_token = '<PAD>'
    unknown_token = '<UNK>'
    max_length = 20
    
    # scheduler parameters... (step_size, gamma)

    # Load the data
    # train_data = ...
    # + vocabs, etc.
    
    input_lang_embeddings = WordEmbeddings('en')
    output_lang_embeddings = WordEmbeddings('es')
    
    input_lang_embeddings.embedding
    
    train_dataloader, val_dataloader, input_lang_class, output_lang_class = get_dataloader(batch_size, input_lang, output_lang, max_length)
    
    # define name 
    name_enc: str = f"model_lr_{lr}_hs_{hidden_size}_{batch_size}_{epochs}_encoder"
    name_dec: str = f"model_lr_{lr}_hs_{hidden_size}_{batch_size}_{epochs}_decoder"
    # Define the writer
    writer: SummaryWriter = SummaryWriter(f"runs/{name_enc}_{name_dec}")
    
    vocab_size_input = input_lang_class.n_words
    vocab_size_output = output_lang_class.n_words

    # Create the model
    encoder = Encoder(vocab_size_input, embed_size, hidden_size, num_layers, input_lang_embeddings).to(device)
    decoder = Decoder(vocab_size_output, embed_size, hidden_size*2, num_layers, output_lang_embeddings).to(device)

    # Define loss functions
    ce_loss = torch.nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer_encoder = torch.optim.AdamW(encoder.parameters(), lr=lr)
    optimizer_decoder = torch.optim.AdamW(decoder.parameters(), lr=lr)

    # Define the scheduler
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=step_size, gamma=gamma)
    scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer_decoder, step_size=step_size, gamma=gamma)

    # Define the vocabularies
    vocab_lang1 = input_lang_class.word2index
    vocab_lang2 = output_lang_class.word2index
    
    # Save the vocabs
    save_vocab(vocab_lang1, input_lang)
    save_vocab(vocab_lang2, output_lang)
    
    print(len(train_dataloader))
    # Training loop
    for epoch in tqdm(range(epochs)):
        
        train_step(encoder, decoder, train_dataloader, ce_loss, optimizer_encoder, optimizer_decoder, writer, epoch, batch_size, device, vocab_lang1, output_lang_class.index2word)
        # val_step(encoder, decoder, val_dataloader, writer, epoch, batch_size, device, vocab_lang1, vocab_lang2)

        scheduler_encoder.step()
        scheduler_decoder.step()

        # Crear un diccionario inverso
        lan2_int2word = {valor: clave for clave, valor in vocab_lang2.items()}
        sentence1 = "the debate is closed"
        print(translator(encoder, decoder, sentence1, vocab_lang1, lan2_int2word, max_length, start_token, end_token,unknown_token, device))
        sentence2 = "I regret that"
        print(translator(encoder, decoder, sentence2, vocab_lang1, lan2_int2word, max_length, start_token, end_token,unknown_token, device))
        sentence3 = "Is this too little too late?"
        print(translator(encoder, decoder, sentence3, vocab_lang1, lan2_int2word, max_length, start_token, end_token,unknown_token, device))

    save_model(encoder, name_enc)
    save_model(decoder, name_dec)
    
if __name__ == '__main__':
    main()