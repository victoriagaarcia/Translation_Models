import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score # pip install torchtext
# Other BLEU source?

from tqdm import tqdm
from typing import Final

# Our own libraries
# from models import Encoder, Decoder
# from train_functions import train_step, eval_step

# save_model functions...

DATA_PATH: Final[str] = 'data/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set the seed...

def main():
    # training parameters... 
    epochs = 0
    lr = 0.0
    batch_size = 0
    
    # model parameters...
    vocab_size = 0
    embed_size = 0
    hidden_size = 0

    # scheduler parameters... (step_size, gamma)

    # Load the data
    # train_data = ...
    # + vocabs, etc.

    # Create the model
    # encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers).to(device)
    # decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers).to(device)

    # Define loss functions
    ce_loss = torch.nn.CrossEntropyLoss()
    bleu_loss = bleu_score

    # Define the optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define the scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Define the writer
    writer = SummaryWriter()

    # Training loop
    for epoch in range(epochs):
        # train_step(model, train_data, mean, std, ce_loss, optimizer, writer, epoch, device)
        # eval_step(model, val_data, mean, std, ce_loss, scheduler, writer, epoch, device)
        pass

    # save_model(model, name)

if __name__ == '__main__':
    main()