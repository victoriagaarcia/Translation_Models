import torch
from torch import nn
import math
from torch.nn.utils.rnn import pad_sequence

class CreateEmbeddings(nn.Module):

    def __init__(self, n_vocab: int, d_embed: int):
        super().__init__()
        self.n_vocab: int = n_vocab
        self.d_embed: int = d_embed

        self.in_embed = nn.Embedding(n_vocab, d_embed)

    def forward(self, input_words: torch.Tensor) -> torch.Tensor:
        input_vectors: torch.Tensor = self.in_embed(input_words)
        return input_vectors
    # En este forward, la práctica de NLP lo devuelve así tal cual aplicando los embeddings
    # Pero en el tutorial lo multiplica por math.sqrt(self.d_embed) para normalizar los embeddings


class PositionEncoding(nn.Module):
    '''
    Esta clase nos calcula un vector de embeddings que indica la posición de la palabra dentro de la frase.
    Este se sumará a los embeddings de las palabras ya calculados.
    '''

    def __init__(self, d_embed: int, max_len: int, dropout: float):
        super().__init__()

        self.d_embed = d_embed
        self.max_len = max_len # Máxima longitud que puede tener una frase
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space
        # Inicalizamos una matriz de ceros de tamaño max_len x d_embed. 
        # porque será de el número de dimensiones que queramos para el embedding por la longitud máxima de una de nuestras frases
        pe = torch.zeros(max_len, d_embed)
        # Creamos el vector que representa la posición de la palabra en la frase
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # Hay que aplicar esta fórmula extraña que me ha dicho copi y tbn salía en el vídeo
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * -(math.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term) # Pares
        pe[:, 1::2] = torch.cos(position * div_term) # Impares 
        pe = pe.unsqueeze(0) # Añadimos una dimensión para que cuadre con el batch
        self.register_buffer('pe', pe) # Lo guardamos de manera que no sea un parámetro entrenable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Sumamos el vector de posición a los embeddings de las palabras
        return self.dropout(x) # Aplicamos dropout


# SEQ2SEQ MODEL
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers = 1):
        super().__init__()
        # Define the network parameters
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        # Flip the input sequence to make the model easier to learn
        x = torch.flip(x, [1])
        # Embed the input
        embedded = self.embedding(x)
        # Pass the embedded input through the LSTM
        output, (hidden, cell) = self.lstm(embedded)
        return output, hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden, cell):
        # Embed the input
        x = self.embedding(x)
        # Pass the embedded input through the LSTM
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        # Pass the LSTM output through the fully connected layer
        output = self.fc(output).reshape(x.shape[0], -1)
        return output, hidden, cell

# definir parámetros
# crear encoder y decoder en el device

# eng_word2int = {word: i for i, word in enumerate(eng_vocab)}

# TRADUCTOR (EVAL)
# lan1_word2int: diccionario en el que cada palabra de la lengua 1 tiene un índice
# lan2_word2int: diccionario en el que cada palabra de la lengua 2 tiene un índice

def translator(encoder, decoder, sentence, lan1_word2int, lan2_word2int, max_length, start_token, end_token, device):
    encoder.eval()
    decoder.eval()

    # Convert the input sentence to a tensor (Esto creo q se hace ya en la función de data.py)

    _, encoder_hidden, encoder_cell = encoder(sentence)
    # Initialize the decoder input with the start token
    decoder_input = torch.tensor([[lan1_word2int[start_token]]], dtype=torch.long)

    # Initialize the decoder hidden state with the encoder hidden state
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    # Decode the sentence
    decoded_words = []
    last_word = torch.tensor([[lan1_word2int[end_token]]]).to(device)

    for _ in range(max_length):
        logits, decoder_hidden, decoder_cell = decoder(last_word, decoder_hidden, decoder_cell)
        next_token = torch.argmax(logits, dim=1)
        last_word = torch.tensor([[next_token]]).to(device)

        if next_token == lan2_word2int[end_token]:
            break
        else:
            decoded_words.append(lan2_word2int.get(next_token.item()))

    return ' '.join(decoded_words)

