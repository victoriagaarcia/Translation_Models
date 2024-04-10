import torch
from torch import nn
import math

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
        self.max_len = max_len
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

