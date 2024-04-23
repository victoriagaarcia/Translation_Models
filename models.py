from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 1
MAX_LENGTH = 15


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_p=0.4):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding: nn.Embedding  = nn.Embedding(input_size, hidden_size)
        self.gru: nn.GRU = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class EncoderRNN_Embed(nn.Module):
    def __init__(self, hidden_size, embeddings, dropout_p=0.4,  embed_size = 300):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding: nn.Embedding  = embeddings.embedding
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input: torch.Tensor):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super(DecoderRNN, self).__init__()
        self.embedding: nn.Embedding = nn.Embedding(output_size, hidden_size)
        self.gru: nn.GRU = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, device: torch.device, SOS_token: int,
                encoder_outputs: torch.Tensor, encoder_hidden: torch.Tensor,
                target_tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_outputs.size(0)
        max_length = target_tensor.size(1) if target_tensor is not None else MAX_LENGTH

        decoder_input = torch.full((batch_size, 1), SOS_token,
                                   dtype=torch.long).to(device)
        decoder_hidden = encoder_hidden
        decoder_outputs_list = []

        for i in range(max_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input,
                                                               decoder_hidden)
            decoder_outputs_list.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs: torch.Tensor = torch.cat(decoder_outputs_list, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs, decoder_hidden
        # We return `None` for consistency in the training loop

    def forward_step(self, input: torch.Tensor, hidden: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super(BahdanauAttention, self).__init__()
        self.Wa: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.Ua: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.Va: nn.Linear = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout_p=0.4):
        super(AttnDecoderRNN, self).__init__()
        self.embedding: nn.Embedding = nn.Embedding(output_size, hidden_size)
        self.attention: BahdanauAttention = BahdanauAttention(hidden_size)
        self.gru: nn.GRU = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out: nn.Linear = nn.Linear(hidden_size, output_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs: torch.Tensor,
                encoder_hidden: torch.Tensor, target_tensor=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = encoder_outputs.size(0)

        max_length = target_tensor.size(1) if target_tensor is not None else MAX_LENGTH

        decoder_input = torch.empty(batch_size, 1, dtype=torch.long,
                                    device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs_list = []
        attentions_list = []

        for i in range(max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            decoder_outputs_list.append(decoder_output)
            attentions_list.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # # Without teacher forcing: use its own predictions as the next input
                # scores, words = decoder_output.topk(beam_size)
                
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input
                
                # # normalize the scores
                # scores = scores / scores.sum()
                
                # words_predictions = []
                # probabilities = []
                    
                # # get the k best captions
                # for i in range(beam_size):
                #     predicted = torch.tensor([words[0, i].item()], device=device)
                #     probability = scores[0, i].item()
                #     words_predictions.append([predicted.item()])
                #     probabilities.append(probability)
                    
                # for j in range(2, MAX_LENGTH):
                #     # initialize the lists
                #     new_words_predictions = []
                #     new_probabilities = []
                        
                #     # for each caption in the beam, get the k best captions
                #     for i in range(beam_size):
                #         # if the last word is the end token, stop
                #         if words_predictions[i][-1] == word2index["EOS"]:
                #             new_words_predictions.append(words_predictions[i])
                #             new_probabilities.append(probabilities[i])
                            
                #         else:
                #             # pass the features and the states through the lstm
                #             decoder_output, decoder_hidden, attn_weights = self.forward_step(
                #                 words_predictions[i][-1], decoder_hidden, encoder_outputs
                #             )
                            
                #             scores, words = decoder_output.topk(beam_size)
                #             # normalize the scores
                #             scores = scores / scores.sum()

                #             # get the k best captions and its probabilities
                #             for k in range(beam_size):
                #                 predicted = torch.tensor(
                #                     [words[0, k].item()], device=device
                #                 )
                #                 new_probabilities.append(
                #                     probabilities[i]
                #                     * scores[0, k].item()
                #                     * (MAX_LENGTH - j)
                #                     / MAX_LENGTH
                #                 )
                #                 new_words_predictions.append(words_predictions[i] + [predicted.item()])
                    
                # # rank the captions by probability and get the k best
                # best_words_predictions = sorted(
                #     list(
                #         zip(
                #             new_words_predictions,
                #             new_probabilities,
                #         )
                #     ),
                #     key=lambda x: x[1],
                #     reverse=True,
                # )[:beam_size]
                
                # words_predictions = [x[0] for x in best_words_predictions]
                # probabilities = [x[1] for x in best_words_predictions]

                # # normalize the probabilities
                # probabilities = [p / sum(probabilities) for p in probabilities]
                
                


        decoder_outputs: torch.Tensor = torch.cat(decoder_outputs_list, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions_list, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
    

class AttnDecoderRNN_Embed(nn.Module):
    def __init__(self, hidden_size, output_size, embeddings, dropout_p=0.1, embed_size=300):
        super().__init__()
        self.embedding = embeddings.embedding
        self.attention = BahdanauAttention(embed_size)
        self.gru = nn.GRU(2 * embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)

        max_length = target_tensor.size(1) if target_tensor is not None else MAX_LENGTH
        
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
