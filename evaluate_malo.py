import torch
from utils_malo import load_model, load_vocab
from data import normalizeString
import torchtext


def translator(encoder,
               decoder,
               sentence,
               lan1_word2int,
               lan2_int2word,
               max_length,
               start_token,
               end_token,
               unknown_token,
               device):
    
    encoder.eval()
    decoder.eval()

    # Convert the input sentence to a tensor
    
    tokens = normalizeString(str(sentence)).split()
    
    if len(tokens) <= max_length: 
        input_tensor = torch.tensor([lan1_word2int[start_token]] + [lan1_word2int[word] if word in lan1_word2int else lan1_word2int[unknown_token] for word in tokens ])
        
        input_tensor = input_tensor.view(1, -1).to(device)  # batch_first=True

    # Get the length of the input sentence
        text_length = torch.tensor(len(tokens)+2).view(1, -1).to(device)

        _, encoder_hidden, encoder_cell = encoder(input_tensor,text_length)
        
        # Initialize the decoder input with the start token
        decoder_input = torch.tensor([[lan1_word2int[start_token]]], dtype=torch.long).to(device)
        # print('shape decoder input en eval', decoder_input.shape)
        # print('shape encoder hidden en eval', encoder_hidden.shape)
        # decoder_input = torch.concat((decoder_input, encoder_hidden.transpose(0, 1).reshape(1, -1)), dim=1).long()
        
        # Initialize the decoder hidden state with the encoder hidden state
        first_dim = int(encoder_hidden.shape[0] / 2)
        decoder_hidden = encoder_hidden.view(first_dim, 1, -1)
        decoder_cell = encoder_cell.view(first_dim, 1, -1)

        # Decode the sentence
        decoded_words = [start_token]
        
        for _ in range(1, max_length):
            logits, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
            next_token = torch.argmax(logits, dim=1)
            decoder_input = torch.tensor([[next_token]]).to(device)
            # print(next_token)
            # decoder_input = torch.concat((decoder_input, encoder_hidden.transpose(0, 1).reshape(1, -1)), dim=1).long()

            if next_token == lan1_word2int[end_token]:
                break
            else:
                decoded_words.append(lan2_int2word.get(next_token.item()))

        return ' '.join(decoded_words)
    else: 
        return f'ERROR: the sentence is longer than {max_length}'


def main():
    max_length = 25
    
    input_lang = 'English'
    output_lang = 'Spanish'
    
    start_token = '<SOS>'
    end_token = '<EOS>'
    unknown_token = '<UNK>'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = load_model("models/best_encoder.pt").to(device)
    decoder = load_model("models/best_decoder.pt").to(device)
    
    lan1_word2int = load_vocab(f"{input_lang}")
    lan2_word2int = load_vocab(f"{output_lang}")
      
    # Crear un diccionario inverso
    lan2_int2word = {valor: clave for clave, valor in lan2_word2int.items()}
    # read line by line from the file in the data folder named evaluate.txt
    with open('data/evaluate.txt', 'r') as file:
        lines = file.readlines()
        
        for sentence in lines:
            sentence_translated = translator(encoder, decoder, sentence, lan1_word2int, lan2_int2word, max_length, start_token, end_token, unknown_token, device)
            print(sentence_translated)  
    
if __name__ == "__main__":
    main()
    