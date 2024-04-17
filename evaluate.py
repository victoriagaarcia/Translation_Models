import torch
from utils import load_model, load_vocab
from data import normalizeString


def translator(encoder,
               decoder,
               sentence,
               lan1_word2int,
               lan2_int2word,
               max_length,
               start_token,
               end_token,
               device):
    
    encoder.eval()
    decoder.eval()

    # Convert the input sentence to a tensor 
    tokens = [normalizeString(str(s)) for s in sentence.split(' ')]
    input_tensor = torch.tensor([lan1_word2int[start_token]] + [lan1_word2int[word] for word in tokens]
                                    + [lan1_word2int[end_token]], dtype=torch.long)
    
    input_tensor = input_tensor.view(1, -1).to(device)  # batch_first=True

    _, encoder_hidden, encoder_cell = encoder(input_tensor)
    # Initialize the decoder input with the start token
    decoder_input = torch.tensor([[lan1_word2int[start_token]]], dtype=torch.long).to(device)

    # Initialize the decoder hidden state with the encoder hidden state
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    # Decode the sentence
    decoded_words = []
    last_word = torch.tensor([[lan1_word2int[end_token]]]).to(device)

    for _ in range(max_length):
        logits, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
        next_token = torch.argmax(logits, dim=1)
        decoder_input = torch.tensor([[next_token]]).to(device)

        if next_token == lan1_word2int[end_token]:
            print('he terminado en: ', _)
            break
        else:
            print(next_token.item())
            decoded_words.append(lan2_int2word.get(next_token.item()))

    return ' '.join(decoded_words)


def main():
   
    sentence = 'my name is hello.'
    max_length = 15
    
    input_lang = 'English'
    output_lang = 'Spanish'
    
    start_token = '<SOS>'
    end_token = '<EOS>'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = load_model("models/best_encoder.pt").to(device)
    decoder = load_model("models/best_decoder.pt").to(device)
    
    lan1_word2int = load_vocab(f"{input_lang}")
    lan2_word2int = load_vocab(f"{output_lang}")
      
    # Crear un diccionario inverso
    lan2_int2word = {valor: clave for clave, valor in lan2_word2int.items()}
    sentence_translated = translator(encoder, decoder, sentence, lan1_word2int, lan2_int2word, max_length, start_token, end_token, device)
    
    print(sentence_translated)  
    
if __name__ == "__main__":
    main()
    