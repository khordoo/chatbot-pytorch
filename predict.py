import os
import torch
from src.utility import Utility
from src.model import AttentionEncoderDecoder
from src.tokenizer import Tokenizer
from train import HIDDEN_SIZE, EMBEDDING_DIM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SAVE_DIRECTORY = 'saves'
SAVED_MODEL_NAME = 'epoch_4_2020-04-25T21:19:44.pt'
TOKENIZER_FILE_NAME = 'tokenizer_dict.pt'
MAX_PRED_LENGTH = 15

tokenizer = Tokenizer()
tokenizer.load_state_dict(os.path.join(SAVE_DIRECTORY, TOKENIZER_FILE_NAME), device=DEVICE)
encoder_decoder = AttentionEncoderDecoder(vocab_size=tokenizer.dictionary_size,
                                          hidden_size=HIDDEN_SIZE,
                                          embedding_dim=EMBEDDING_DIM,
                                          bidirectional=False
                                          ).to(DEVICE)
encoder_decoder.load_state_dict(torch.load(os.path.join(SAVE_DIRECTORY,
                                                        SAVED_MODEL_NAME),
                                           map_location=DEVICE))


def predict_response(text):
    """Predicts the response to the input text
       text: list
           input text to be used as input to the seq2seq model.
       Example:
       text=['who are you?']
    """
    assert isinstance(text, list)
    tokens = tokenizer.convert_text_to_number(text)
    try:
        # Skip prediction if input contains any unrecognized words
        found_index = tokens[0][:-1].tolist().index(tokenizer.unknown_index)
        return ["Sorry, there is a word that I don't understand:\n"]
    except ValueError:
        # No unknown word was found,
        # continue with prediction
        pass
    response, indexes = Utility.predict(source_texts=text, model=encoder_decoder, tokenizer=tokenizer, device=DEVICE,
                                        max_prediction_len=MAX_PRED_LENGTH)
    return response


query = ['who are you?']
response = predict_response(query)[0]
print('response:\n', response)
