import os
import torch
from src.utility import Utility
from src.model import EncoderDecoder
import train

MAX_PRED_LENGTH = 15
BASE_DIR = 'saves'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_states_path = 'saves/epoch_85.pt'
tokenizer = torch.load(os.path.join(BASE_DIR, 'tokenizer_comedy-no-contraction.pt'), map_location=DEVICE)

encoder_decoder = EncoderDecoder(vocab_size=tokenizer.dictionary_size,
                                 hidden_size=train.HIDDEN_SIZE,
                                 embedding_dim=train.EMBEDDING_DIM,
                                 bidirectional=False
                                 ).to(DEVICE)

encoder_decoder.load_state_dict(
    torch.load(model_states_path, map_location=DEVICE))


def predict(text):
    assert isinstance(text, list)
    response, indexes = Utility.predict(source_texts=text, model=encoder_decoder, tokenizer=tokenizer, device=DEVICE,
                                        max_prediction_len=MAX_PRED_LENGTH)
    return response


response = predict(['how are you?'])
print('response:', response)
