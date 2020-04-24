import os
import torch
from src.utility import Utility
from src.model import EncoderDecoder
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import train
MAX_PRED_LENGTH=5
BASE_DIR = 'saves'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_states_path = 'saves/epoch_2_blue_0.165_bleu_test0.148.pt'
model_states_path = 'epoch_40_blue_0.579_bleu_test_0.104.pt'
tokenizer = torch.load(os.path.join(BASE_DIR, 'tokenizer_comedy-no-contraction.pt'), map_location=DEVICE)

encoder_decoder = EncoderDecoder(vocab_size=tokenizer.dictionary_size,
                                 hidden_size=train.HIDDEN_SIZE,
                                 embedding_dim=train.EMBEDDING_DIM,
                                 bidirectional=False
                                 ).to(DEVICE)

encoder_decoder.load_state_dict(
    torch.load(os.path.join('api/save/', 'epoch_40_blue_0.579_bleu_test_0.104.pt'), map_location=DEVICE))

texts = ['hi there', 'where are you?', 'hi']
texts, indexes = Utility.predict(source_texts=texts, model=encoder_decoder, tokenizer=tokenizer, device=DEVICE,
                                 max_prediction_len=MAX_PRED_LENGTH)
print(texts)
