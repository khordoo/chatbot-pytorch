import os
import joblib
from datetime import datetime
from src.model import EncoderGRU
from src.model import DecoderGRU
from src.model import EncoderDecoderMediator
from src.tokenizer import Tokenizer
from src.exceptions import UnrecognizedWordException


class Chatbot:
    """A conversational Chatbot class that loads the previously trained encoder and decoder
       to respond to the user conversation.
    """

    def __init__(self, saved_models_directory):
        self.model = {}
        self._load_saved_nets(saved_models_directory)

    def _load_saved_nets(self, directory, step=0):
        self.model = joblib.load(os.path.join(directory, 'api/save/encoder-decoder-{}.joblib'.format(step)))
        encoder_path = os.path.join(directory, 'api/save/encoder-{}.dat'.format(step))
        decoder_path = os.path.join(directory, 'api/save/decoder-{}.dat'.format(step))
        self.model.load(encoder_state_path=encoder_path, decoder_state_path=decoder_path, map_location='cpu')

    def reply(self, genre=None, query=None, response_length=10, mode='argmax'):
        respond = self.model.predict_response(question_text=query, max_len=response_length, mode=mode)
        return {
            "reply": respond,
            'time': datetime.now().isoformat(timespec='seconds') + 'Z'
        }

    def _get_genre(self, filename):
        return filename.split('.')[0]
