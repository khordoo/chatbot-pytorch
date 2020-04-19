import os
import joblib
from datetime import datetime
from src.seq2seq import EncoderDecoder, EncoderLSTM
from src.seq2seq import DecoderLSTM, Tokenizer, UnrecognizedWordException


class Chatbot:
    """A Chatbot class that used a saved trained PyTorch
       NLP models to respond to text queries"""

    def __init__(self, saved_models_directory):
        self.models = {}
        self._load_saved_nets(saved_models_directory)

    def _load_saved_nets(self, directory):
        for filename in os.listdir(directory):
            self.models[self._get_genre(filename)] = joblib.load(os.path.join(directory, filename))

    def reply(self, genre, query, response_length=10, mode='argmax'):
        respond = self.models[genre].predict_response(question_text=query, max_len=response_length, mode=mode)
        return {
            "reply": respond,
            "personality": genre,
            'time': datetime.now().isoformat(timespec='seconds') + 'Z'
        }

    def _get_genre(self, filename):
        return filename.split('.')[0]
