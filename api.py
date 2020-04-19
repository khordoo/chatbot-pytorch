import json
from flask import Flask, request, abort
from api.src.chatbot import Chatbot
from src.seq2seq import EncoderDecoder, EncoderLSTM, DecoderLSTM, Tokenizer, UnrecognizedWordException

directory = '/Users/mahmoodkhordoo/personal/git/Reinforcement-Learning-LSTM-chartbot-pytorch/api/save'
chatbot = Chatbot(saved_models_directory=directory)

app = Flask(__name__)


@app.route('/chat', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return 'Hi! Its me Chatbot, You can talk to me using the POST request.'

    if request.method == 'POST':
        if not request.json:
            abort(400)
        question = request.json
        reply = chatbot.reply(query=question['text'], genre=question['genre'])
        response = app.response_class(
            response=json.dumps(reply),
            status=200,
            mimetype='application/json'
        )
        return response


app.run('0.0.0.0', port=5000)
