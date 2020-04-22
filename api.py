import os
import json
from flask import Flask, request, abort
from flask_cors import CORS
from api.chatbot import Chatbot

BASE_DIRECTORY = os.path.dirname(__file__)

app = Flask(__name__)
CORS(app)

chatbot = Chatbot(saved_models_directory=BASE_DIRECTORY)


@app.route('/chat', methods=['POST', 'GET'])
def index():
    """Flask endpoint that receives the use text in the json format
      and returns with predicted response from the bot."""

    if request.method == 'GET':
        return "Hi, It's me the Chatbot, You can talk to me using the POST request!"

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


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)
