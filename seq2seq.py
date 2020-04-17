import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from datetime import datetime

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.01
LSTM_HIDDEN_SIZE = 100
EMBEDDINGS_DIMS = 50
TEACHER_FORCING_PROB = 0.5
MAX_SEQUENCE_LENGTH = 10

import torch
import torch.nn.functional as F
from src.movie_parser import MetaDataParser
import numpy as np
from nltk.translate import bleu_score

MOVIES_TITLE_HEADERS = ['movieId', 'title', 'year', 'rating', 'votes', 'genres']
MOVIE_LINES_HEADERS = ['lineId', 'characterId', 'movieId', 'characterName', 'text']
MOVE_CONVERSATION_SEQUENCE_HEADERS = ['characterID1', 'characterID2', 'movieId', 'lineIds']
DELIMITER = '+++$+++'
DATA_DIRECTORY = 'data'
MAX_TOKEN_LENGTH = 20
MIN_TOKEN_FREQ = 10
HIDDEN_STATE_SIZE = 512
EMBEDDING_DIMS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GENRE = 'family'
BATCH_SIZE = 32
EPOCHS = 100


class EncoderLSTM(nn.Module):
    """A simple decoder with word embeddings"""

    def __init__(self, input_size, hidden_size, embeddings_dims):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(num_embeddings=input_size, embedding_dim=embeddings_dims)
        self.lstm = nn.LSTM(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, x, hidden_states):
        x = self.emb(x)
        out, hidden_states = self.lstm(x, hidden_states)
        return out, hidden_states


class DecoderLSTM(nn.Module):
    """A simple decoder with embedding, and a linear layer to project the output of
       the layer LSTM to a vocabulary size dimension:  hidden_size -> vocab_size
    """

    def __init__(self, input_size, hidden_size, embeddings_dims, vocab_size):
        super(DecoderLSTM, self).__init__()
        self.emb = nn.Embedding(num_embeddings=input_size, embedding_dim=embeddings_dims)
        self.lstm = nn.LSTM(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden_states):
        x = self.emb(x)
        out, hidden_states = self.lstm(x, hidden_states)
        out = self.linear(out)
        return out, hidden_states


class Tokenizer:
    """Converts Text into its numerical representation"""

    def __init__(self, max_sequence_length):
        self.START_TOKEN = '<sos>'
        self.PADDING_TOKEN = '<pad>'
        self.END_TOKEN = '<eos>'
        self.word2index = {self.PADDING_TOKEN: 0, self.START_TOKEN: 1, self.END_TOKEN: 2}
        self.index2word = {0: self.PADDING_TOKEN, 1: self.START_TOKEN, 2: self.END_TOKEN}
        self.words_count = len(self.word2index)
        self.max_length = max_sequence_length

    def fit_on_text(self, text_array):
        """Creates a numerical index value for every unique word"""
        for sentence in text_array:
            self._add_sentence(sentence)

    def _add_sentence(self, sentence):
        """Creates indexes for unique word in the sentences and
        adds them to the dictionary"""
        for word in sentence.strip().lower().split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.words_count
                self.index2word[self.words_count] = word
                self.words_count += 1

    def normalize(self, text):
        pass

    def texts_to_index(self, sentences):
        """Convert words in sentences to their numerical index values"""
        sentences_index = []
        end_token_index = self.word2index[self.END_TOKEN]
        for sentence in sentences:
            sentence_index = []
            for word in sentence.strip().lower().split(' '):
                sentence_index.append(self.word2index[word])
            sentence_index.append(end_token_index)
            sentence_index = self._pad(sentence_index)
            sentence_index = self._clip(sentence_index)
            sentences_index.append(sentence_index)
        return sentences_index

    def _clip(self, sequence):
        return sequence[:self.max_length]

    def _pad(self, sequence):
        pad_index = self.word2index[self.PADDING_TOKEN]
        while len(sequence) < self.max_length:
            sequence.append(pad_index)
        return sequence

    def indexes_to_text(self, word_numbers):
        """Converts an array of numbers to a text string"""
        ignore_index = [self.word2index[self.PADDING_TOKEN],
                        self.word2index[self.END_TOKEN]
                        ]
        return ' '.join([self.index2word[idx] for idx in word_numbers if idx not in ignore_index])

    @property
    def start_token_index(self):
        return self.word2index[self.START_TOKEN]


class TrainingSession:
    """A container class that runs the training job"""

    def __init__(self, encoder, decoder, tokenizer, device, learning_rate, teacher_forcing_prob):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.teacher_forcing_prob = teacher_forcing_prob
        self.start_token_index = self.tokenizer.start_token_index
        self.pad_token_index = self.tokenizer.word2index[self.tokenizer.PADDING_TOKEN]
        self.end_token_index = self.tokenizer.word2index[self.tokenizer.END_TOKEN]
        self.writer = SummaryWriter(comment='-' + datetime.now().isoformat(timespec='seconds'))

    def train_evaluate(self, train_source, train_targets, teacher_forcing_prob=0.5, batch_size=10, epochs=20):
        pass

    def train(self, train_source, train_targets, teacher_forcing_prob=0.5, batch_size=10, epochs=20):
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        steps = 0
        for epoch in range(epochs):
            for batch_idx, (sources, targes) in enumerate(
                    self.batch_generator(train_source, train_targets, batch_size)):
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                encoder_out, encoder_hidden = self.encode_batch(sources)
                loss = 0
                steps += 1
                belu = 0
                for idx, (source, target) in enumerate(zip(sources, targets)):
                    ## Extracting hidden states(h,c) for the current item in the batch
                    decoder_hidden = [encoder_hidden[0][:, idx:idx + 1].contiguous(),
                                      encoder_hidden[1][:, idx: idx + 1].contiguous()]
                    decoder_input = torch.LongTensor([[self.start_token_index]]).to(self.device)
                    predicted_indexes = []
                    for target_idx in target:
                        if target_idx == self.end_token_index:
                            break
                        decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        predicted_target = decoder_out.argmax(dim=2)
                        actual_target = torch.LongTensor([[target_idx]]).to(self.device)
                        # teacher forcing
                        if np.random.random() < teacher_forcing_prob:
                            decoder_input = actual_target
                        else:
                            decoder_input = predicted_target
                        loss += F.cross_entropy(decoder_out.squeeze(0), actual_target.flatten(),
                                                ignore_index=self.pad_token_index)
                        predicted_indexes.append(predicted_target.item())

                        if steps % 50 == 0 and len(predicted_indexes) == len(target):
                            print(f'T{idx}', self.tokenizer.indexes_to_text(target))
                            print(f'P{idx}:', self.tokenizer.indexes_to_text(predicted_indexes))
                            print('----------------------------------')
                    belu += self.belu_score(predicted_indexes, target)
                belu /= batch_size

                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                print(f'\rEpoch: {epoch}, Batch:{batch_idx}, Loss: {loss.item()}, Belu:{belu:.5f}')
                self.writer.add_scalar('loss:', loss.item(), steps)
                self.writer.add_scalar('belu:', belu, steps)

        torch.save(self.decoder.state_dict(), 'decode-model.dat')
        self.writer.close()
        print('Decoder model succesfuly saved!')

    def encode_batch(self, sources):
        encoder_hidden = (torch.zeros(1, len(sources), self.encoder.hidden_size).to(self.device),
                          torch.zeros(1, len(sources), self.encoder.hidden_size).to(self.device))

        encoder_input = torch.LongTensor(sources).to(self.device)
        return self.encoder(encoder_input, encoder_hidden)

    def batch_generator(self, sources, targets, batch_size):
        for i in range(0, len(sources), batch_size):
            yield sources[i:i + batch_size], targets[i:i + batch_size]

    def belu_score(self, predicted_seq, reference_sequences):
        sf = bleu_score.SmoothingFunction()
        reference_sequences = np.expand_dims(reference_sequences, axis=0)
        return bleu_score.sentence_bleu(reference_sequences, predicted_seq,
                                        smoothing_function=sf.method1,
                                        weights=(0.5, 0.5))

    def validate(self, val_source, val_target):
        encoder_out, encoder_hidden = self.encode_batch(val_source)
        for source, target1 in zip(val_source, val_target):
            pass


tokenizer = Tokenizer(max_sequence_length=MAX_TOKEN_LENGTH)
parser = MetaDataParser(data_directory=DATA_DIRECTORY, delimiter=DELIMITER, movie_titles_headers=MOVIES_TITLE_HEADERS,
                        movie_lines_headers=MOVIE_LINES_HEADERS,
                        movie_conversation_headers=MOVE_CONVERSATION_SEQUENCE_HEADERS)

parser.load_data()
conversation_pair = parser.get_conversation_pairs(genre=GENRE)

sources, targets = zip(*conversation_pair)
print('Total number of data pars:', len(sources))
tokenizer.fit_on_text(sources + targets)
sources = tokenizer.texts_to_index(sources)

targets = tokenizer.texts_to_index(targets)
encoder = EncoderLSTM(input_size=tokenizer.words_count, hidden_size=HIDDEN_STATE_SIZE,
                      embeddings_dims=EMBEDDINGS_DIMS).to(
    DEVICE)
decoder = DecoderLSTM(input_size=tokenizer.words_count, hidden_size=HIDDEN_STATE_SIZE, embeddings_dims=EMBEDDINGS_DIMS,
                      vocab_size=tokenizer.words_count).to(DEVICE)
trainer = TrainingSession(encoder=encoder, decoder=decoder, tokenizer=tokenizer, learning_rate=LEARNING_RATE,
                          teacher_forcing_prob=TEACHER_FORCING_PROB,
                          device=DEVICE)
trainer.train(sources, targets, batch_size=BATCH_SIZE, epochs=EPOCHS)
