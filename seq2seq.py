import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
from src.movie_parser import MetaDataParser
import numpy as np
from nltk.translate import bleu_score
import re
import collections
import joblib

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MOVIES_TITLE_HEADERS = ['movieId', 'title', 'year', 'rating', 'votes', 'genres']
MOVIE_LINES_HEADERS = ['lineId', 'characterId', 'movieId', 'characterName', 'text']
MOVE_CONVERSATION_SEQUENCE_HEADERS = ['characterID1', 'characterID2', 'movieId', 'lineIds']
DELIMITER = '+++$+++'
DATA_DIRECTORY = 'data'

LEARNING_RATE = 0.01
EMBEDDINGS_DIMS = 50
TEACHER_FORCING_PROB = 0.5
MAX_TOKEN_LENGTH = 20
MIN_TOKEN_FREQ = 10
HIDDEN_STATE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GENRE = 'family'
BATCH_SIZE = 32
EPOCHS = 100
CLIP = 10


class EncoderLSTM(nn.Module):
    """A simple decoder with word embeddings"""

    def __init__(self, input_size, hidden_size, embeddings_dims, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embeddings_dims)
        self.lstm = nn.LSTM(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)

    def forward(self, x):
        return self.lstm(x)


class DecoderLSTM(nn.Module):
    """A simple decoder with embedding, and a linear layer to project the output of
       the layer LSTM to a vocabulary size dimension:  hidden_size -> vocab_size
    """

    def __init__(self, input_size, hidden_size, embeddings_dims, vocab_size, num_layers=1):
        super(DecoderLSTM, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embeddings_dims)
        self.lstm = nn.LSTM(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden_states):
        x = self.embedding(x)
        out, hidden_states = self.lstm(x, hidden_states)
        out = self.linear(out)
        return out, hidden_states


class Tokenizer:
    """Converts Text into its numerical representation"""

    def __init__(self, max_sequence_length, min_token_frequency=10):
        self.START_TOKEN = "<sos>"
        self.PADDING_TOKEN = "<pad>"
        self.END_TOKEN = "<eos>"
        self.UNKNOWN_TOKEN = "<unk>"
        self.max_length = max_sequence_length
        self.min_token_frequency = min_token_frequency
        self._init_dict()

    def _init_dict(self):
        self.word2index = {}
        self.index2word = {}
        self.word_counter = collections.Counter()
        for token in [self.PADDING_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN]:
            self._add_word(token)

    def text_to_index_paris(self, conversation_pairs):
        """We want to ignore both source and reply if an unknown word found in either of them
          So we have to process the together
        """
        sources, targets = zip(*conversation_pairs)
        end_token_index = self.word2index[self.END_TOKEN]
        source_indexes, target_indexes = [], []
        for source_text, target_text in zip(sources, targets):
            try:
                tokenized_source = self._tokenize(source_text)
                tokenized_target = self._tokenize(target_text)
                if len(tokenized_source) > self.max_length and len(tokenized_target) > self.max_length:
                    continue
            except KeyError:
                # Ignore a text pair with unknown words
                continue
            tokenized_source.append(end_token_index)
            tokenized_target.append(end_token_index)
            source_indexes.append(tokenized_source)
            target_indexes.append(tokenized_target)

        print(f"Tokenized sources:{len(source_indexes)} , Tokenized targets:{len(target_indexes)}")
        return source_indexes, target_indexes

    def _tokenize(self, sentence_text):
        sentence_text = self._sanitize(sentence_text)
        return [
            self.word2index[word]
            for word in sentence_text.strip().split(" ")
        ]

    def texts_to_index(self, sentences):
        """Convert words in sentences to their numerical index values"""
        indexes = []
        end_token_index = self.word2index[self.END_TOKEN]
        unknown_token_index = self.word2index[self.UNKNOWN_TOKEN]
        print("Received text sequences:", len(sentences))
        for sentence_text in sentences:
            sentence_text = self._sanitize(sentence_text)
            sentence_index = [
                self.word2index.get(word, unknown_token_index)
                for word in sentence_text.strip().lower().split(" ")
            ]
            if self.is_valid_token(sentence_index):
                sentence_index.append(end_token_index)
                indexes.append(sentence_index)
        print("Valid text sequences:", len(indexes))
        return indexes

    def indexes_to_text(self, word_numbers):
        """Converts an array of numbers to a text string"""
        ignore_index = [self.word2index[self.PADDING_TOKEN],
                        self.word2index[self.END_TOKEN]
                        ]
        return " ".join([self.index2word[idx] for idx in word_numbers if idx not in ignore_index])

    def fit_on_text(self, text_array):
        """Creates a numerical index value for every unique word"""
        print(f"Fitting on {len(text_array)} pairs ")
        for sentence in text_array:
            sentence = self._sanitize(sentence)
            self._add_sentence(sentence)
        print(f"Added {len(self.word2index)} unique words to the dictionary")
        self._reduce_dict_size()

    def _add_sentence(self, sentence):
        """Creates indexes for unique word in the sentences and
        adds them to the dictionary"""
        for word in sentence.strip().lower().split(" "):
            self.word_counter[word] += 1
            if word not in self.word2index:
                self._add_word(word)

    def _add_word(self, word):
        index = len(self.word2index)
        self.word2index[word] = index
        self.index2word[index] = word

    def _sanitize(self, text):
        text = text.lower().strip()
        # text = re.sub(r"([.]{3})", r' ', text)
        text = re.sub(r"([.?!])", r' \1 ', text)
        text = re.sub(r'[^a-zA-Z.?!]+', r' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _reduce_dict_size(self):
        frequent_words = [word for word, count in self.word_counter.items()
                          if count > self.min_token_frequency]
        if frequent_words:
            self._init_dict()
            for word in frequent_words:
                self._add_word(word)
        print(f'Removed non frequent words: Updated words count:{len(self.word2index)} ')

    def is_valid_token(self, sentence):
        """Filter very long text or
         texts with words that are not in our dictionary"""
        unknown_token_index = self.word2index[self.UNKNOWN_TOKEN]
        if unknown_token_index in sentence or len(sentence) > self.max_length:
            return False
        return True

    @property
    def dictionary_size(self):
        # TODO:We might need to remove the count of tokens
        # From the dictionary size
        return len(self.word2index)

    @property
    def start_token_index(self):
        return self.word2index[self.START_TOKEN]


class EncoderDecoder:
    def __init__(self, encoder, decoder, tokenizer, device):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device
        self.start_token_index = self.tokenizer.start_token_index
        self.pad_token_index = 0
        self.end_token_index = self.tokenizer.word2index[self.tokenizer.END_TOKEN]

    def step(self, sources, targets, teacher_forcing_prob):
        """Takes one full encode-decoder step.
           Creates a context vector from the encoder and predicts the full target sequence
           using decoder..
        """
        encoder_out, encoder_hidden = self._encode(sources)
        loss, average_blue_score_batch, predicted_indexes_batch = self._decode(sources, targets, encoder_hidden,
                                                                               teacher_forcing_prob)
        return loss, average_blue_score_batch, predicted_indexes_batch

    def _encode(self, sources):
        packed_padded_batch = self._pack_pad_batch(sources, self.encoder.embedding)
        packed_out, hidden_states = self.encoder(packed_padded_batch)
        return self.unpack_padded(packed_out), hidden_states

    def _pack_pad_batch(self, sequences, embeddings):
        sequences_length = list(map(len, sequences))
        padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True).to(self.device)
        embedded_sequences = embeddings(padded_sequences)
        return nn.utils.rnn.pack_padded_sequence(embedded_sequences, sequences_length, batch_first=True,
                                                 enforce_sorted=False).to(self.device)

    def unpack_padded(self, packed_output):
        unpacked_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return unpacked_output

    def _decode(self, sources, targets, encoder_hidden_states, teacher_forcing_prob):
        batch_size = len(sources)
        average_belu_score = 0
        loss = 0
        predicted_indexes_batch = []
        for decode_step, (source, target) in enumerate(zip(sources, targets)):
            decoder_hidden = self._extract_step_hidden_state(encoder_hidden_states, decode_step)
            decoder_input = torch.LongTensor([[self.start_token_index]]).to(self.device)
            predicted_indexes = []
            # Avoid making predictions for the last index in target (<eos>)
            target = target[:-1]
            for target_idx in target:
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
            average_belu_score += self.belu_score(predicted_indexes, target.cpu().data.numpy())
            predicted_indexes_batch.append(predicted_indexes)

        average_belu_score /= batch_size
        loss /= batch_size
        return loss, average_belu_score, predicted_indexes_batch

    def _extract_step_hidden_state(self, hidden_states, decode_step):
        ## LSTM has two hidden states(h,c)
        #  Get those (h,c)) for the current batch item
        return [hidden_states[0][:, decode_step:decode_step + 1].contiguous(),
                hidden_states[1][:, decode_step: decode_step + 1].contiguous()]

    def belu_score(self, predicted_seq, reference_sequences):
        sf = bleu_score.SmoothingFunction()
        reference_sequences = np.expand_dims(reference_sequences, axis=0)
        return bleu_score.sentence_bleu(reference_sequences, predicted_seq,
                                        smoothing_function=sf.method1,
                                        weights=(0.5, 0.5))


class TrainingSession:
    """A container class that runs the training job"""

    def __init__(self, encoder, decoder, encoder_decoder, tokenizer, device, learning_rate,
                 teacher_forcing_prob):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_decoder = encoder_decoder
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

    def train(self, train_sources, train_targets, teacher_forcing_prob=0.5, batch_size=10, epochs=20):
        encoder_optimizer = torch.optim.Adam(self.encoder_decoder.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = torch.optim.Adam(self.encoder_decoder.decoder.parameters(), lr=self.learning_rate)
        print('Received training pairs with sizes :', len(train_sources), len(train_targets))
        cum_batch_steps = 0
        for epoch in range(epochs):
            batch_step = 0
            for sources, targets in self.batch_generator(train_sources, train_targets, batch_size):
                batch_step += 1
                cum_batch_steps += 1
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss, bleu_score_average, predicted_indexes_batch = self.encoder_decoder.step(sources, targets,
                                                                                              teacher_forcing_prob=teacher_forcing_prob)
                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), CLIP)
                nn.utils.clip_grad_norm_(self.decoder.parameters(), CLIP)
                encoder_optimizer.step()
                decoder_optimizer.step()
                self.show_prediction_text(predicted_indexes_batch,targets)
                print(
                    f'Epoch: {epoch}, Total batch:{cum_batch_steps}, Batch:{batch_step},Batch size: {len(sources)},  Loss: {loss.item()}, Belu:{bleu_score_average:.5f}')
                self.writer.add_scalar('loss:', loss.item(), cum_batch_steps)
                self.writer.add_scalar('belu:', bleu_score_average, cum_batch_steps)


        self.save_models()

    def batch_generator(self, sources, targets, batch_size, drop_last=False):
        """Creates tensor batches from list of sequences.
           If the source is not exactly dividable by the batch size,
           the last batch would be smaller than the rest and might create a bumpy loss trend.
           drop_last =True will drip that smaller batch.
        """
        last_index = len(sources)
        if drop_last:
            last_index -= last_index % batch_size

        for i in range(0, last_index, batch_size):
            yield self._batch(sources, i, batch_size), self._batch(targets, i, batch_size)

    def _batch(self, source, current_index, batch_size):
        """Receives a list of sequences and and returns a batch of tensors"""
        batch = source[current_index:current_index + batch_size]
        return [torch.LongTensor(sequence).to(self.device) for sequence in batch]

    @torch.no_grad()
    def show_prediction_text(self, predicted_indexes_batch, targets):
        random_index = np.random.randint(0, len(predicted_indexes_batch))
        target_text = self.tokenizer.indexes_to_text(targets[random_index].cpu().data.numpy())
        prediction_text = self.tokenizer.indexes_to_text(predicted_indexes_batch[random_index])
        print(target_text)
        print(prediction_text)

    def save_models(self):
        torch.save(self.encoder.state_dict(), 'encoder-model.dat')
        torch.save(self.decoder.state_dict(), 'decoder-model.dat')
        joblib.dump(self.tokenizer, 'tokenizer.joblib')
        self.writer.close()
        print('Decoder model successfully saved!')


tokenizer = Tokenizer(min_token_frequency=MIN_TOKEN_FREQ, max_sequence_length=MAX_TOKEN_LENGTH)
parser = MetaDataParser(data_directory=DATA_DIRECTORY, delimiter=DELIMITER,
                        movie_titles_headers=MOVIES_TITLE_HEADERS,
                        movie_lines_headers=MOVIE_LINES_HEADERS,
                        movie_conversation_headers=MOVE_CONVERSATION_SEQUENCE_HEADERS)

parser.load_data()
conversation_pairs = parser.get_conversation_pairs(genre=GENRE, randomize=True)

sources_conversation, targets_replies = zip(*conversation_pairs)
print('Total number of data pars:', len(sources_conversation), 'Total replies:', len(targets_replies))
tokenizer.fit_on_text(sources_conversation + targets_replies)
print('Dictionary size:', tokenizer.dictionary_size)

sources_conversation, targets_replies = tokenizer.text_to_index_paris(conversation_pairs)
encoder = EncoderLSTM(input_size=tokenizer.dictionary_size, hidden_size=HIDDEN_STATE_SIZE,
                      embeddings_dims=EMBEDDINGS_DIMS).to(DEVICE)
decoder = DecoderLSTM(input_size=tokenizer.dictionary_size, hidden_size=HIDDEN_STATE_SIZE,
                      embeddings_dims=EMBEDDINGS_DIMS,
                      vocab_size=tokenizer.dictionary_size).to(DEVICE)
encoder_decoder = EncoderDecoder(encoder, decoder, tokenizer=tokenizer, device=DEVICE)
trainer = TrainingSession(encoder=encoder, decoder=decoder, encoder_decoder=encoder_decoder, tokenizer=tokenizer,
                          learning_rate=LEARNING_RATE,
                          teacher_forcing_prob=TEACHER_FORCING_PROB,
                          device=DEVICE)
trainer.train(sources_conversation, targets_replies, batch_size=BATCH_SIZE, epochs=EPOCHS)
