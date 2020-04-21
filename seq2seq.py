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
from data.contractions import contractions_dict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MOVIES_TITLE_HEADERS = ['movieId', 'title', 'year', 'rating', 'votes', 'genres']
MOVIE_LINES_HEADERS = ['lineId', 'characterId', 'movieId', 'characterName', 'text']
MOVE_CONVERSATION_SEQUENCE_HEADERS = ['characterID1', 'characterID2', 'movieId', 'lineIds']
DELIMITER = '+++$+++'
DATA_DIRECTORY = 'data'

LEARNING_RATE = 0.001
EMBEDDINGS_DIMS = 50
TEACHER_FORCING_PROB = 0.5
MAX_TOKEN_LENGTH = 10
MIN_TOKEN_FREQ = 3
HIDDEN_STATE_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GENRE = 'family'
BATCH_SIZE = 32
DROPOUT = 0.1
CLIP = 10
SAVE_CHECK_POINT_STEP = 50
EPOCHS = 20
PRINT_EVERY = 10


class UnrecognizedWordException(Exception):
    """:raise when query text contains texts that are not defined for the chatbot."""
    pass


class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)


class EncoderGRU(nn.Module):
    """A simple decoder with word embeddings"""

    def __init__(self, input_size, hidden_size, embeddings_dims, num_layers=1, dropout=0.0, bidirectional=True,
                 device='cpu'):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device
        self.dropout = 0 if num_layers == 1 else dropout
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embeddings_dims)
        self.gru = nn.GRU(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=self.dropout, bidirectional=bidirectional)

    def forward(self, x):
        self.gru.flatten_parameters()
        # TODO : we might need to unpack here
        x = self._pack_pad_embed(x)
        packed_out, hidden = self.gru(x)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        if self.bidirectional:
            # Sum the results of forward(0) and backward(1) pass together
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
            hidden = hidden[0:1, :, :] + hidden[1:, :, :]
        return output, hidden

    def _pack_pad_embed(self, sequences):
        sequences_length = list(map(len, sequences))
        padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True).to(self.device)
        embedded_sequences = self.embedding(padded_sequences)
        return nn.utils.rnn.pack_padded_sequence(embedded_sequences, sequences_length, batch_first=True,
                                                 enforce_sorted=False).to(self.device)

    def init_hidden(self, batch_size):
        num_layers = self.num_layers
        if self.bidirectional:
            num_layers = self.num_layers * 2
        return torch.zeros(num_layers, batch_size, self.hidden_size)


class DecoderGRU(nn.Module):
    """A simple decoder with embedding, and a linear layer to project the output of
       the layer GRU to a vocabulary size dimension:  hidden_size -> vocab_size
    """

    def __init__(self, input_size, hidden_size, embeddings_dims, vocab_size, num_layers=1, dropout=0.0):
        super(DecoderGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embeddings_dims)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=(0 if num_layers == 1 else dropout))
        self.Watt = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        # not sure why its 2*hidden size in pytorch
        # self.lineear_combined_outputs = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)
        self.word_mapper = nn.Linear(in_features=2 * hidden_size, out_features=vocab_size)

    # My own implenetation
    def forward(self, decoder_input, previous_decoder_hidden, encoder_outputs=None):
        self.gru.flatten_parameters()
        emb = self.embedding(decoder_input)
        emb_drop = self.embedding_dropout(emb)
        decoder_out, decoder_hidden_states = self.gru(emb_drop, previous_decoder_hidden)

        attention_weights = self._attention_weights(decoder_out, encoder_outputs)
        # output, att_weights = self._attention(decoder_out, encoder_outputs)

        # a.ak.a context vector
        # We want to compress the outputs of all seqeunces into a single sequence
        # (1,1,6)*(1,6,512) ==> (1,1,512)
        # the result of wightxendocderoutputs is simillar to decoder out put
        # multiply decoder outputs by  encoder out and collapse all of them into a single vector
        # the resulting vector has the same shape as the output of decoder ( since both enc and dec have the same architecture)
        context = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs)

        # combines two output each with hidden_size length --> 2*hidden_size
        out_combined = torch.cat((context, decoder_out), -1)
        # Shrinks from 2*hidden_size --> hidden_size  (
        # combined_size_adjusted = self.lineear_combined_outputs(combined_encoder_decoder_outputs)
        # compress output ot -1,1 range
        # decoder_out = torch.tanh(combined_size_adjusted)
        vocab_logits = self.word_mapper(out_combined)
        # we jusr teturn the att_weights for visiualization
        return vocab_logits, decoder_hidden_states

    def _attention_weights(self, decoder_output, encoder_outputs):
        # We use general score formula
        out = self.Watt(decoder_output)
        # Add outputs of all neuruns for a single word togheter, to have just a single output for a word (each word in encoder)
        # [ [n1,n2] ,[n1,n2],[n1,n2 ] *[ [n1,n2] ] --sum -- > [ [n1'] ,[n1'] ,[n3'] ]
        # raw_scaled_encoder_outputs = torch.sum(encoder_outputs * decoder_output, dim=2)
        # out2 = out.view(1, -1, 1)
        # combined = encoder_outputs.bmm(out2)
        # final = combined.squeeze(-1)
        return encoder_outputs.bmm(out.view(1, -1, 1)).squeeze(-1)


class Tokenizer:
    """Converts Text into its numerical representation"""

    def __init__(self, contractions_dict=None, max_sequence_length=20, min_token_frequency=10):
        self.START_TOKEN = "<sos>"
        self.PADDING_TOKEN = "<pad>"
        self.END_TOKEN = "<eos>"
        self.UNKNOWN_TOKEN = "<unk>"
        self.contractions_dict = contractions_dict
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
                if len(tokenized_source) > self.max_length or len(tokenized_target) > self.max_length:
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

    def texts_to_index(self, sentences, raise_unknown=False):
        """Convert words in sentences to their numerical index values"""
        indexes = []
        end_token_index = self.word2index[self.END_TOKEN]
        unknown_token_index = self.word2index[self.UNKNOWN_TOKEN]
        print("Received text sequences:", len(sentences))
        for sentence_text in sentences:
            sentence_text = self._sanitize(sentence_text)
            sentence_index = []
            for word in sentence_text.strip().lower().split(" "):
                try:
                    sentence_index.append(
                        self.word2index[word])
                except KeyError:
                    if raise_unknown:
                        raise UnrecognizedWordException(f'Unrecognized Word:{word}')
                    else:
                        sentence_index.append(unknown_token_index)

            sentence_index.append(end_token_index)
            indexes.append(sentence_index)

        print("Valid text sequences:", len(indexes))
        return indexes

    def indexes_to_text(self, word_numbers):
        """Converts an array of numbers to a text string"""
        ignore_index = [self.word2index[self.PADDING_TOKEN]]
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
        if self.contractions_dict is not None:
            for contraction, expanded in self.contractions_dict.items():
                text = re.sub(contraction, ' ' + expanded + ' ', text)

        text = re.sub(r"([.]{3})", r' ', text)
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

    def step(self, batched_sources, batched_targets, teacher_forcing_prob):
        """Takes one full encode-decoder step.
           Creates a context vector from the encoder and predicts the full target sequence
           using decoder..
        """

        batched_encoder_outs, batched_encoder_hidden = self.encoder(batched_sources)
        loss, average_blue_score_batch, predicted_indexes_batch = self._decode(batched_targets, batched_encoder_outs,
                                                                               batched_encoder_hidden,
                                                                               teacher_forcing_prob)
        return loss, average_blue_score_batch, predicted_indexes_batch

    def _decode(self, batched_targets, batched_encoder_outs, batched_encoder_hidden_states, teacher_forcing_prob):
        batch_size = len(batched_targets)
        average_belu_score = 0
        predicted_indexes_batch = []
        decoder_prob_predictions = []
        target_indexes_flat = []
        for batch_idx, target_sequence in enumerate(batched_targets):
            # Fo initial step we pass the last decoder hiden state to the encoder
            encoder_hidden = self._extract_encoder_hidden_state(batched_encoder_hidden_states, batch_idx)
            encoder_outputs = self._extract_encoder_outputs(batched_encoder_outs, batch_idx)
            decoder_input = torch.LongTensor([[self.start_token_index]]).to(self.device)
            predicted_indexes = []
            decoder_hidden = encoder_hidden
            for target_sequence_item in target_sequence:
                # decoder_input, decoder_hidden, encoder_output
                # the hidden state here is from previous step not for current input
                # for the first item it comes from the encoder last hidden state.
                decoder_predicted_prob, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                predicted_target = decoder_predicted_prob.argmax(dim=2)
                # teacher forcing
                if np.random.random() < teacher_forcing_prob:
                    # Actual target
                    decoder_input = torch.LongTensor([[target_sequence_item]]).to(self.device)
                else:
                    decoder_input = predicted_target

                predicted_indexes.append(predicted_target.item())
                decoder_prob_predictions.append(decoder_predicted_prob.squeeze(0))

            target_indexes_flat.extend(target_sequence.detach())

            average_belu_score += self.belu_score(predicted_indexes, target_sequence.cpu().data.numpy())
            predicted_indexes_batch.append(predicted_indexes)

        decoder_outputs_t = torch.cat(decoder_prob_predictions).to(self.device)
        target_indexes_flat_t = torch.LongTensor(target_indexes_flat).to(self.device)
        loss = F.cross_entropy(decoder_outputs_t, target_indexes_flat_t, ignore_index=self.pad_token_index)
        average_belu_score /= batch_size
        return loss, average_belu_score, predicted_indexes_batch

    def _extract_encoder_hidden_state(self, hidden_states, batch_idx):
        # tehse are only the last hidden state for each item in the batch
        return hidden_states[:, batch_idx:batch_idx + 1, :].contiguous()

    def _extract_encoder_outputs(self, outputs, batch_idx):
        # (batch ,sequence,oututs)
        # e.g source sequence [1,2,3,4] --> [ [],[] ,[],[]]
        # we have a decoder output for each item in the source sequence
        return outputs[batch_idx:batch_idx + 1, :, :].contiguous()

    def belu_score(self, predicted_seq, reference_sequences):
        sf = bleu_score.SmoothingFunction()
        reference_sequences = np.expand_dims(reference_sequences, axis=0)
        return bleu_score.sentence_bleu(reference_sequences, predicted_seq,
                                        smoothing_function=sf.method1,
                                        weights=(0.5, 0.5))

    def predict_response(self, question_text, max_len=10, mode='argmax'):
        try:
            question_indexes = self.tokenizer.texts_to_index(sentences=[question_text], raise_unknown=True)
        except UnrecognizedWordException as err:
            unknown_word = err.args[0].split(':')[1]
            return f"Sorry! I don't understand the word: {unknown_word}"

        question_indexes = [torch.LongTensor(question).to(self.device) for question in question_indexes]
        predicted_response_indexes = self._decode_prediction_response(question_indexes, max_len, mode)

        return self.tokenizer.indexes_to_text(predicted_response_indexes)

    def _decode_prediction_response(self, sources, max_response_length=10, mode='max'):
        encoder_outputs_batched, encoder_hidden_batched = self.encoder(sources)
        response_indexes = []
        for decode_step, source in enumerate(sources):
            encoder_hidden = self._extract_encoder_hidden_state(encoder_hidden_batched, decode_step)
            encoder_outputs = self._extract_encoder_outputs(encoder_outputs_batched, decode_step)
            decoder_input = torch.LongTensor([[self.start_token_index]]).to(self.device)
            decoder_hidden = encoder_hidden
            for _ in range(max_response_length):
                decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                predicted_index = decoder_out.argmax(dim=2).cpu().item()
                response_indexes.append(predicted_index)
                if predicted_index == self.end_token_index:
                    break
        return response_indexes


class TrainingSession:
    """A container class that runs the training job"""

    def __init__(self, encoder, decoder, encoder_decoder, tokenizer, device, learning_rate,
                 teacher_forcing_prob, print_every):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_decoder = encoder_decoder
        self.tokenizer = tokenizer
        self.device = device
        self.print_every = print_every
        self.learning_rate = learning_rate
        self.teacher_forcing_prob = teacher_forcing_prob
        self.start_token_index = self.tokenizer.start_token_index
        self.pad_token_index = self.tokenizer.word2index[self.tokenizer.PADDING_TOKEN]
        self.end_token_index = self.tokenizer.word2index[self.tokenizer.END_TOKEN]
        self.writer = SummaryWriter(comment='-' + datetime.now().isoformat(timespec='seconds'))

    def train_evaluate(self, train_source, train_targets, teacher_forcing_prob=0.5, batch_size=10, epochs=20):
        pass

    def train(self, train_sources, train_targets, teacher_forcing_prob=0.5, batch_size=10, epochs=20,
              check_point_step=200):
        if isinstance(train_sources, list):
            train_sources = np.array(train_sources)
            train_targets = np.array(train_targets)

        encoder_optimizer = torch.optim.Adam(self.encoder_decoder.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = torch.optim.Adam(self.encoder_decoder.decoder.parameters(), lr=self.learning_rate)
        print('Received training pairs with sizes :', len(train_sources), len(train_targets))
        total_batch_steps = 0
        for epoch in range(epochs):
            batch_step = 0
            for sources, targets in self.batch_generator(train_sources, train_targets, batch_size):
                batch_step += 1
                total_batch_steps += 1
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss, bleu_score_average, predicted_indexes_batch = self.encoder_decoder.step(sources, targets,
                                                                                              teacher_forcing_prob=teacher_forcing_prob)
                loss.backward()
                _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), CLIP)
                _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), CLIP)
                encoder_optimizer.step()
                decoder_optimizer.step()

                if total_batch_steps % self.print_every == 0:
                    self._show_prediction_text(predicted_indexes_batch, targets, total_batch_steps)
                    print(
                        f'Epoch: {epoch}, Total batch:{total_batch_steps}, Batch:{batch_step},Batch size: {len(sources)},  Loss: {loss.item()}, Belu:{bleu_score_average:.5f}')
                    self.writer.add_scalar('loss:', loss.item(), total_batch_steps)
                    self.writer.add_scalar('belu:', bleu_score_average, total_batch_steps)

                self.save_check_point(total_batch_steps, check_point_step)

    def batch_generator(self, sources, targets, batch_size, shuffle=False, drop_last=True):
        """Creates tensor batches from list of sequences.
           If the source is not exactly dividable by the batch size,
           the last batch would be smaller than the rest and might create a bumpy loss trend.
           drop_last =True will drip that smaller batch.
           We shuffle the samples per epoch
        """
        if shuffle:
            # We shuffle the data before each epoch
            # sources and targets have the same size
            random_idx = np.random.choice(len(sources), len(sources), replace=False)
            sources = sources[random_idx]
            targets = targets[random_idx]

        last_index = len(sources)
        if drop_last:
            last_index -= last_index % batch_size

        for i in range(0, last_index, batch_size):
            yield self._batch(sources, i, batch_size), self._batch(targets, i, batch_size)

    def _batch(self, source, current_index, batch_size, shuffle=True):
        """Receives a list of sequences and and returns a batch of tensors"""
        batch = source[current_index:current_index + batch_size]
        return [torch.LongTensor(sequence).to(self.device) for sequence in batch]

    @torch.no_grad()
    def _show_prediction_text(self, predicted_indexes_batch, targets, step):
        random_index = np.random.randint(0, len(predicted_indexes_batch))
        target_text = self.tokenizer.indexes_to_text(targets[random_index].cpu().data.numpy())
        prediction_text = self.tokenizer.indexes_to_text(predicted_indexes_batch[random_index])
        if step % 20 == 0:
            print(target_text)
            print(prediction_text)
            print('========================================')
            question = 'Hi'
            response = self.encoder_decoder.predict_response(question_text=question, max_len=10)
            print('Question:', question)
            print('Response:', response)

    def save_check_point(self, total_batch_steps, check_point_step):
        if total_batch_steps % check_point_step == 0:
            file_name = f'encoder-decoder_{total_batch_steps}.joblib'
            joblib.dump(self.encoder_decoder, file_name)
            self.writer.close()
            print(f'Checkpoint saved: -> {file_name}')


if __name__ == '__main__':
    tokenizer = Tokenizer(contractions_dict=contractions_dict, min_token_frequency=MIN_TOKEN_FREQ,
                          max_sequence_length=MAX_TOKEN_LENGTH)
    parser = MetaDataParser(data_directory=DATA_DIRECTORY, delimiter=DELIMITER,
                            movie_titles_headers=MOVIES_TITLE_HEADERS,
                            movie_lines_headers=MOVIE_LINES_HEADERS,
                            movie_conversation_headers=MOVE_CONVERSATION_SEQUENCE_HEADERS)
    # TODO: after testing
    parser.load_data()
    # samples = parser.show_sample_dialog(genre='comedy')
    # print(samples)
    conversation_pairs = parser.get_conversation_pairs(genre=GENRE, randomize=True)
    # conversation_pairs = [
    #     ['Hi how are you?', 'I am good'],
    #     ['How was your day?', 'It was a fantastic day'],
    #     ['Good morning!', 'Good morning to you too'],
    #     ['How everything is going', 'Things are going great'],
    # ]
    # EPOCHS = 100
    # MIN_TOKEN_FREQ = 0
    sources_conversation, targets_replies = zip(*conversation_pairs)
    print('Total number of data pars:', len(sources_conversation), 'Total replies:', len(targets_replies))
    tokenizer.fit_on_text(sources_conversation + targets_replies)
    print('Dictionary size:', tokenizer.dictionary_size)

    sources_conversation, targets_replies = tokenizer.text_to_index_paris(conversation_pairs)
    # Encoder is a bi-directional RNN
    encoder = EncoderGRU(input_size=tokenizer.dictionary_size, hidden_size=HIDDEN_STATE_SIZE,
                         embeddings_dims=EMBEDDINGS_DIMS, dropout=DROPOUT, bidirectional=True, device=DEVICE).to(DEVICE)
    # Note Decoder is a uni-directional RNN
    decoder = DecoderGRU(input_size=tokenizer.dictionary_size, hidden_size=HIDDEN_STATE_SIZE,
                         embeddings_dims=EMBEDDINGS_DIMS,
                         vocab_size=tokenizer.dictionary_size,
                         dropout=DROPOUT).to(DEVICE)
    encoder_decoder = EncoderDecoder(encoder, decoder, tokenizer=tokenizer, device=DEVICE)

    trainer = TrainingSession(encoder=encoder, decoder=decoder, encoder_decoder=encoder_decoder, tokenizer=tokenizer,
                              learning_rate=LEARNING_RATE,
                              teacher_forcing_prob=TEACHER_FORCING_PROB,
                              print_every=PRINT_EVERY,
                              device=DEVICE)

    trainer.train(sources_conversation, targets_replies, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  check_point_step=SAVE_CHECK_POINT_STEP)
