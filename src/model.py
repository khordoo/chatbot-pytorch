import torch
import joblib
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk.translate import bleu_score
from src.exceptions import UnrecognizedWordException


class EncoderGRU(nn.Module):
    """Encoder class with word embeddings
     and bidirectional GRU layers"""

    def __init__(self, input_size, hidden_size, embeddings_dims, num_layers=1, dropout=0.0, bidirectional=True,
                 device='cpu'):
        super(EncoderGRU, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = 0 if num_layers == 1 else dropout
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embeddings_dims)
        self.gru = nn.GRU(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=self.dropout, bidirectional=bidirectional)

    def forward(self, x):
        self.gru.flatten_parameters()
        x = self._pack_pad_embed(x)
        packed_out, hidden = self.gru(x)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        if self.bidirectional:
            # Sum the outputs of forward and backward pass together
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
    """A attention based decoder with embedding, and a linear layer to project the output of
       the layer GRU to a vocabulary size dimension:  hidden_size -> vocab_size
       Attention is calculated based on the Luong approach.
    """

    def __init__(self, input_size, hidden_size, embeddings_dims, vocab_size, num_layers=1, dropout=0.0):
        super(DecoderGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embeddings_dims)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=(0 if num_layers == 1 else dropout))
        self.att_scoring = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.word_mapper = nn.Linear(in_features=2 * hidden_size, out_features=vocab_size)

    def forward(self, decoder_input, previous_decoder_hidden, encoder_outputs=None):
        # Fixes contiguous memory warning
        self.gru.flatten_parameters()

        emb = self.embedding(decoder_input)
        emb_drop = self.embedding_dropout(emb)
        decoder_out, decoder_hidden_states = self.gru(emb_drop, previous_decoder_hidden)
        # attention_weights shape is similar to the decoder output
        attention_weights = self._attention_weights(decoder_out, encoder_outputs)

        context = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs)

        # since context has the same shape as the encoder output:
        # hidden_size+hidden_size --> 2*hidden_size
        out_combined = torch.cat((context, decoder_out), -1)

        # project: 2*hidden_size --> vocab_size
        vocab_logits = self.word_mapper(out_combined)
        return vocab_logits, decoder_hidden_states

    def _attention_weights(self, decoder_output, encoder_outputs):
        # We use general scoring formula
        out = self.att_scoring(decoder_output)
        return encoder_outputs.bmm(out.view(1, -1, 1)).squeeze(-1)


class EncoderDecoderMediator:
    """A simple wrapper class to simplify the encoder-decoder interactions"""

    def __init__(self, encoder, decoder, tokenizer, device):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device
        self.sos_index = self.tokenizer.sos_index
        self.pad_token_index = 0
        self.end_token_index = self.tokenizer.word2index[self.tokenizer.END_TOKEN]

    def step(self, batched_sources, batched_targets, teacher_forcing_ratio):
        """Receives a batch of source and target items and Takes one encode-decoder step.
           For each item in the batch, passes the source to encoder to get the required states and then
           sequentially makes predictions for every index in the target.
        """

        batched_encoder_outs, batched_encoder_hidden = self.encoder(batched_sources)
        loss, batch_mean_bleu, batch_predicted_indexes = self._decode(batched_targets, batched_encoder_outs,
                                                                               batched_encoder_hidden,
                                                                               teacher_forcing_ratio)
        return loss, batch_mean_bleu, batch_predicted_indexes

    def _decode(self, batched_targets, batched_encoder_outs, batched_encoder_hidden_states, teacher_forcing_prob):
        """Receives the encoders hidden states and outputs for all the source sequences in a batch form, and then
           sequentially performs the decoding for each target sequence.
        """
        batch_size = len(batched_targets)
        mean_bleu = 0
        batch_predicted_indexes = []
        batch_decoder_out = []
        batch_target_indexes = []
        for batch_idx, targets in enumerate(batched_targets):
            # For initial step we pass the decoder's last hidden state to the encoder
            predicted_indexes = []
            decoder_input = torch.LongTensor([[self.sos_index]]).to(self.device)
            encoder_outputs = self._extract_encoder_outputs(batched_encoder_outs, batch_idx)
            encoder_hidden = self._extract_encoder_hidden_state(batched_encoder_hidden_states, batch_idx)

            decoder_hidden = encoder_hidden
            for target_index in targets:
                decoder_prob_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                predicted_target = decoder_prob_out.argmax(dim=2)
                # teacher forcing
                if np.random.random() < teacher_forcing_prob:
                    # actual target
                    decoder_input = torch.LongTensor([[target_index]]).to(self.device)
                else:
                    decoder_input = predicted_target

                predicted_indexes.append(predicted_target.item())
                batch_decoder_out.append(decoder_prob_out.squeeze(0))

            batch_target_indexes.extend(targets.detach())

            mean_bleu += self._belu_score(predicted_indexes, targets.cpu().data.numpy())
            batch_predicted_indexes.append(predicted_indexes)

        batch_decoder_outputs_t = torch.cat(batch_decoder_out).to(self.device)
        batched_target_indexes_t = torch.LongTensor(batch_target_indexes).to(self.device)
        loss = F.cross_entropy(batch_decoder_outputs_t, batched_target_indexes_t, ignore_index=self.pad_token_index)
        mean_bleu /= batch_size
        return loss, mean_bleu, batch_predicted_indexes

    def _extract_encoder_hidden_state(self, hidden_states, batch_idx):
        """Extracts the hidden states for an item, given its index in the batch"""
        return hidden_states[:, batch_idx:batch_idx + 1, :].contiguous()

    def _extract_encoder_outputs(self, outputs, batch_idx):
        """Extract the output for an item, given its index in the batch"""
        # (batch,sequence,outputs)
        return outputs[batch_idx:batch_idx + 1, :, :].contiguous()

    def _belu_score(self, predicted_seq, reference_sequences):
        smoothing_fn = bleu_score.SmoothingFunction()
        reference_sequences = np.expand_dims(reference_sequences, axis=0)
        return bleu_score.sentence_bleu(reference_sequences, predicted_seq,
                                        smoothing_function=smoothing_fn.method1,
                                        weights=(0.5, 0.5))

    def predict_response(self, question_text, max_len=10, mode='argmax'):
        """This is used during the inference.
           It receives a question text as an input and returns
           the predicted indexes in a text from back."""
        try:
            question_indexes = self.tokenizer.text_to_index(sentences=[question_text], raise_unknown=True)
        except UnrecognizedWordException as err:
            unknown_word = err.args[0].split(':')[1]
            return f"Sorry! I don't understand the word: {unknown_word}"

        question_indexes = [torch.LongTensor(question).to(self.device) for question in question_indexes]
        predicted_response_indexes = self._decode_prediction_response(question_indexes, max_len, mode)

        return self.tokenizer.index_to_text(predicted_response_indexes)

    def _decode_prediction_response(self, sources, max_response_length=10, mode='max'):
        """Performs the prediction for inference"""
        encoder_outputs_batched, encoder_hidden_batched = self.encoder(sources)
        response_indexes = []
        for decode_step, source in enumerate(sources):
            encoder_hidden = self._extract_encoder_hidden_state(encoder_hidden_batched, decode_step)
            encoder_outputs = self._extract_encoder_outputs(encoder_outputs_batched, decode_step)
            decoder_input = torch.LongTensor([[self.sos_index]]).to(self.device)
            decoder_hidden = encoder_hidden
            for _ in range(max_response_length):
                decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                predicted_index = decoder_out.argmax(dim=2).cpu().item()
                response_indexes.append(predicted_index)
                if predicted_index == self.end_token_index:
                    break
        return response_indexes

    def save_states(self, step='0'):
        """Save the encoder,decoder and tokenizer to the file"""
        torch.save(self.encoder.state_dict(), 'encoder-{}.dat'.format(step))
        torch.save(self.decoder.state_dict(), 'decoder-{}.dat'.format(step))
        joblib.dump(self, 'encoder-decoder-{}.joblib'.format(step))

    def load(self, encoder_state_path, decoder_state_path, map_location='cpu'):
        """Loads the encoder and decoder state dictionaries"""
        self.encoder.load_state_dict(torch.load(encoder_state_path, map_location=map_location))
        self.decoder.load_state_dict(torch.load(decoder_state_path, map_location=map_location))
