import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence


class AttentionEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, bidirectional=False, device='cpu',
                 teacher_forcing_ration=0.5):
        super(AttentionEncoderDecoder, self).__init__()
        self.device = device
        self.teacher_forcing_ration = teacher_forcing_ration
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True,
                               bidirectional=bidirectional)
        self.attention_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.attention_combine = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.word_projection = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def encode(self, batch):
        """Performs the encoding for a batch of source sequences.
           it then  return a batch containing the last hidden states
           for each sequence in the batch.
        """
        packed_outs, hidden_states = self.encoder(self.pack_input(batch))
        unpacked_outs, _ = pad_packed_sequence(packed_outs, batch_first=True)
        return unpacked_outs, hidden_states

    def pack_input(self, x):
        sequence_lengths = list(map(len, x))
        padded = pad_sequence(x, batch_first=True)
        embedded = self.embedding(padded)
        return pack_padded_sequence(embedded, sequence_lengths, batch_first=True,
                                    enforce_sorted=False)

    def decode(self, decoder_input, decoder_hidden, encoder_outs):
        """Performs the attention based decoding for a single target value.
           The method assumes a batch size of one.
        """
        embedded = self.embedding(decoder_input)
        decoder_out, decoder_hidden = self.decoder(embedded, decoder_hidden)
        attention_weights = self.attention_weights(decoder_out, encoder_outs)
        # Collapses all the encoder outputs into a single output
        # to be combined with the decoder output
        attended_encoder_outs = torch.bmm(attention_weights, encoder_outs)
        # now that they both have the same shape, combine encode and decoder outs together
        combined = torch.cat([attended_encoder_outs, decoder_out], dim=2)
        # shrink the combined size back to the hidden size
        combined_outs = self.attention_combine(combined)
        # project the hidden size to our vocab size
        projected_out = self.word_projection(combined_outs)
        return projected_out, decoder_hidden

    def attention_weights(self, decoder_out, encoder_outs):
        """Calculates the attention weights"""
        scores = self.attention_linear(decoder_out)
        unnormalized_weights = torch.bmm(scores, encoder_outs.transpose(1, 2))
        return F.softmax(unnormalized_weights, dim=2)
