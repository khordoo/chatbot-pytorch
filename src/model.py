import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, bidirectional=False, device='cpu',
                 teacher_forcing_ration=0.5):
        super(EncoderDecoder, self).__init__()
        self.device = device
        self.teacher_forcing_ration = teacher_forcing_ration
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True,
                               bidirectional=bidirectional)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)

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

    def decode(self, input, hidden_state):
        """Performs the decoding for a single target value.
           The method assumes a batch size of one.
        """
        embedded = self.embedding(input)
        output, hidden_state = self.decoder(embedded, hidden_state)
        projected_out = self.linear(output)
        return projected_out, hidden_state
