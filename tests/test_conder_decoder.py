import unittest
import torch
from src.model import EncoderDecoder


class TestEncoderDecoder(unittest.TestCase):
    """Testes the EncoderDecoder class"""

    def setUp(self):
        self.source_sequences = [list([4, 5, 6, 7, 8, 2]), list([5, 9, 10, 11, 8, 2]), list([12, 13, 14, 2]),
                                 list([5, 15, 16, 17, 2])]
        self.target_sequences = [list([18, 19, 12, 20, 7, 21, 2]), list([22, 9, 23, 24, 11, 2]),
                                 list([12, 13, 25, 7, 26, 2]), list([27, 6, 17, 28, 2])]

        self.source_sequences = [torch.FloatTensor(source_sequence) for source_sequence in self.source_sequences]
        self.target_sequences = [torch.FloatTensor(target_sequence) for target_sequence in self.target_sequences]
        INPUT_SIZE = 29
        HIDDEN_SIZE = 512
        EMBEDDIN_DIM = 50
        VOCAB_SIZE = 29
        self.encoder_decoder = EncoderDecoder(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                                              embedding_dim=EMBEDDIN_DIM, vocab_size=VOCAB_SIZE, bidirectional=False)

    def test_pad_sequence(self):
        """tests the sequences are padded"""
        inputs = [torch.FloatTensor([1, 2, 3]), torch.FloatTensor(1, 2)]
        padded = self.encoder_decoder.pack(inputs)

    def test_encoder_encodes_the_sources(self):
        self.encoder_decoder()

    def test_batch_encode(self):
        INPUT_SIZE = 10
        HIDDEN_SIZE = 20
        EMBEDDIN_DIM = 50
        VOCAB_SIZE = INPUT_SIZE
        self.encoder_decoder = EncoderDecoder(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                                              embedding_dim=EMBEDDIN_DIM, vocab_size=VOCAB_SIZE, bidirectional=False)
