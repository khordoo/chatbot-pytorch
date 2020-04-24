import unittest
from src.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    """Tests The utility class methods"""

    def setUp(self):
        self.text_input = ['Hi! , How are you?', 'How everything is going?', 'how old are you']
        self.tokenizer = Tokenizer()

    def test_tokenizer_dictionary_contains_special_tokens(self):
        """test that creation creation,
         dictionary contains the <sos>, <eos> <pad> and <unk> tokens"""
        expected_num_dictionary_items = len(['<sos>', '<eos>', '<pad>', '<unk>'])
        self.assertEqual(expected_num_dictionary_items, self.tokenizer.dictionary_size)

    def test_text_to_number_works_without_trimming(self):
        """tests the methods properly splits the input sequence."""
        self.tokenizer.fit_on_text(self.text_input, min_keep_frequency=0)
        expected_dictionary_size = 14
        self.assertEqual(expected_dictionary_size, self.tokenizer.dictionary_size)

    def test_text_to_number_works_with_trimming(self):
        """tests the methods properly splits the input sequence."""
        self.tokenizer.fit_on_text(self.text_input, min_keep_frequency=3)
        expected_dictionary_size = 5
        self.assertEqual(expected_dictionary_size, self.tokenizer.dictionary_size)

    def test_text_to_numbers(self):
        """tests tokenizer converts text into numbers"""
        input_text = ['how are you?']
        self.tokenizer.fit_on_text(input_text)
        text_indexes = self.tokenizer.convert_text_to_number(input_text)
        expected = [[4, 5, 6, 7, 2]]
        comparison = expected == text_indexes
        self.assertTrue(all(comparison[0]))

    def test_numbers_to_text(self):
        """tests tokenizer converts text into numbers"""
        input_text = ['pytorch is awesome']
        self.tokenizer.fit_on_text(input_text)
        text = self.tokenizer.convert_number_to_text([4, 5, 6, 2])
        expected = input_text[0]
        self.assertEqual(expected, text)

    def test_filter_is_filtering_long_sentences(self):
        """testes the filter function removes the long token
         jointly together from both sources and targets"""
        source_numbers = [[1, 4], [4, 5, 6], [9]]
        target_numbers = [[11, 22, 33, 44], [44, 55], [88, 99, 100, 110]]

        filtered_sources, filtered_targets = self.tokenizer.filter(source_numbers, target_numbers,
                                                                   max_token_size=3,
                                                                   remove_unknown=False)
        expected_source = [[4, 5, 6]]
        expected_targets = [[44, 55]]
        self.assertListEqual(expected_source[0], filtered_sources[0])
        self.assertEqual(expected_targets[0], filtered_targets[0])

    def test_filter_removes_token_containing_unknown_token_index(self):
        """testes the filter function removes with unknown tokens
        """
        unknown_index = self.tokenizer.unknown_index
        source_numbers = [[1, unknown_index], [4, 5], [9]]
        target_numbers = [[11, 22, 33], [44, unknown_index], [88, 99, 100]]

        filtered_sources, filtered_targets = self.tokenizer.filter(source_numbers, target_numbers,
                                                                   max_token_size=3,
                                                                   remove_unknown=True)
        expected_source = [[9]]
        expected_targets = [[88, 99, 100]]
        self.assertListEqual(expected_source[0], filtered_sources[0])
        self.assertEqual(expected_targets[0], filtered_targets[0])
