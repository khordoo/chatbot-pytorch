import re
import collections
import numpy as np
import logging


class Tokenizer:
    """This class converts the text into its numerical
    representation and vice versa."""

    def __init__(self, contractions_dict=None):
        self.START_TOKEN = "<sos>"
        self.PADDING_TOKEN = "<pad>"
        self.END_TOKEN = "<eos>"
        self.UNKNOWN_TOKEN = "<unk>"
        self.contractions_dict = contractions_dict
        self.logger = logging.getLogger(__name__)
        self._init_dictionary()

    def _init_dictionary(self):
        """We reserve 0 index for pad token,
        although we don't do any padding in here."""
        self.word2index = {}
        self.index2word = {}
        self.word_counter = collections.Counter()
        for token in [self.PADDING_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN]:
            self._add_word(token)

    def fit_on_text(self, text_array, min_keep_frequency=None):
        """Creates a numerical index value for every unique word"""
        self.logger.info(f"Creating a dictionary of words from {len(text_array)} sentences. ")
        for sentence in text_array:
            sentence = self._sanitize(sentence)
            self._add_sentence(sentence)
        self.logger.info(f"Dictionary creation completed. Number of unique words:  {len(self.word2index)}")

        if min_keep_frequency is not None:
            self._trim_dictionary(min_keep_frequency)

    def convert_text_to_number(self, text_sentences):
        """Converts text into its numerical representation"""
        self.logger.info('Converting words to indexes.')
        eos_index = self.word2index[self.END_TOKEN]
        source_indexes = []
        for source in text_sentences:
            tokenized_source = self._tokenize(source)
            tokenized_source.append(eos_index)
            source_indexes.append(tokenized_source)

        self.logger.info(f"Tokenized completed: {len(source_indexes)} text sentences")
        return np.array(source_indexes)

    def filter(self, source_numbers, target_numbers, max_token_size, remove_unknown=True):
        """Performs filtering on two sequences. Either if item in sequences would be filtered
           Both items from source and target will be remove if filtering conditions is not true for any of them.
        """
        unknown_token = self.word2index[self.UNKNOWN_TOKEN]
        filtered_sources, filtered_targets = [], []
        for source, target in zip(source_numbers, target_numbers):
            if len(source) > max_token_size or len(target) > max_token_size:
                continue
            if remove_unknown:
                if unknown_token in source or unknown_token in target:
                    continue
            filtered_sources.append(source)
            filtered_targets.append(target)
        self.logger.info(f'Filter completed, Sequences reduced from {len(filtered_sources)} to {len(filtered_sources)}')
        return filtered_sources, filtered_targets

    def convert_number_to_text(self, indexes):
        """Converts an array of numbers to a text string"""
        ignore_indexes = [self.eos_index]
        return " ".join([self.index2word[idx] for idx in indexes if idx not in ignore_indexes])

    def _sanitize(self, text):
        """Removes numbers and undesired characters from the text."""
        text = text.lower().strip()
        if self.contractions_dict is not None:
            for contraction, expanded in self.contractions_dict.items():
                text = re.sub(contraction, ' ' + expanded + ' ', text)

        text = re.sub(r"([.]{3})", r' ', text)
        text = re.sub(r"([.?!])", r' \1 ', text)
        text = re.sub(r'[^a-zA-Z.?!]+', r' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _tokenize(self, sentence_text):
        """Converts each word to its numerical representation"""
        sentence_text = self._sanitize(sentence_text)
        return [
            self.word2index.get(word, self.unknown_index)
            for word in sentence_text.strip().split(" ")
        ]

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

    def _trim_dictionary(self, min_keep_frequency=0):
        """Reduce dictionary vocab size based on minimum required word frequency."""
        top_words = [word for word, count in self.word_counter.items()
                     if count >= min_keep_frequency]
        self.logger.info(
            f'Found {len(self.word2index) - len(top_words)} with frequency less that {min_keep_frequency}')
        if top_words:
            self._init_dictionary()
            for word in top_words:
                self._add_word(word)

        self.logger.info(f'Trimmed dictionary word count:{len(self.word2index)} ')

    @property
    def dictionary_size(self):
        return len(self.word2index)

    @property
    def sos_index(self):
        return self.word2index[self.START_TOKEN]

    @property
    def eos_index(self):
        return self.word2index[self.END_TOKEN]

    @property
    def unknown_index(self):
        return self.word2index[self.UNKNOWN_TOKEN]
