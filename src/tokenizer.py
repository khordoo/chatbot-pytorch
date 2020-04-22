import re
import collections
import numpy as np
from src.exceptions import UnrecognizedWordException


class Tokenizer:
    """This class converts the text into its numerical
    representation and vice versa."""

    def __init__(self, contractions_dict=None, max_sequence_length=20, min_token_frequency=10):
        self.START_TOKEN = "<sos>"
        self.PADDING_TOKEN = "<pad>"
        self.END_TOKEN = "<eos>"
        self.UNKNOWN_TOKEN = "<unk>"
        self.contractions_dict = contractions_dict
        self.max_length = max_sequence_length
        self.min_token_frequency = min_token_frequency
        self._initialize()

    def _initialize(self):
        """We reserve 0 index for pad token,
        although we don't do any padding in here."""
        self.word2index = {}
        self.index2word = {}
        self.word_counter = collections.Counter()
        for token in [self.PADDING_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN]:
            self._add_word(token)

    def create_dictionary(self, text_array):
        """Creates a numerical index value for every unique word"""
        print(f"Creating a dictionary of words from {len(text_array)} sentences. ")
        for sentence in text_array:
            sentence = self._normalize(sentence)
            self._add_sentence(sentence)
        print(f"Dictionary creation completed. Number of unique words:  {len(self.word2index)}")
        self._trim()

    def batch_convert_text_to_index(self, source_texts, target_texts):
        """We process the source and target sentences together.
          The reason is that,to be able to discard both
          source and target if an unknown word found in either of them.
          This is also the case, for maximum length filtering.
        """
        print('Converting words to indexes.')
        eos_index = self.word2index[self.END_TOKEN]
        source_indexes, target_indexes = [], []
        for source, target in zip(source_texts, target_texts):
            try:
                tokenized_source = self._tokenize(source)
                tokenized_target = self._tokenize(target)
                if len(tokenized_source) > self.max_length or len(tokenized_target) > self.max_length:
                    # Ignore the pair, if either of source or target sentence is long
                    continue
            except KeyError:
                # Ignore the pair,if either source or target sentence has an unknown word
                continue
            tokenized_source.append(eos_index)
            tokenized_target.append(eos_index)
            source_indexes.append(tokenized_source)
            target_indexes.append(tokenized_target)

        print(f"Tokenized completed: {len(source_indexes)} sources and {len(target_indexes)} targets")
        return np.array(source_indexes), np.array(target_indexes)

    def text_to_index(self, sentences, raise_unknown=False):
        """Convert words in sentences to their numerical index values
        """
        indexes = []
        end_token_index = self.word2index[self.END_TOKEN]
        unknown_token_index = self.word2index[self.UNKNOWN_TOKEN]
        for sentence_text in sentences:
            sentence_text = self._normalize(sentence_text)
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
        return indexes

    def index_to_text(self, word_numbers):
        """Converts an array of numbers to a text string"""
        ignore_index = [self.word2index[self.PADDING_TOKEN]]
        return " ".join([self.index2word[idx] for idx in word_numbers if idx not in ignore_index])

    def _normalize(self, text):
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
        sentence_text = self._normalize(sentence_text)
        return [
            self.word2index[word]
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

    def _trim(self):
        top_words = [word for word, count in self.word_counter.items()
                     if count > self.min_token_frequency]
        print(
            f'Found {len(self.word2index) - len(top_words)} with frequency less that {self.min_token_frequency}')
        if top_words:
            self._initialize()
            for word in top_words:
                self._add_word(word)

        print(f'Trimmed dictionary word count:{len(self.word2index)} ')

    @property
    def dictionary_size(self):
        return len(self.word2index)

    @property
    def sos_index(self):
        return self.word2index[self.START_TOKEN]

    @property
    def eos_index(self):
        return self.word2index[self.END_TOKEN]
