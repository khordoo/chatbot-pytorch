from collections import Counter


class TokenPreprocessor:
    def __init__(self, max_token_length=20, min_token_freq=10):
        self.UNKNOWN_TOKEN = '#UNK'
        self.BEGIN_TOKEN = '#BGN'
        self.END_TOKEN = '#END'
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counter = Counter()
        self.max_token_length = max_token_length
        self.min_token_freq = min_token_freq

    def process(self, talks_seq, responses_seq):
        talks_seq, responses_seq = self.text_to_sequence(talks_seq, responses_seq)
        self.fit_tokenizer(talks_seq, responses_seq)
        self._remove_low_frequency_words(self.min_token_freq)
        talks_seq, responses_seq = self.encode_sequences(talks_seq, responses_seq, self.max_token_length)
        print('Total conversation pairs: ', len(talks_seq))
        print('Encoding done for all the dialogs.')
        return talks_seq, responses_seq

    def text_to_sequence(self, talks, responses):
        print('Converting test strings into tokens...')
        talks_tokenized, responses_tokenized = [], []
        for talk, response in zip(talks, responses):
            talks_tokenized.append(talk.lower().split(' '))
            responses_tokenized.append(response.lower().split(' '))
        return talks_tokenized, responses_tokenized

    def fit_tokenizer(self, talks, responses):
        print('Fitting the tokenizer...')
        for index, word in enumerate([self.BEGIN_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN]):
            self.word_to_index[word] = index
            self.index_to_word[index] = word

        for talk, response in zip(talks, responses):
            all = talk + response
            for word in all:
                self.word_counter[word] += 1
                if word in self.word_to_index:
                    continue
                index = len(self.word_to_index)
                self.word_to_index[word] = index
                self.index_to_word[index] = word

    def _remove_low_frequency_words(self, min_count):
        print('Removing low frequency words....')
        low_frequency = []
        for word, count in self.word_counter.items():
            if count < min_count:
                low_frequency.append(word)

        for word in low_frequency:
            index = self.word_to_index[word]
            del self.word_to_index[word]
            del self.index_to_word[index]

    def encode_sequences(self, talks, responses, max_length):
        print('Encoding the words sequences....')
        talks_encoded, responses_encoded = [], []
        unknown_token_index = self.word_to_index[self.UNKNOWN_TOKEN]
        for talk, response in zip(talks, responses):
            talk = self._encode_words(talk, unknown_token_index)
            response = self._encode_words(response, unknown_token_index)
            if len(talk) > max_length or len(response) > max_length:
                continue
            if unknown_token_index in talk or unknown_token_index in response:
                continue
            talks_encoded.append(talk)
            responses_encoded.append(response)

        return talks_encoded, responses_encoded

    def _encode_words(self, word_sequence, unknown_token_index):
        encoded = [self.word_to_index[self.BEGIN_TOKEN]]
        encoded_word = [self.word_to_index.get(word, unknown_token_index) for word in word_sequence]
        encoded.extend(encoded_word)
        encoded.append(self.word_to_index[self.END_TOKEN])
        return encoded
