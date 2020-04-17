from collections import Counter


class TokenPreprocessor:
    def __init__(self, max_token_length=20, min_token_freq=10):
        self.UNKNOWN_TOKEN = '#UNK'
        self.BEGIN_TOKEN = '#BGN'
        self.END_TOKEN = '#END'
        self.PAD_TOKEN = '#PAD'
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counter = Counter()
        self.max_token_length = max_token_length
        self.min_token_freq = min_token_freq

    def process(self, questions, responses):
        questions, responses = self.text_to_sequence(questions, responses)
        self.fit_tokenizer(questions, responses)
        self._remove_low_frequency_words(self.min_token_freq)
        questions, responses = self.encode_sequences(questions, responses, self.max_token_length)
        print('Total conversation pairs: ', len(questions))
        print('Encoding done for all the dialogs.')
        return questions, responses

    def text_to_sequence(self, questions, responses):
        print('Converting test strings into tokens...')
        questions_tokenized, responses_tokenized = [], []
        for talk, response in zip(questions, responses):
            questions_tokenized.append(talk.lower().split(' '))
            responses_tokenized.append(response.lower().split(' '))
        return questions_tokenized, responses_tokenized

    def fit_tokenizer(self, questions, responses):
        print('Fitting the tokenizer...')
        for index, word in enumerate([self.BEGIN_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN, self.PAD_TOKEN]):
            self.word_to_index[word] = index
            self.index_to_word[index] = word

        for talk, response in zip(questions, responses):
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

        print('vocab before removing:',len(self.word_to_index))
        unknown_index=self.word_to_index[self.UNKNOWN_TOKEN]
        for word in low_frequency:
            # index = self.word_to_index[word]
            self.word_to_index[word]=unknown_index
            # self.index_to_word[index]=self.UNKNOWN_TOKEN
        print('vocab after removing :',len(self.word_to_index))

    def encode_sequences(self, questions, responses, max_length):
        print('Encoding the words sequences....')
        questions_encoded, responses_encoded = [], []
        unknown_token_index = self.word_to_index[self.UNKNOWN_TOKEN]
        for talk, response in zip(questions, responses):
            talk = self._encode_words(talk, unknown_token_index)
            response = self._encode_words(response, unknown_token_index)
            if len(talk) > max_length or len(response) > max_length:
                continue
            if unknown_token_index in talk or unknown_token_index in response:
                continue
            questions_encoded.append(talk)
            responses_encoded.append(response)

        return questions_encoded, responses_encoded

    def _encode_words(self, word_sequence, unknown_token_index):
        encoded = [self.word_to_index[self.BEGIN_TOKEN]]
        encoded_word = [self.word_to_index.get(word, unknown_token_index) for word in word_sequence]
        encoded.extend(encoded_word)
        encoded.append(self.word_to_index[self.END_TOKEN])
        return self._add_padding(encoded)

    def _add_padding(self, encoded):
        pad_token = self.word_to_index[self.PAD_TOKEN]
        while len(encoded) < self.max_token_length:
            encoded.append(pad_token)
        return encoded

    def get_vocab_size(self):
        return len(self.word_to_index)
