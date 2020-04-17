import torch
import torch.nn as nn


class ChatBot(nn.Module):
    def __init__(self, vocab_size, embeddings_dims, hidden_size):
        super(ChatBot, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims)
        self.encoder = nn.LSTM(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        return self.embeddings(x)

    def encode(self, emb):
        _, (hidden_state, context) = self.encoder(emb)
        return hidden_state

    # def decode(self, x):
    #     emb = self.embeddings(x)
    #     hidden_state = self.encode(emb)
    #     out, _ = self.decoder(emb, hidden_state)
    #     out = self.word_mapper(out)
    #     return out

    def decode_teacher(self, inputs, outputs):
        emb = self.embeddings(inputs)
        hidden_state = self.encode(emb)
        emb_out = self.embeddings(outputs)
        out, _ = self.decoder(emb_out, hidden_state.unsqueeze(0))
        out = self.linear(out)
        return out

    def decode_argmax(self, questions, responses_index_batch, tokenizer):
        sentence_prob_batch = self.decode_teacher(questions, responses_index_batch)
        sentence_indexes_batch = []
        sentence_words_batch = []
        for sentence_prob, sentence_response_indexes in zip(sentence_prob_batch, responses_index_batch):
            sentence_indexes = self._prob_to_index_argmax(sentence_prob, sentence_response_indexes, tokenizer)
            sentence_indexes_batch.append(sentence_indexes)
            # sentence_words_batch.append(self._index_to_word(sentence_indexes, tokenizer))
        return sentence_indexes_batch

    def _prob_to_index_argmax(self, sentence_prob, sentence_response_indexes, tokenizer):
        indexes = []
        end_token_index = tokenizer.word_to_index[tokenizer.END_TOKEN]
        for word_prob, response_word_index in zip(sentence_prob, sentence_response_indexes):
            # NOt sure the best strategy
            # if response_word_index == end_token_index:
            #     indexes.append(end_token_index)
            #     while len(indexes) < len(sentence_response_indexes):
            #         indexes.append(response_word_index)
            #     break
            # else:
            indexes.append(word_prob.argmax())

        return indexes

    def _index_to_word(self, indexes, tokenizer):
        unknown_token = tokenizer.UNKNOWN_TOKEN
        return [
            tokenizer.index_to_word.get(index.item(), unknown_token)
            for index in indexes
        ]
