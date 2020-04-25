import torch
import numpy as np
from nltk.translate import bleu_score


class Utility:

    @staticmethod
    def tensorize(sequences, dtype, device):
        """Converts an array of sequences to an array of tensors"""
        return [torch.tensor(sequence, dtype=dtype).to(device) for sequence in sequences]

    @staticmethod
    def split_train_test(sources, targets, test_fraction=None, test_size=None):
        """Splits the sequence and returns the train and test sequences."""
        if test_size is None:
            test_size = int(test_fraction * len(sources))
            assert test_size < len(sources)
        print('Traing size:', len(sources) - test_size, 'test size:', test_size)

        return sources[test_size:], \
               targets[test_size:], \
               sources[:test_size], \
               targets[:test_size]

    @staticmethod
    def batch_generator(sources, targets, batch_size):
        """Creates data batches from source and target sequences.
           If number of samples is not exactly dividable by the batch size,
           then the length of the last batch would be smaller than the rest of batches.
           we are not dropping this last batch, since we are gonna use the mean loss.
        """
        for position in range(0, len(sources), batch_size):
            yield (sources[position:position + batch_size],
                   targets[position:position + batch_size])

    @staticmethod
    def get_batch_item(batch_values, position):
        return (batch_values[0][:, position:position + 1, :].contiguous(),
                batch_values[1][:, position:position + 1, :].contiguous())

    @staticmethod
    def get_single_batch_tensor(sos_index, device):
        return torch.LongTensor([[sos_index]]).to(device)

    @staticmethod
    def belu_score(predicted_seq, reference_sequences):
        smoothing_fn = bleu_score.SmoothingFunction()
        reference_sequences = np.expand_dims(reference_sequences, axis=0)
        return bleu_score.sentence_bleu(reference_sequences, predicted_seq,
                                        smoothing_function=smoothing_fn.method1,
                                        weights=(0.5, 0.5))

    @staticmethod
    def evaluate_net(model, batch_source, batch_target, sos_index, device):
        """Evaluates the performance of the model using the test set"""
        encoder_out, batch_encoder_hidden = model.encode(batch_source)
        bleu_sum = 0
        for position, target_sequence in enumerate(batch_target):
            encoder_hidden = Utility.get_batch_item(batch_encoder_hidden, position=position)
            decoder_hidden = encoder_hidden
            decoder_input = torch.LongTensor([[sos_index]]).to(device)
            predicted_indexes = []
            for _ in range(len(target_sequence)):
                decoder_out, decoder_hidden = model.decode(decoder_input, decoder_hidden)
                predicted_index = decoder_out.argmax(dim=2)
                decoder_input = predicted_index
                predicted_indexes.append(predicted_index.item())

            bleu_sum += Utility.belu_score(predicted_indexes, reference_sequences=target_sequence.cpu().numpy())
        return bleu_sum / len(batch_source)

    @staticmethod
    def predict(source_texts, model, tokenizer, device, max_prediction_len=10):
        """Receives an array of texts and provdes replies for all texts separately"""
        assert isinstance(source_texts, list)
        source_indexes = tokenizer.convert_text_to_number(source_texts)
        source_indexes = [torch.LongTensor(source_index).to(device) for source_index in source_indexes]
        sos_index = tokenizer.sos_index
        eos_index = tokenizer.eos_index
        encoder_out, batch_encoder_hidden = model.encode(source_indexes)
        # Doing prediction
        batch_predicted_texts = []
        batch_predicted_indexes = []
        for position in range(len(source_indexes)):
            encoder_hidden = Utility.get_batch_item(batch_encoder_hidden, position=position)
            decoder_hidden = encoder_hidden
            decoder_input = torch.LongTensor([[sos_index]]).to(device)
            predicted_indexes = []
            for _ in range(max_prediction_len):
                decoder_out, decoder_hidden = model.decode(decoder_input, decoder_hidden)
                predicted_index = decoder_out.argmax(dim=2)
                decoder_input = predicted_index
                if predicted_index.item() == eos_index:
                    break
                predicted_indexes.append(predicted_index.item())
            batch_predicted_indexes.append(predicted_indexes)
            batch_predicted_texts.append(tokenizer.convert_number_to_text(predicted_indexes))

        return batch_predicted_texts, batch_predicted_indexes
