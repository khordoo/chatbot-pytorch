import torch
import numpy as np
from nltk.translate import bleu_score


class Utility:

    @staticmethod
    def tensorize(sequences, dtype, device):
        """Converts an array of sequences to an array of tensors"""
        return [torch.tensor(sequence, dtype=dtype).to(device) for sequence in sequences]

    @staticmethod
    def split_train_test(sources, targets, test_fraction=0.5):
        """Splits the sequence and returns the train and test sequences."""
        test_start_index = int(test_fraction * len(sources))
        return sources[test_start_index:], \
               targets[test_start_index:], \
               sources[:test_start_index], \
               targets[:test_start_index]

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
        return batch_values[:, position:position + 1, :].contiguous()

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
