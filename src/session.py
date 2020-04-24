import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import itertools
from tensorboardX import SummaryWriter


class TrainingSession:
    """A container class that runs the training job"""

    def __init__(self, encoder, decoder, encoder_decoder, tokenizer, device, learning_rate,
                 teacher_forcing_prob, print_every, num_keep_state_files=5, gradient_clip=5):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_decoder = encoder_decoder
        self.tokenizer = tokenizer
        self.device = device
        self.print_every = print_every
        self.learning_rate = learning_rate
        self.teacher_forcing_prob = teacher_forcing_prob
        self.gradient_clip = gradient_clip
        self.start_token_index = self.tokenizer.sos_index
        self.pad_token_index = self.tokenizer.word2index[self.tokenizer.PADDING_TOKEN]
        self.end_token_index = self.tokenizer.word2index[self.tokenizer.END_TOKEN]
        self.writer = SummaryWriter(comment='-' + datetime.now().isoformat(timespec='seconds'))
        self.checkpoint_cycler = itertools.cycle([*range(num_keep_state_files)])
        self.start_time = datetime.now()

    def train(self, sources, targets, teacher_forcing_ratio=0.5, batch_size=10, epochs=20,
              checkpoint_every=200):
        print('Training started with training {} sources and {} targets'.format(len(sources), len(targets)))

        encoder_optimizer = torch.optim.Adam(self.encoder_decoder.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = torch.optim.Adam(self.encoder_decoder.decoder.parameters(), lr=self.learning_rate)
        total_processed_batches = 0
        for epoch in range(1, epochs + 1):
            epoch_processed_batch = 0
            for sources, targets in self.batch_generator(sources, targets, batch_size):
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                loss, mean_bleu, predicted_indexes = self.encoder_decoder.step(sources, targets, teacher_forcing_ratio)

                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clip)
                nn.utils.clip_grad_norm_(self.decoder.parameters(), self.gradient_clip)
                encoder_optimizer.step()
                decoder_optimizer.step()

                epoch_processed_batch += 1
                total_processed_batches += 1
                self.report_progress(epoch, total_processed_batches, epoch_processed_batch, targets, loss, mean_bleu,
                                     predicted_indexes)
                self.create_check_point(total_processed_batches, checkpoint_every)

    def batch_generator(self, sources, targets, batch_size, shuffle=False, drop_last=True):
        """Creates data batches from source and target sequences.
           If number of samples is not exactly dividable by the batch size,
           then the length of the last batch would be smaller than the rest of batches.
           This might create a bumpy loss trend. Setting drop_last=True will drip that last smaller batch.
           We can also shuffle the samples per each epoch by setting shuffle=True
        """
        if shuffle:
            # If True, shuffles the data before each epoch
            random_idx = np.random.choice(len(sources), len(sources), replace=False)
            sources = sources[random_idx]
            targets = targets[random_idx]

        num_samples = len(sources)
        if drop_last:
            num_samples -= num_samples % batch_size

        for i in range(0, num_samples, batch_size):
            yield self._next_batch(sources, i, batch_size), \
                  self._next_batch(targets, i, batch_size)

    def _next_batch(self, source, position, batch_size):
        """Receives a list of sequences and and returns a batch of tensors"""
        batch = source[position:position + batch_size]
        return [torch.LongTensor(sequence).to(self.device) for sequence in batch]

    def report_progress(self, epoch, total_processed_batches, epoch_processed_batch, targets, loss, mean_bleu_score,
                        predicted_indexes):
        if total_processed_batches % self.print_every == 0:
            self._verify_predictions(predicted_indexes, targets, total_processed_batches)
            elapsed = (datetime.now() - self.start_time)
            print(f'Elapsed time:{elapsed}, epoch: {epoch}, total batches:{total_processed_batches},'
                  f' batch:{epoch_processed_batch}, loss: {loss.item()}, belu:{mean_bleu_score:.5f}')
            print('-------------------------------------------------------------')
            self.writer.add_scalar('loss:', loss.item(), total_processed_batches)
            self.writer.add_scalar('belu:', mean_bleu_score, total_processed_batches)

    def create_check_point(self, total_processed_batches, check_point_step):
        if total_processed_batches % check_point_step == 0:
            # Sve checkpoint, overwrites the oldest in a cycle
            loop_indexer = next(self.checkpoint_cycler)
            self.encoder_decoder.save_states(step=loop_indexer)
            print(f'Checkpoint saved: -> encoder/decoder{loop_indexer + 1}.dat')

    @torch.no_grad()
    def _verify_predictions(self, predicted_indexes_batch, targets, step):
        """Creates a target indexes for a random sample data point.
           The predicted indexes are then converted to text
           and displayed for visual inspection during the training.
        """
        random_target_index = np.random.randint(0, len(predicted_indexes_batch))
        target_text = self.tokenizer.convert_number_to_text(targets[random_target_index].cpu().data.numpy())
        # Retrieves the predicted indexes for the random target sample
        prediction_text = self.tokenizer.convert_number_to_text(predicted_indexes_batch[random_target_index])
        if step % self.print_every == 0:
            print('>', target_text)
            print('>', prediction_text)
