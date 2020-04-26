import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from src import data_loader as dl
from tensorboardX import SummaryWriter
from src.model import AttentionEncoderDecoder
from src.utility import Utility
from src.tokenizer import Tokenizer
from data.contractions import contractions_dict

logging.basicConfig(format='%(asctime)-15s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GENRE = 'family'  # None -> all genres
SAVE_DIR = 'saves'
MAX_TOKEN_LENGTH = 20
TEST_FRACTION = 0.05
MIN_TOKEN_FREQ = 10
HIDDEN_SIZE = 512
EMBEDDING_DIM = 50
DROPOUT = 0.1
TEACHER_FORCING_PROB = 0.5
CLIP = 10
BATCH_SIZE = 32
PRINT_EVERY = 10
SAVE_CHECKPOINT_EVERY = 10
LEARNING_RATE = 0.001
EPOCHS = 100

os.makedirs(SAVE_DIR, exist_ok=True)
tokenizer = Tokenizer(contractions_dict=contractions_dict)


def load_data():
    """Loads, cleans and transforms the raw text data and
     returns the tokenized source and target phrases"""
    data_loader = dl.DialogLoaderTransformer(data_directory=dl.DATA_DIRECTORY,
                                             delimiter=dl.DELIMITER,
                                             movie_titles_headers=dl.MOVIES_TITLE_HEADERS,
                                             movie_lines_headers=dl.MOVIE_LINES_HEADERS,
                                             movie_conversation_headers=dl.MOVE_CONVERSATION_SEQUENCE_HEADERS)
    # loading and cleaning
    source_texts, target_texts = data_loader.get_training_data(genre=GENRE, shuffle=True)

    tokenizer.fit_on_text(source_texts + target_texts, min_keep_frequency=MIN_TOKEN_FREQ)
    # converting texts to numbers
    source_sequences = tokenizer.convert_text_to_number(source_texts)
    target_sequences = tokenizer.convert_text_to_number(target_texts)
    source_sequences, target_sequences = tokenizer.filter(source_numbers=source_sequences,
                                                          target_numbers=target_sequences,
                                                          max_token_size=MAX_TOKEN_LENGTH,
                                                          remove_unknown=True)
    # converting numbers to tensors
    source_sequences = Utility.tensorize(source_sequences, dtype=torch.long, device=DEVICE)
    target_sequences = Utility.tensorize(target_sequences, dtype=torch.long, device=DEVICE)

    return source_sequences, target_sequences


class Trainer:
    """A wrapper class for managing a training session"""

    def __init__(self, util, device, save_checkpoint_every):
        self.util = util
        self.device = device
        self.save_checkpoint_every = save_checkpoint_every
        self.writer = SummaryWriter(comment='-' + datetime.now().isoformat(timespec='seconds'))

    def train(self, encoder_decoder, tokenizer, train_source, train_target, test_source, test_target, learning_rate,
              batch_size, teacher_forcing_ratio):
        """Receives batched tokenized source and target phrases to train the encoder-decoder model.
         For a better performance, we encode all sources in one run as a batch. We will then extract
         the individual encoded representations of each source from the batch to decode their corresponding
         target sequentially.
        """

        optimizer = torch.optim.Adam(encoder_decoder.parameters(), lr=learning_rate)

        for epoch in range(EPOCHS):
            losses = []
            blue_scores = []
            for batch_source, batch_target in self.util.batch_generator(train_source, train_target,
                                                                        batch_size=batch_size):
                optimizer.zero_grad()
                batch_encoder_outs, batch_encoder_hidden = encoder_decoder.encode(batch_source)
                batch_decoder_outs = []
                batch_target_indexes = []
                bleu_sum = 0
                # Sequentially decodes every index in every target sequence
                for position, target_sequence in enumerate(batch_target):
                    encoder_hidden = self.util.get_hidden_state_batch_item(batch_encoder_hidden, position=position)
                    encoder_outs = self.util.get_outputs_batch_item(batch_encoder_outs, position=position)
                    decoder_hidden = encoder_hidden
                    decoder_input = self.util.get_single_batch_tensor(tokenizer.sos_index, self.device)
                    predicted_indexes = []
                    for target_index in target_sequence:
                        decoder_out, decoder_hidden = encoder_decoder.decode(decoder_input=decoder_input,
                                                                             decoder_hidden=decoder_hidden,
                                                                             encoder_outs=encoder_outs)
                        predicted_index = decoder_out.argmax(dim=2)
                        # teacher forcing
                        if np.random.random() < teacher_forcing_ratio:
                            decoder_input = self.util.get_single_batch_tensor(target_index, self.device)
                        else:
                            decoder_input = predicted_index

                        batch_decoder_outs.append(decoder_out.squeeze(0))
                        predicted_indexes.append(predicted_index.item())

                    batch_target_indexes.extend(target_sequence)
                    bleu_sum += self.util.belu_score(predicted_indexes,
                                                     reference_sequences=target_sequence.cpu().numpy())

                # Calculating the loss for all the indexes in the batch, at once
                batch_decoder_outs = torch.cat(batch_decoder_outs).to(DEVICE)
                batch_target_indexes = torch.LongTensor(batch_target_indexes).to(DEVICE)
                loss = F.cross_entropy(batch_decoder_outs, batch_target_indexes)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                bleu_sum /= len(batch_target)
                blue_scores.append(bleu_sum)

            mean_loss = np.array(losses).mean()
            mean_bleu_train = np.array(blue_scores).mean()

            mean_bleu_test = self.util.evaluate_net(encoder_decoder, test_source, test_target, tokenizer.sos_index,
                                                    self.device)
            self.persist(epoch, mean_loss, mean_bleu_train, mean_bleu_test, encoder_decoder)
        self.writer.close()

    def persist(self, epoch, mean_loss, mean_bleu_train, mean_bleu_test, encoder_decoder):
        """Persists the data to the disk. This includes witting metrics logs
         to the tensorboard and network sates to the disk.
        """
        logger.info(
            f'Epoch:{epoch}, Mean loss:{mean_loss:.4f}, Mean BLEU:{mean_bleu_train:.4f}, Mean Test BLEU:{mean_bleu_test:.4f}')
        self.writer.add_scalar('Mean_loss', mean_loss, epoch)
        self.writer.add_scalar('Mean_Bleu', mean_bleu_train, epoch)
        self.writer.add_scalar('Mean_Test_Bleu', mean_bleu_test, epoch)

        if epoch % self.save_checkpoint_every == 0 or (epoch == EPOCHS - 1):
            save_name = f"epoch_{epoch}_{datetime.now().isoformat(timespec='seconds')}.pt"
            torch.save(encoder_decoder.state_dict(), os.path.join(SAVE_DIR, save_name))
            logger.info(f'Checkpoint saved --> {save_name}')


if __name__ == "__main__":
    source_sequences, target_sequences = load_data()

    train_source, train_target, test_source, test_target = Utility.split_train_test(source_sequences, target_sequences,
                                                                                    test_fraction=TEST_FRACTION)
    # Saving the words dictionary for later inference
    tokenizer.save_state_dict(SAVE_DIR)
    encoder_decoder = AttentionEncoderDecoder(vocab_size=tokenizer.dictionary_size,
                                              hidden_size=HIDDEN_SIZE,
                                              embedding_dim=EMBEDDING_DIM,
                                              bidirectional=False
                                              ).to(DEVICE)

    trainer = Trainer(device=DEVICE, save_checkpoint_every=SAVE_CHECKPOINT_EVERY, util=Utility)
    trainer.train(encoder_decoder, tokenizer, train_source, train_target, test_source, test_target,
                  learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, teacher_forcing_ratio=TEACHER_FORCING_PROB)
