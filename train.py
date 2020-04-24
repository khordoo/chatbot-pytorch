import os
import logging
import joblib
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from src import data_loader as dl
from tensorboardX import SummaryWriter
from src.model import EncoderDecoder
from src.utility import Utility
from src.tokenizer import Tokenizer
from data.contractions import contractions_dict

logging.basicConfig(format='%(asctime)-15s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GENRE = 'comedy'  # None -> all genres
SAVE_DIR = 'saves'
MAX_TOKEN_LENGTH = 20
# TEST_FRACTION = 0.2  # TODO :Reduce to 0.1 in actual run.
TEST_SIZE=1500
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
EPOCHS = 20

if __name__ == '__main__':
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer = Tokenizer(contractions_dict=contractions_dict)
    data_loader = dl.DialogLoaderTransformer(data_directory=dl.DATA_DIRECTORY,
                                             delimiter=dl.DELIMITER,
                                             movie_titles_headers=dl.MOVIES_TITLE_HEADERS,
                                             movie_lines_headers=dl.MOVIE_LINES_HEADERS,
                                             movie_conversation_headers=dl.MOVE_CONVERSATION_SEQUENCE_HEADERS)
    # loading the data
    source_texts, target_texts = data_loader.get_training_data(genre=GENRE, shuffle=True)

    tokenizer.fit_on_text(source_texts + target_texts, min_keep_frequency=MIN_TOKEN_FREQ)
    source_sequences = tokenizer.convert_text_to_number(source_texts)
    target_sequences = tokenizer.convert_text_to_number(target_texts)
    source_sequences, target_sequences = tokenizer.filter(source_numbers=source_sequences,
                                                          target_numbers=target_sequences,
                                                          max_token_size=MAX_TOKEN_LENGTH,
                                                          remove_unknown=True)

    source_sequences = Utility.tensorize(source_sequences, dtype=torch.long, device=DEVICE)
    target_sequences = Utility.tensorize(target_sequences, dtype=torch.long, device=DEVICE)

    train_source, train_target, test_source, test_target = Utility.split_train_test(source_sequences, target_sequences,
                                                                                    test_size=TEST_SIZE)

    encoder_decoder = EncoderDecoder(vocab_size=tokenizer.dictionary_size,
                                     hidden_size=HIDDEN_SIZE,
                                     embedding_dim=EMBEDDING_DIM,
                                     bidirectional=False
                                     ).to(DEVICE)

    optimizer = torch.optim.Adam(encoder_decoder.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(comment='-' + datetime.now().isoformat(timespec='seconds'))
    for epoch in range(EPOCHS):
        losses = []
        blue_scores = []
        for batch_source, batch_target in Utility.batch_generator(train_source, train_target, batch_size=BATCH_SIZE):
            optimizer.zero_grad()
            batch_encoder_outs, batch_encoder_hidden = encoder_decoder.encode(batch_source)
            decoder_outs_batch = []
            target_indexes_batch = []
            bleu_sum = 0
            for position, target_sequence in enumerate(batch_target):
                encoder_hidden = Utility.get_batch_item(batch_encoder_hidden, position=position)
                decoder_hidden = encoder_hidden
                decoder_input = Utility.get_single_batch_tensor(tokenizer.sos_index, DEVICE)
                predicted_indexes = []
                for target_index in target_sequence:
                    decoder_out, decoder_hidden = encoder_decoder.decode(decoder_input, decoder_hidden)
                    predicted_index = decoder_out.argmax(dim=2)
                    if np.random.random() < TEACHER_FORCING_PROB:
                        decoder_input = Utility.get_single_batch_tensor(target_index, DEVICE)
                    else:
                        decoder_input = predicted_index

                    decoder_outs_batch.append(decoder_out.squeeze(0))
                    predicted_indexes.append(predicted_index.item())

                target_indexes_batch.extend(target_sequence)
                bleu_sum += Utility.belu_score(predicted_indexes, reference_sequences=target_sequence.cpu().numpy())

            decoder_outs_batch = torch.cat(decoder_outs_batch).to(DEVICE)
            target_indexes_batch = torch.LongTensor(target_indexes_batch).to(DEVICE)
            loss = F.cross_entropy(decoder_outs_batch, target_indexes_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            bleu_sum /= len(batch_target)
            blue_scores.append(bleu_sum)
        mean_loss = np.array(losses).mean()
        mean_bleu_train = np.array(blue_scores).mean()
        mean_bleu_test = Utility.evaluate_net(encoder_decoder, test_source, test_target, tokenizer.sos_index, DEVICE)
        logger.info(
            f'Epoch:{epoch}, Mean loss:{mean_loss:.4f}, Mean BLEU:{mean_bleu_train:.4f}, Mean Test BLEU:{mean_bleu_test:.4f}')
        writer.add_scalar('Mean_loss', mean_loss, epoch)
        writer.add_scalar('Mean_Bleu', mean_bleu_train, epoch)
        writer.add_scalar('Mean_Test_Bleu', mean_bleu_test, epoch)

        if epoch % SAVE_CHECKPOINT_EVERY == 0:
            save_name = f'epoch_{epoch}_blue_{mean_bleu_train:.3f}_bleu_test_{mean_bleu_test:.3f}.pt'
            torch.save(encoder_decoder.state_dict(), os.path.join(SAVE_DIR, save_name))
    writer.close()
