import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from src.data_loader import DialogLoaderTransformer
from tensorboardX import SummaryWriter
# from src.model import EncoderGRU
# from src.model import DecoderGRU
# from src.model import EncoderDecoderMediator
from src.model import EncoderDecoder
from src.utility import Utility
from src.session import TrainingSession
from src.tokenizer import Tokenizer
from data.contractions import contractions_dict
import logging

logging.basicConfig(format='%(asctime)-15s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MOVIES_TITLE_HEADERS = ['movieId', 'title', 'year', 'rating', 'votes', 'genres']
MOVIE_LINES_HEADERS = ['lineId', 'characterId', 'movieId', 'characterName', 'text']
MOVE_CONVERSATION_SEQUENCE_HEADERS = ['characterID1', 'characterID2', 'movieId', 'lineIds']
DELIMITER = '+++$+++'
DATA_DIRECTORY = 'data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GENRE = 'family'  # None -> all genres
MAX_TOKEN_LENGTH = 20
TEST_FRACTION = 0.2  # TODO :Reduce to 0.1 in actual run.
MIN_TOKEN_FREQ = 10
HIDDEN_SIZE = 512
EMBEDDING_DIM = 50
DROPOUT = 0.1
TEACHER_FORCING_PROB = 0.5
CLIP = 10
BATCH_SIZE = 32
PRINT_EVERY = 10
CHECKPOINT_EVERY = 10
LEARNING_RATE = 0.001
EPOCHS = 20

if __name__ == '__main__':
    tokenizer = Tokenizer(contractions_dict=contractions_dict)
    data_loader = DialogLoaderTransformer(data_directory=DATA_DIRECTORY,
                                          delimiter=DELIMITER,
                                          movie_titles_headers=MOVIES_TITLE_HEADERS,
                                          movie_lines_headers=MOVIE_LINES_HEADERS,
                                          movie_conversation_headers=MOVE_CONVERSATION_SEQUENCE_HEADERS)
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
                                                                                    test_fraction=TEST_FRACTION)

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
                bleu_sum += Utility.belu_score(predicted_indexes, reference_sequences=target_sequence)

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
        writer.add_scalar('Mean loss', mean_loss, epoch)
        writer.add_scalar('Mean Bleu', mean_bleu_train, epoch)
        writer.add_scalar('Mean Test Bleu', mean_bleu_test, epoch)
