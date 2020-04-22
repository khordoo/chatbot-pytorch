import torch
from src.data_loader import DialogLoaderTransformer
from src.model import EncoderGRU
from src.model import DecoderGRU
from src.model import EncoderDecoderMediator
from src.session import TrainingSession
from src.tokenizer import Tokenizer
from data.contractions import contractions_dict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MOVIES_TITLE_HEADERS = ['movieId', 'title', 'year', 'rating', 'votes', 'genres']
MOVIE_LINES_HEADERS = ['lineId', 'characterId', 'movieId', 'characterName', 'text']
MOVE_CONVERSATION_SEQUENCE_HEADERS = ['characterID1', 'characterID2', 'movieId', 'lineIds']
DELIMITER = '+++$+++'
DATA_DIRECTORY = 'data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GENRE = 'family'  # None -> all genres
MAX_TOKEN_LENGTH = 10
MIN_TOKEN_FREQ = 3
HIDDEN_STATE_SIZE = 512
EMBEDDINGS_DIMS = 50
DROPOUT = 0.1
TEACHER_FORCING_PROB = 0.5
CLIP = 10
BATCH_SIZE = 32
PRINT_EVERY = 10
CHECKPOINT_EVERY = 10
LEARNING_RATE = 0.001
EPOCHS = 20

if __name__ == '__main__':
    tokenizer = Tokenizer(contractions_dict=contractions_dict,
                          min_token_frequency=MIN_TOKEN_FREQ,
                          max_sequence_length=MAX_TOKEN_LENGTH)

    data_loader = DialogLoaderTransformer(data_directory=DATA_DIRECTORY,
                                          delimiter=DELIMITER,
                                          movie_titles_headers=MOVIES_TITLE_HEADERS,
                                          movie_lines_headers=MOVIE_LINES_HEADERS,
                                          movie_conversation_headers=MOVE_CONVERSATION_SEQUENCE_HEADERS)
    # loading the data
    source_texts, target_texts = data_loader.get_training_data(genre=GENRE, shuffle=True)
    tokenizer.create_dictionary(source_texts + target_texts)
    source_indexes, target_indexes = tokenizer.batch_convert_text_to_index(source_texts=source_texts,
                                                                           target_texts=target_texts)

    # Creating encoder decoder classes
    encoder = EncoderGRU(input_size=tokenizer.dictionary_size,
                         hidden_size=HIDDEN_STATE_SIZE,
                         embeddings_dims=EMBEDDINGS_DIMS,
                         dropout=DROPOUT,
                         bidirectional=True,
                         device=DEVICE).to(DEVICE)

    decoder = DecoderGRU(input_size=tokenizer.dictionary_size,
                         hidden_size=HIDDEN_STATE_SIZE,
                         embeddings_dims=EMBEDDINGS_DIMS,
                         vocab_size=tokenizer.dictionary_size,
                         dropout=DROPOUT).to(DEVICE)
    # A simple class for encapsulating the encode-decode steps
    encoder_decoder = EncoderDecoderMediator(encoder, decoder, tokenizer=tokenizer, device=DEVICE)

    # start the training
    trainer = TrainingSession(encoder=encoder,
                              decoder=decoder,
                              encoder_decoder=encoder_decoder,
                              tokenizer=tokenizer,
                              learning_rate=LEARNING_RATE,
                              teacher_forcing_prob=TEACHER_FORCING_PROB,
                              print_every=PRINT_EVERY,
                              device=DEVICE)

    trainer.train(source_indexes,
                  target_indexes,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  checkpoint_every=CHECKPOINT_EVERY)
