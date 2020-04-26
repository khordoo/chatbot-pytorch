"""
The data is from the Cornel Movies Dialogs Corpus
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
"""
import random
import logging
import argparse
from collections import defaultdict
from ast import literal_eval
import numpy as np
from pprint import pprint

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)

MOVIES_TITLE_HEADERS = ['movieId', 'title', 'year', 'rating', 'votes', 'genres']
MOVIE_LINES_HEADERS = ['lineId', 'characterId', 'movieId', 'characterName', 'text']
MOVE_CONVERSATION_SEQUENCE_HEADERS = ['characterID1', 'characterID2', 'movieId', 'lineIds']
DELIMITER = '+++$+++'
DATA_DIRECTORY = 'data'
random.seed(5871)


class DialogLoaderTransformer:
    """A class to load, parse and transform the the movies conversations """

    def __init__(self, data_directory, delimiter, movie_titles_headers, movie_lines_headers,
                 movie_conversation_headers):
        self.data_directory = data_directory
        self.movie_titles_headers = movie_titles_headers
        self.movie_lines_headers = movie_lines_headers
        self.movie_conversation_headers = movie_conversation_headers
        self.delimiter = delimiter
        self.movies = {}
        self.genres = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        self._load_data()

    def get_training_data(self, genre=None, shuffle=True):
        """Splits the conversation pairs into two separate arrays, source, target , for training."""
        self.logger.info('Loading movies dialogs...')
        conversation_pairs = self._get_conversation_pairs(genre, shuffle)
        return zip(*conversation_pairs)

    def show_sample_dialog(self, genre, limit=10):
        """Shows some sample dialogs for the specified genre"""
        conversations = self._get_conversation_pairs(genre=genre, shuffle=False)
        start = np.random.randint(0, len(conversations) - limit - 1)
        return conversations[start:start + limit]

    def show_genres(self):
        self.logger.info(f'Total number of movies: {len(self.movies)}')
        self.logger.info(f'Total number of genres: {len(self.genres)}')
        for genre, movies in self.genres.items():
            print(f'{genre}:{len(movies)}')
        return self.genres.keys()

    def _load_data(self):
        self._load_movies_metadata()
        self._load_movie_lines()
        self._load_conversations()

    def _load_movies_metadata(self):
        for movie in self._parse('movie_titles_metadata.txt', headers=self.movie_titles_headers):
            genres = literal_eval(movie['genres'].strip())
            self._add_movie(movie)
            self._add_genre(genres, movie)
        self.logger.info('Metadata loaded. movies: {}, genres:{}'.format(len(self.movies), len(self.genres)))

    def _load_movie_lines(self):
        """loads actual lines(conversation texts)"""
        for line_content in self._parse('movie_lines.txt', headers=self.movie_lines_headers):
            self._add_lines(line_content)

    def _load_conversations(self):
        """Loads the conversation sequences"""
        for parsed_line in self._parse('movie_conversations.txt', headers=self.movie_conversation_headers):
            self._add_conversations(parsed_line)

    def _add_conversations(self, line):
        """Reads the sequence of line ids and pairs them together."""
        line_ids = [line_id.strip() for line_id in literal_eval(line['lineIds'].strip())]
        movie = self.movies[line['movieId']]
        conversations = []
        for index in range(len(line_ids) - 1):
            conversation_pair = [
                movie['lines'][line_ids[index]]['text'],
                movie['lines'][line_ids[index + 1]]['text']
            ]
            conversations.append(conversation_pair)

        if 'conversations' not in movie:
            movie['conversations'] = []

        movie['conversations'].extend(conversations)

    def _parse(self, file_name, headers):
        with open(self.data_directory + '/' + file_name, 'r', encoding='ISO-8859-1') as f:
            return [
                self._parse_dict(line, headers=headers)
                for line in f.readlines()
            ]

    def _parse_dict(self, raw_line, headers):
        parsed_values = [meta.strip() for meta in raw_line.strip().split(self.delimiter)]
        return dict(zip(headers, parsed_values))

    def _add_movie(self, movie):
        self.movies[movie['movieId']] = movie

    def _add_genre(self, genres, movie):
        for genre in genres:
            self.genres[genre].append(movie['movieId'])

    def _add_lines(self, lines_data):
        movie = self.movies[lines_data['movieId']]
        if 'lines' not in movie:
            movie['lines'] = {}
        movie['lines'][lines_data['lineId']] = lines_data

    def _get_conversation_pairs(self, genre=None, shuffle=True):
        self.logger.info(f'Total movies with {genre} genre are {len(self.genres[genre])}')
        conversations = []
        if genre is not None:
            for movie_id in self.genres[genre]:
                conversations.extend(self.movies[movie_id]['conversations'])
        else:
            # return all the genres
            for movie in self.movies.values():
                conversations.extend(movie['conversations'])

        if shuffle:
            random.shuffle(conversations)
        self.logger.info('Loaded all the conversations , Total conversation pairs: {}'.format(len(conversations)))
        return conversations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-genres', required=False, action='store_true', default=False,
                        help='Show a list of available genres')
    parser.add_argument('--show-dial', required=False, action='store_true', default=False,
                        help='Display sample dialog from a specific genre')
    parser.add_argument('--genre', required=False, default=None,
                        help='Genre used for displaying the dialogs')
    args = parser.parse_args()
    data_loader = DialogLoaderTransformer(data_directory=DATA_DIRECTORY,
                                          delimiter=DELIMITER,
                                          movie_titles_headers=MOVIES_TITLE_HEADERS,
                                          movie_lines_headers=MOVIE_LINES_HEADERS,
                                          movie_conversation_headers=MOVE_CONVERSATION_SEQUENCE_HEADERS)

    if args.show_genres:
        data_loader.show_genres()
    if args.show_dial:
        dialogs = data_loader.show_sample_dialog(genre=args.genre)
        for i, dialog in enumerate(dialogs):
            if i % 2 == 0:
                print(dialog[0])
                print(dialog[1])
