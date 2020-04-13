"""
The data is from the Cornel Movies Dialogs Corpus
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
"""

from collections import defaultdict
from ast import literal_eval
import numpy as np


class MetaDataParser:
    """A class to parse the movies metadata """

    def __init__(self, data_directory, delimiter, movie_titles_headers, movie_lines_headers,
                 movie_conversation_headers):
        self.data_directory = data_directory
        self.movie_titles_headers = movie_titles_headers
        self.movie_lines_headers = movie_lines_headers
        self.movie_conversation_headers = movie_conversation_headers
        self.delimiter = delimiter
        self.movies = {}
        self.genres = defaultdict(list)
        self._load_movies_metadata()
        self._load_movie_lines()
        self._load_conversations()

    def _load_movies_metadata(self):
        print('Loading movie title metadata...')
        for movie in self._parse('movie_titles_metadata.txt', headers=self.movie_titles_headers):
            genres = literal_eval(movie['genres'].strip())
            self._add_movie(movie)
            self._add_genre(genres, movie)

    def _load_movie_lines(self):
        """loads actual lines(conversation texts)"""
        print('Loading movie lines...')
        for line_content in self._parse('movie_lines.txt', headers=self.movie_lines_headers):
            self._add_lines(line_content)

    def _load_conversations(self):
        """Loads the conversation sequences"""
        print('Loading conversations....')
        for parsed_line in self._parse('movie_conversations.txt', headers=self.movie_conversation_headers):
            self._add_conversations(parsed_line)

    def _parse(self, file_name, headers):
        with open(self.data_directory + '/' + file_name, 'r', encoding='ISO-8859-1') as f:
            return [
                self._dict_reader(line, headers=headers)
                for line in f.readlines()
            ]

    def _dict_reader(self, raw_line, headers):
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

    def display_genres(self):
        print('Total number of movies', len(self.movies))
        print('Total number of genres', len(self.genres))
        for genre, movies in self.genres.items():
            print(f'{genre}:{len(movies)}')
        return self.genres.keys()

    def get_conversation_pairs(self, genre):
        conversations = []
        for movie_id in self.genres[genre]:
            conversations.extend(self.movies[movie_id]['conversations'])
        return conversations

    def get_separated_phrase_reply(self, genre):
        first_person, second_person = [], []
        for conversation_pair in self.get_conversation_pairs(genre):
            first_person.append(conversation_pair[0])
            second_person.append(conversation_pair[1])
        return first_person, second_person

    def show_sample_dialog(self, genre, limit=10):
        conversations = self.get_conversation_pairs(genre=genre)
        start = np.random.randint(0, len(conversations) - limit - 1)
        return conversations[start:start + limit]
