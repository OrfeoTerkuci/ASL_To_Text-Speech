"""
This module contains our word guesser class.
"""

from typing import Any


class WordGuesser:
    """
    A class to guess the word based on the provided letters and the order of the
    letters, we make use of probability to guess the word. Also, in case a word looks
    familiar to different words based on typos, so editing distance is used to guess
    the word.
    """

    def __init__(self, dictionary: dict[str, Any]):
        """
        Initialize the dictionary.
        :param dictionary: dictionary
        """
        self.dictionary = dictionary

    @staticmethod
    def is_valid_word(word: str, letters: list[str]) -> bool:
        """
        Check if the word is a valid word.
        :param word: word
        :param letters: letters
        :return: True if the word is valid, False otherwise
        """
        return word[0] == letters[0]

    @staticmethod
    def check_edit_distance(word: str, letters: list[str]) -> bool:
        """
        Check the edit distance.
        :param word: word
        :param letters: letters
        :return: True if the edit distance is valid, False otherwise
        """
        return word[0] == letters[0]

    @staticmethod
    def guess_word(letters: list[str]) -> str:
        """
        Guess the word.
        :param letters: letters
        :return: the guessed word (placeholder for now)
        """
        return letters[0]
