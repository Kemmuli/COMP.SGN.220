#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableSequence

import numpy as np

__docformat__ = 'reStructuredText'
__all__ = ['create_one_hot_encoding',
           'get_word_from_one_hot_encoding']


def create_one_hot_encoding(word: str,
                            unique_words: MutableSequence[str]) \
        -> np.ndarray:
    """Creates an one-hot encoding of the `word` word, based on the\
    list of unique words `unique_words`.

    :param word: Word to generate one-hot encoding for.
    :type word: str
    :param unique_words: List of unique words.
    :type unique_words: list[str]
    :return: One-hot encoding of the specified word.
    :rtype: numpy.ndarray
    """
    to_return = np.zeros((len(unique_words)))
    to_return[unique_words.index(word)] = 1
    return to_return


def get_word_from_one_hot_encoding(encoded_word: np.ndarray,
                                   unique_words: MutableSequence[str]):
    """Retrieves the word from its one-hot encoding `encoded_word`,\
    using the list of unique words `unique_words`.

    :param encoded_word: One-hot encoded word.
    :type encoded_word: numpy.ndarray
    :param unique_words: List of unique words.
    :type unique_words: list[str]
    :return: Actual word.
    :rtype: str
    """
    elements_one = np.where(encoded_word == 1)[0]

    if len(elements_one) > 1:
        raise AttributeError('More than one elements are equal to 1. '
                             'No proper one-hot encoding.')

    return unique_words[elements_one[0]]

# EOF
