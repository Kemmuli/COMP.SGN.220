#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableSequence, List, Union, MutableMapping

import numpy as np
import torch


__docformat__ = 'reStructuredText'
__all__ = [
    'get_audio_file_data',
    'extract_mel_band_energies',
    'serialize_features_and_classes',
    'dataset_iteration'
]


def get_audio_file_data(audio_file: str) -> np.ndarray:
    """Loads and returns the audio data from the `audio_file`.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str
    :return: Data of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    pass


def extract_mel_band_energies(audio_file: str) -> np.ndarray:
    """Extracts and returns the mel-band energies from the `audio_file` audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: str
    :return: Mel-band energies of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    pass


def serialize_features_and_classes(features_and_classes: MutableMapping[str, Union[np.ndarray, int]]) -> None:
    """Serializes the features and classes.

    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    """
    pass


def dataset_iteration(dataset: torch.utils.data.Dataset) -> None:
    """Iterates over the dataset using the DataLoader of PyTorch.

    :param dataset: Dataset to iterate over.
    :type dataset: torch.utils.data.Dataset
    """
    pass

# EOF
