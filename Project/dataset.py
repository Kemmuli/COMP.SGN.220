#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import necessary modules
from typing import Tuple, Optional, Union, Dict, NoReturn
from pickle import load as pickle_load
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np

# Import a custom module
from feature_extraction import extract_and_serialize_features

# Set the docstring format and define the list of names that can be imported
__docformat__ = 'reStructuredText'
__all__ = ['GenreDataset']


# Define a custom Dataset class that loads audio data and labels
class GenreDataset(Dataset):

    def __init__(self,
                 data_dir: Union[str, Path] = 'training',
                 n_mels: Optional[int] = 40,
                 load_into_memory: Optional[bool] = True) -> NoReturn:
        """
        Constructor for the GenreDataset class.

        :param data_dir: Directory to read data from.
        :type data_dir: str or Path
        :param n_mels: Number of Mel frequency bands to extract from audio data.
        :type n_mels: int, optional
        :param load_into_memory: Whether to load data into memory.
        :type load_into_memory: bool, optional
        """

        # Call the constructor of the parent class
        super().__init__()

        # Set the path to the directory containing the data
        data_parent_dir = 'mel_features_n_' + str(n_mels)
        data_path = Path(data_parent_dir, data_dir)

        # If the data has not been preprocessed, extract and serialize features
        if not data_path.is_dir():
            extract_and_serialize_features(n_mels)

        # Get a list of all the data files
        self.files = list(data_path.iterdir())

        # Load the data into memory if specified
        self.load_into_memory = load_into_memory
        if self.load_into_memory:
            for i, a_file in enumerate(self.files):
                self.files[i] = self._load_file(a_file)

    @staticmethod
    def _load_file(file_path: Path) -> Dict[str, Union[int, np.ndarray]]:
        """
        Load a file from disk using its path.

        :param file_path: Path to the file to load.
        :type file_path: Path
        :return: A dictionary containing the audio features and label.
        :rtype: dict[str, int|numpy.ndarray]
        """
        # Open the file and load its contents using pickle
        with file_path.open('rb') as f:
            return pickle_load(f)

    def __len__(self) -> int:
        """
        Get the number of audio files in the dataset.

        :return: The number of audio files in the dataset.
        :rtype: int
        """
        return len(self.files)

    def __getitem__(self,
                    item: int) -> Tuple[np.ndarray, int]:
        """
        Get an audio file and its corresponding label.

        :param item: The index of the audio file to get.
        :type item: int
        :return: A tuple containing the audio features and the label.
        :rtype: tuple(numpy.ndarray, int)
        """
        if self.load_into_memory:
            # If the data is already loaded into memory, get it from the list
            the_item: Dict[str, Union[int, np.ndarray]] = self.files[item]
        else:
            # If the data is not in memory, load it from disk
            the_item = self._load_file(self.files[item])

        # Return the audio features and label
        return the_item['features'], the_item['class']


# EOF
