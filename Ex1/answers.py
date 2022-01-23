#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableSequence, List, Union, MutableMapping
from os.path import splitext, basename, join
from pathlib import Path
from Task1 import file_io
from random import randint

import plotting
import numpy as np
import torch
import librosa
import pickle


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
    sr = 44100
    data = librosa.load(path=audio_file, sr=sr)
    length = data[0].shape[-1] / sr
    print(f"File {audio_file}, with the length of {length:.1f} seconds")
    return data[0]


def extract_mel_band_energies(audio_file: str) -> np.ndarray:
    """Extracts and returns the mel-band energies from the `audio_file` audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: str
    :return: Mel-band energies of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    #  Task 2
    n_fft = 1024
    n_mels = 40
    sr = 44100

    ex_data = get_audio_file_data(audio_file)
    ex_data_stft = librosa.stft(ex_data, n_fft=n_fft, hop_length=512)
    ex_ESD = np.abs(ex_data_stft) ** 2
    M = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    MBE = np.dot(ex_ESD.T, M.T)

    return MBE


def serialize_features_and_classes(features_and_classes: MutableMapping[str, Union[np.ndarray, int]]) -> None:
    """Serializes the features and classes.

    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    """
    fp_prefix = './Task1/audio/'
    filepaths = file_io.get_files_from_dir_with_os(fp_prefix)
    c_dict = {}
    for fp in filepaths:
        mbe = extract_mel_band_energies(join(fp_prefix, fp))
        c = randint(0, 1)


def dataset_iteration(dataset: torch.utils.data.Dataset) -> None:
    """Iterates over the dataset using the DataLoader of PyTorch.

    :param dataset: Dataset to iterate over.
    :type dataset: torch.utils.data.Dataset
    """
    pass


def main():
    #  Task 1
    file_prefix = './Task1/audio/'  # Needs to be changed to the path of the folder holding the audiofiles.
    path1 = file_io.get_files_from_dir_with_os(file_prefix)
    path2 = file_io.get_files_from_dir_with_pathlib(file_prefix)
    print(*path1)
    print(*path2)
    """
    The os function returns a list of strings with only the filename, while the pathlib function returns
    the path given as a parameter + filename as a WindowsPath object, the object is dependant of the OS of course.
    """
    print(*sorted(path1))
    print(*sorted(path2))
    """
    Because the paths are not considered as a numerical value, the sorting is based on strings.
    This way "10" comes before 2 etc.
    This can be fixed by adding a key to the sorted function, which uses a lambda function that takes the last part of
    the paths without a suffix and converts it into an integer. This way the sorted function uses natural integer sorting
    """

    print(*sorted(path2, key=lambda p: int(p.stem)))  # pathlibs function stem takes the last part without a suffix

    for filepath in sorted(path1, key=lambda i: int(splitext(basename(i))[0])):
        data = get_audio_file_data(join(file_prefix, filepath))

    #  Task 2
    ex_path = './ex1.wav'
    ex_data = get_audio_file_data(ex_path)
    mbe = extract_mel_band_energies(ex_path)
    plotting.plot_audio_signal(ex_data)
    plotting.plot_mel_band_energies(mbe)


if __name__ == '__main__':
    main()

# EOF
