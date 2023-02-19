#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, MutableSequence, Union, Dict, Any, NoReturn
from pathlib import Path
from pickle import dump
from librosa.core import load as lb_load, stft
from librosa.filters import mel
from torchaudio.datasets import GTZAN
import numpy as np

from copy import deepcopy

__docformat__ = 'reStructuredText'
__all__ = ['extract_and_serialize_features']


def extract_and_serialize_features(n_mels: int = 40) -> NoReturn:
    """
    Extracts Mel Band Energies (MBEs) from audio files and serializes them to disk.

    :param n_mels: The number of Mel bands to use when computing the MBEs.
    :type n_mels: int
    """
    parent_folder = './genres/'
    sub_folders = list(Path(parent_folder).iterdir())
    n_fft = 2048

    # Remove the ".mf" folder, which contains the metadata for the GTZAN dataset.
    audio_folders = list(filter(lambda name: '.mf' not in str(name), sub_folders))

    def get_genre_from(path: Path) -> str:
        """
        Extracts the genre name from a Path object.

        :param path: The path to an audio file.
        :type path: Path
        :return: The genre name.
        :rtype: str
        """
        return str(path).split("\\")[1]

    genres = [get_genre_from(dirname) for dirname in audio_folders]

    for subdir in audio_folders:
        # List all audio files in the current genre folder.
        files = list(Path(subdir).iterdir())

        for i, file in enumerate(files):
            # Load the audio file with librosa.
            data, sr = lb_load(path=file, sr=None, mono=True)

            # Minimum length of found file was 645 samples,
            # therefore we will cut all clips to that length.
            features = extract_mel_band_energies(data, sr, n_fft, n_mels)[:,:645]
            genre = get_genre_from(subdir)
            print(f"Shape of the features {features.shape} of genre {genre}")

            # Create a one-hot encoding of the genre for use in classification.
            genre_one_hot = create_one_hot_encoding(genre, genres)

            # Serialize the MBEs and the corresponding genre to disk.
            data_purpose = '/training/' if i < 80 else '/testing/'
            out_dir = Path('mel_features_n_' + str(n_mels) + data_purpose)
            f_name = genre + '_' + str(i)
            serialize(f_name, {'features': features, 'class': genre_one_hot}, out_dir)

            # Add noise and apply SpecAugment to the MBEs, then serialize the result.
            if 'training' in data_purpose:
                data_noised = add_noise(data)
                features_noised = extract_mel_band_energies(data_noised, sr, n_fft, n_mels)[:,:645]
                features_noised = spec_augment(features_noised)
                features_and_classes_noised = {'features': features_noised, 'class': genre_one_hot}
                f_name_noised = genre + '_noised_' + str(i)
                serialize(f_name_noised, features_and_classes_noised, out_dir)

    print(f"Serialised features with {n_mels} Mel-bands.")


def create_one_hot_encoding(word: str,
                            unique_words: MutableSequence[str]) \
        -> np.ndarray:
    """
    Function that creates a one-hot encoded numpy array for a given word
    based on a list of unique words.

    :param word: The word to encode.
    :type word: str
    :param unique_words: A list of unique words.
    :type unique_words: MutableSequence[str]

    :return: A one-hot encoded numpy array for the given word.
    :rtype:  np.ndarray
    """
    encoded = np.zeros((len(unique_words)))  # Initialize an array of zeros
    encoded[unique_words.index(word)] = 1  # Set the index of the word to 1

    return encoded  # Return the one-hot encoded array


def serialize(
        f_name: Union[str, Path],
        features_and_classes,
        output_directory: Path) -> None:
    """
    Serializes and saves a given object to disk in the specified directory.

    :param f_name: The name of the file to save the object as.
    :type f_name: Union[str, Path]
    :param features_and_classes: The object to save to disk.
    :param output_directory: The directory to save the object in.
    :type output_directory: Path

    :return: None
    :rtype: None
    """
    f_path = output_directory.joinpath(f_name)
    output_directory.mkdir(parents=True, exist_ok=True)
    with f_path.open('wb') as f:
        dump(features_and_classes, f)
    f_path = output_directory.joinpath(f_name)
    output_directory.mkdir(parents=True, exist_ok=True)
    with f_path.open('wb') as f:
        dump(features_and_classes, f)


def extract_mel_band_energies(data: np.ndarray,
                              sr: Optional[int] = 44100,
                              n_fft: Optional[int] = 1024,
                              n_mels: Optional[int] = 40,
                              hop_length: Optional[int] = 1024,
                              window: Optional[str] = 'hamming') \
        -> np.ndarray:
    """
    Computes the Mel Band Energies (MBEs) from a given audio signal.

    :param data: The audio signal to extract MBEs from.
    :type data: np.ndarray
    :param sr: The sampling rate of the audio signal.
    :type sr: Optional[int]
    :param n_fft: The length of the FFT window.
    :type n_fft: Optional[int]
    :param n_mels: The number of Mel bands.
    :type n_mels: Optional[int]
    :param hop_length: The number of samples between adjacent STFT columns.
    :type hop_length: Optional[int]
    :param window: The window function to apply.
    :type window: Optional[str]

    :return: The Mel Band Energies (MBEs) of the input audio signal.
    :rtype: np.ndarray
    """
    # Calculate the short-time Fourier transform (STFT) of the audio signal
    spec = stft(y=data,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window)

    # Calculate the Mel filterbank for the given parameters
    mel_filters = mel(sr=sr,
                      n_fft=n_fft,
                      n_mels=n_mels)

    # Calculate the Mel Band Energies (MBEs) by taking the dot product
    # of the filterbank and the squared magnitude of the STFT
    MBEs = np.dot(mel_filters, np.abs(spec) ** 2)

    return MBEs


def add_noise(audio_signal: np.ndarray) -> np.ndarray:
    """
    Adds Gaussian noise to an audio signal.

    :param audio_signal: The audio signal to add noise to.
    :type audio_signal: np.ndarray

    :return: The audio signal with added noise.
    :rtype: np.ndarray
    """
    samples = audio_signal.shape[0]
    # Stdev. of noise is 1/10th of max in signal
    noise = np.random.normal(0, audio_signal.max()/10, size=samples)
    return audio_signal + noise


def spec_augment(mel_spectrogram: np.ndarray):
    """
    Applies frequency and time masking to a given mel spectrogram.

    :param mel_spectrogram: The mel spectrogram to apply masking to.
    :type mel_spectrogram: np.ndarray

    :return: The mel spectrogram with frequency and time masking applied.
    """
    spectrogram = deepcopy(mel_spectrogram)
    # Making possible mask size max 10 bands
    f = np.random.randint(low=0, high=10)
    f0 = np.random.randint(low=0, high=40-f)
    min, max = f0, f+f0
    for i in range(min, max):
        spectrogram[i,:] = 0
    return spectrogram


if __name__ == '__main__':
    download = False
    if download:
        GTZAN(root=".", download=download)
    extract_and_serialize_features(80)

# EOF
