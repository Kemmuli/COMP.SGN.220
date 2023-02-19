import librosa.effects

from utils import get_files_from_dir_with_pathlib, get_files_from_dir_with_pathlib, get_audio_file_data, \
    create_one_hot_encoding, plot_confusion_matrix, extract_mel_band_energies

from random import randint
from data_augmentation import add_white_noise, pitch_shift, add_impulse_response, spec_augment
from pickle import dump
from pathlib import Path
from os.path import splitext, join
import pandas as pd
import librosa as lb


def serialize_data(augment=None):
    """

    :param augment: Possible options ['None', 'noise', 'shift', 'reverberation', 'spec']
    :type augment: List[str]
    :return:
    """
    # Helper function
    def handle_pickling(audio, label, path, raw=True):
        if raw:
            mbe = extract_mel_band_energies(audio)
            data = {'features': mbe, 'class': label}
        else:
            data = {'features': audio, 'class': label}
        with path.open('wb') as f:
            dump(data, f)

    dirs = ['./training/', './testing/', './validation/']
    metafiles = ['train_meta.csv', 'test_meta.csv', 'val_meta.csv']
    labels = ['rain', 'sea_waves', 'chainsaw', 'helicopter']
    for i, folder in enumerate(dirs):
        df = pd.read_csv(metafiles[i])
        for idx, row in df.iterrows():
            filename = row['filename']
            label = row['label']
            label = create_one_hot_encoding(label, labels)
            fp = folder + filename

            # Extract raw audio and augment it.
            audio, sr = get_audio_file_data(fp)
            mbe = extract_mel_band_energies(audio)
            rir, isr = lb.load(path='impulse_response.wav', sr=sr, mono=True)
            pickle_prefix = splitext(fp)[0]
            pickle_path = Path(pickle_prefix + '.pickle')
            handle_pickling(audio, label, pickle_path)

            if 'noise' in augment:
                white_noise_audio = add_white_noise(audio, noise_level=0.05)
                noisy_pickle_path = Path(pickle_prefix + '-noise.pickle')
                handle_pickling(white_noise_audio, label, noisy_pickle_path)

            if 'shift' in augment:
                random_shift = randint(1, 8)
                shifted_audio = pitch_shift(data=audio, sr=sr, n_steps=random_shift)
                shifted_pickle_path = Path(pickle_prefix + '-shifted.pickle')
                handle_pickling(shifted_audio, label, shifted_pickle_path)

            if 'reverberation' in augment:
                reverberated_audio = add_impulse_response(audio, rir)
                reverb_pickle_path = Path(pickle_prefix + '-reverb.pickle')
                handle_pickling(reverberated_audio, label, reverb_pickle_path)

            if 'spec' in augment:
                spec_augmented_audio = spec_augment(mbe)
                spec_pickle_path = Path(pickle_prefix + '-spec.pickle')
                handle_pickling(spec_augmented_audio, label, spec_pickle_path, raw=False)


def main():
    serialize_data(['shift', 'reverberation'])
    pass


if __name__ == "__main__":
    main()