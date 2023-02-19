import numpy as np
from data_handling import read_captions, preprocess_captions
from aux_functions import get_word_from_one_hot_encoding
from torch.utils.data import Dataset


class CsvDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.captions = read_captions()
        self.one_hot_captions, self.uniques = preprocess_captions(self.captions)

    def __getitem__(self, item):
        # Load the desired caption
        caption = self.one_hot_captions[item]

        # Return all but the last element as input
        x = caption[:-1]

        # Return the input shifted by one as the desired output
        y = caption[1:]
        return np.array(x), np.array(y)

    def __len__(self):
        return len(self.captions)

    def get_uniques(self):
        # This is for checking the actual words corresponding to the one-hot codes
        # with the get_word_from_one_hot_encoding
        return self.uniques


