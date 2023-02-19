from typing import Tuple, Optional, Union, Dict
from pickle import load as pickle_load
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy
from file_io import get_files_from_dir_with_pathlib


class MyDataset(Dataset):

    def __init__(self,
                 data_dir: Union[str, Path],
                 data_parent_dir: Optional[str] = '',
                 key_features: Optional[str] = 'features',
                 key_labels: Optional[str] = 'labels',
                 load_into_memory: Optional[bool] = True) \
            -> None:
        """An example of an object of labels torch.utils.data.Dataset

        :param data_dir: Directory to read data from.
        :type data_dir: str
        :param data_parent_dir: Parent directory of the data, defaults\
                                to ``.
        :type data_parent_dir: str
        :param key_features: Key to use for getting the features,\
                             defaults to `features`.
        :type key_features: str
        :param key_labels: Key to use for getting the labels, defaults\
                          to `labels`.
        :type key_labels: str
        :param load_into_memory: Load the data into memory? Default to True
        :type load_into_memory: bool
        """
        super().__init__()
        data_path = Path(data_parent_dir, data_dir)
        self.files = get_files_from_dir_with_pathlib(data_path)
        self.load_into_memory = load_into_memory
        self.key_features = key_features
        self.key_labels = key_labels

        if self.load_into_memory:
            for i, a_file in enumerate(self.files):
                self.files[i] = self._load_file(a_file)

    @staticmethod
    def _load_file(file_path: Path) \
            -> Dict[str, Union[int, numpy.ndarray]]:
        """Loads a file using pathlib.Path

        :param file_path: File path.
        :type file_path: pathlib.Path
        :return: The file.
        :rtype: dict[str, int|numpy.ndarray]
        """
        with file_path.open('rb') as f:
            return pickle_load(f)

    def __len__(self) \
            -> int:
        """Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return len(self.files)

    def __getitem__(self,
                    item: int) \
            -> Tuple[numpy.ndarray, int]:
        """Returns an item from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Features and labels of the item.
        :rtype: (numpy.ndarray, int)
        """
        if self.load_into_memory:
            the_item: Dict[str, Union[int, numpy.ndarray]] = self.files[item]
        else:
            the_item = self._load_file(self.files[item])

        return the_item[self.key_features], the_item[self.key_labels]


def get_dataset(data_dir: Union[str, Path],
                data_parent_dir: Optional[str] = '',
                key_features: Optional[str] = 'features',
                key_labels: Optional[str] = 'labels',
                load_into_memory: Optional[bool] = True) \
        -> MyDataset:
    """Creates and returns a dataset, according to `MyDataset` labels.

    :param data_dir: Directory to read data from.
    :type data_dir: str|pathlib.Path
    :param data_parent_dir: Parent directory of the data, defaults\
                            to ``.
    :type data_parent_dir: str
    :param key_features: Key to use for getting the features,\
                         defaults to `features`.
    :type key_features: str
    :param key_labels: Key to use for getting the labels, defaults\
                      to `labels`.
    :type key_labels: str
    :param load_into_memory: Load the data into memory? Default to True
    :type load_into_memory: bool
    :return: Dataset.
    :rtype: dataset_labels.MyDataset
    """
    return MyDataset(data_dir=data_dir,
                     data_parent_dir=data_parent_dir,
                     key_features=key_features,
                     key_labels=key_labels,
                     load_into_memory=load_into_memory)


def get_data_loader(data_type: str,
                    batch_size: int,
                    shuffle: bool) \
        -> DataLoader:
    """Creates and returns a data loader.

    :param data_type: Dataset type to use (training, validation, testing)
    :type data_type: str
    :param batch_size: Batch size to use.
    :type batch_size: int
    :param shuffle: Shuffle the data?
    :type shuffle: bool
    :return: Data loader, using the specified dataset.
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(dataset=get_dataset(data_type), batch_size=batch_size, shuffle=shuffle)


def main():
    pass


if __name__ == '__main__':
    main()


