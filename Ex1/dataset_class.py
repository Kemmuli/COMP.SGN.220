#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

from torch.utils.data import Dataset
import numpy


__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']


class MyDataset(Dataset):

    def __init__(self, ) -> None:
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, item: int) -> Tuple[numpy.ndarray, int]:
        pass

# EOF

