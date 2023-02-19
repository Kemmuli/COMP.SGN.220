#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Tuple

from torch import Tensor
from torch.nn import Module, Conv2d, MaxPool2d, \
    BatchNorm2d, ReLU, Linear, Sequential, Dropout2d, Softmax

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['MyCNNSystem']


class MyCNNSystem(Module):

    def __init__(self,
                 cnn_channels_out_1: int,
                 cnn_kernel_1: Union[Tuple[int], int],
                 cnn_stride_1: Union[Tuple[int], int],
                 cnn_padding_1: Union[Tuple[int], int],
                 pooling_kernel_1: Union[Tuple[int], int],
                 pooling_stride_1: Union[Tuple[int], int],
                 cnn_channels_out_2: int,
                 cnn_kernel_2: Union[Tuple[int], int],
                 cnn_stride_2: Union[Tuple[int], int],
                 cnn_padding_2: Union[Tuple[int], int],
                 pooling_kernel_2: Union[Tuple[int], int],
                 pooling_stride_2: Union[Tuple[int], int],
                 classifier_input_features: int,
                 output_classes: int,
                 dropout: float) -> None:
        """MyCNNSystem, using two CNN layers, followed by a ReLU, a batch norm,\
        and a max-pooling process.

        :param cnn_channels_out_1: Output channels of first CNN.
        :type cnn_channels_out_1: int
        :param cnn_kernel_1: Kernel shape of first CNN.
        :type cnn_kernel_1: int|Tuple[int, int]
        :param cnn_stride_1: Strides of first CNN.
        :type cnn_stride_1: int|Tuple[int, int]
        :param cnn_padding_1: Padding of first CNN.
        :type cnn_padding_1: int|Tuple[int, int]
        :param pooling_kernel_1: Kernel shape of first pooling.
        :type pooling_kernel_1: int|Tuple[int, int]
        :param pooling_stride_1: Strides of first pooling.
        :type pooling_stride_1: int|Tuple[int, int]
        :param cnn_channels_out_2: Output channels of second CNN.
        :type cnn_channels_out_2: int
        :param cnn_kernel_2: Kernel shape of second CNN.
        :type cnn_kernel_2: int|Tuple[int, int]
        :param cnn_stride_2: Strides of second CNN.
        :type cnn_stride_2: int|Tuple[int, int]
        :param cnn_padding_2: Padding of second CNN.
        :type cnn_padding_2: int|Tuple[int, int]
        :param pooling_kernel_2: Kernel shape of second pooling.
        :type pooling_kernel_2: int|Tuple[int, int]
        :param pooling_stride_2: Strides of second pooling.
        :type pooling_stride_2: int|Tuple[int, int]
        :param classifier_input_features: Input features to the\
                                          classifier.
        :type classifier_input_features: int
        :param dropout: Dropout to use.
        :type dropout: float
        :param output_classes: Output classes.
        :type output_classes: int
        """
        super().__init__()

        self.block_1 = Sequential(
            Conv2d(in_channels=1,
                   out_channels=cnn_channels_out_1,
                   kernel_size=cnn_kernel_1,
                   stride=cnn_stride_1,
                   padding=cnn_padding_1),
            ReLU(),
            BatchNorm2d(cnn_channels_out_1),
            MaxPool2d(kernel_size=pooling_kernel_1,
                      stride=pooling_stride_1),
            Dropout2d(dropout))

        self.block_2 = Sequential(
            Conv2d(in_channels=cnn_channels_out_1,
                   out_channels=cnn_channels_out_2,
                   kernel_size=cnn_kernel_2,
                   stride=cnn_stride_2,
                   padding=cnn_padding_2),
            ReLU(),
            BatchNorm2d(cnn_channels_out_2),
            MaxPool2d(kernel_size=pooling_kernel_2,
                      stride=pooling_stride_2))

        self.classifier = Linear(in_features=classifier_input_features,
                                 out_features=output_classes)
        self.softmax = Softmax(dim=1)

    def forward(self,
                x: Tensor)\
            -> Tensor:
        """Forward pass.

        :param x: Input features\
                  (shape either `batch x time x features` or\
                  `batch x channels x time x features`).
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        h = x if x.ndimension() == 4 else x.unsqueeze(1)
        h = self.block_1(h)
        h = self.block_2(h)
        h = h.permute(0, 2, 1, 3).contiguous().view(x.size()[0], -1)
        h = self.classifier(h)
        h = self.softmax(h)
        return h


def main():
    pass


if __name__ == '__main__':
    main()

# EOF
