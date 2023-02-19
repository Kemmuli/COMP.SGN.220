#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional
from torch import Tensor, reshape
from torch.optim import Adam
from torch.nn import Module, Conv2d, MaxPool2d, \
    BatchNorm2d, ReLU, Linear, Sequential, Dropout2d, \
    RNNBase, Sigmoid, LazyLinear, LSTM, GRU, BCEWithLogitsLoss
from torch.nn.utils.rnn import PackedSequence
from pickle import load as pickle_load
from file_io import get_files_from_dir_with_pathlib
from sed_data_loading import get_data_loader, get_dataset, MyDataset


class MyCRNNSystem(Module):

    def __init__(self,
                 cnn_channels_out_1: int,
                 cnn_kernel_1: Union[Tuple[int, int], int],
                 cnn_stride_1: Union[Tuple[int, int], int],
                 cnn_padding_1: Union[Tuple[int, int], int],
                 pooling_kernel_1: Union[Tuple[int, int], int],
                 pooling_stride_1: Union[Tuple[int, int], int],
                 cnn_channels_out_2: int,
                 cnn_kernel_2: Union[Tuple[int, int], int],
                 cnn_stride_2: Union[Tuple[int, int], int],
                 cnn_padding_2: Union[Tuple[int, int], int],
                 pooling_kernel_2: Union[Tuple[int, int], int],
                 pooling_stride_2: Union[Tuple[int, int], int],
                 rnn_input_size: int,
                 rnn_hidden_size: int,
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

        self.rnn_block = GRU(input_size=rnn_input_size,
                             hidden_size=rnn_hidden_size)

        self.classifier = Linear(in_features=classifier_input_features,
                                 out_features=output_classes)

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
        cs = h.size()
        h = reshape(h, (cs[0],
                        cs[2],
                        cs[3],
                        cs[1]))
        h = h.view(h.size()[0], h.size()[1], -1)
        h, _ = self.rnn_block(h)
        return self.classifier(h)


def main():
    dataset = get_dataset('training')
    # 6 classes
    x, y = dataset[0]
    print(x.shape, y.shape)
    # Here's an example of "carefully selected hyper-parameters"
    # Essentially only need to make sure the padding matches the cnn kernel and that pooling doesn't
    # reduce the time dimension

    ExampleCRNN = MyCRNNSystem(cnn_channels_out_1=4, cnn_kernel_1=3, cnn_stride_1=1, cnn_padding_1=1,
                               pooling_kernel_1=(1, 3), pooling_stride_1=1, cnn_channels_out_2=8,
                               cnn_kernel_2=3, cnn_stride_2=1, cnn_padding_2=1, pooling_kernel_2=(1, 3),
                               pooling_stride_2=1, rnn_input_size=8, rnn_hidden_size=4,
                               classifier_input_features=4, output_classes=6, dropout=0.2)


if __name__ == '__main__':
    main()

# EOF
