#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Tuple
from torch import Tensor
from torch.nn import Module, Conv2d, MaxPool2d, BatchNorm2d, ReLU, Linear, \
    Sequential, Dropout2d, Softmax, LazyLinear, Dropout


class GenreClf(Module):

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
                 cnn_channels_out_3: int,
                 cnn_kernel_3: Union[Tuple[int, int], int],
                 cnn_stride_3: Union[Tuple[int, int], int],
                 cnn_padding_3: Union[Tuple[int, int], int],
                 pooling_kernel_3: Union[Tuple[int, int], int],
                 pooling_stride_3: Union[Tuple[int, int], int],
                 fc_out_1: int,
                 clf_output_classes: int,
                 dropout_conv_1: float,
                 dropout_conv_2: float,
                 dropout_conv_3: float,
                 dropout_fc_1: float,
                 ) -> None:

        super().__init__()

        # Define the first convolutional block as a sequential module
        self.conv1 = Sequential(
            # 2D convolutional layer with specified input and output channels, kernel size, stride, and padding
            Conv2d(in_channels=1,
                   out_channels=cnn_channels_out_1,
                   kernel_size=cnn_kernel_1,
                   stride=cnn_stride_1,
                   padding=cnn_padding_1),
            # Rectified linear unit activation function
            ReLU(),
            # Batch normalization layer
            BatchNorm2d(cnn_channels_out_1),
            # Max pooling layer with specified kernel size and stride
            MaxPool2d(kernel_size=pooling_kernel_1,
                      stride=pooling_stride_1),
            # Dropout layer for regularization
            Dropout2d(dropout_conv_1)
        )

        # Define the second convolutional block as a sequential module
        self.conv2 = Sequential(
            # 2D convolutional layer with specified input and output channels, kernel size, stride, and padding
            Conv2d(in_channels=cnn_channels_out_1,
                   out_channels=cnn_channels_out_2,
                   kernel_size=cnn_kernel_2,
                   stride=cnn_stride_2,
                   padding=cnn_padding_2),
            # Rectified linear unit activation function
            ReLU(),
            # Batch normalization layer
            BatchNorm2d(cnn_channels_out_2),
            # Max pooling layer with specified kernel size and stride
            MaxPool2d(kernel_size=pooling_kernel_2,
                      stride=pooling_stride_2),
            # Dropout layer for regularization
            Dropout2d(dropout_conv_2)
        )

        # Define the third convolutional block as a sequential module
        self.conv3 = Sequential(
            # 2D convolutional layer with specified input and output channels, kernel size, stride, and padding
            Conv2d(in_channels=cnn_channels_out_2,
                   out_channels=cnn_channels_out_3,
                   kernel_size=cnn_kernel_3,
                   stride=cnn_stride_3,
                   padding=cnn_padding_3),
            # Rectified linear unit activation function
            ReLU(),
            # Batch normalization layer
            BatchNorm2d(cnn_channels_out_3),
            # Max pooling layer with specified kernel size and stride
            MaxPool2d(kernel_size=pooling_kernel_3,
                      stride=pooling_stride_3),
            # Dropout layer for regularization
            Dropout2d(dropout_conv_3)
        )

        # Define the first fully connected block as a sequential module
        self.fc1 = Sequential(
            # Linear layer with specified output features
            LazyLinear(out_features=fc_out_1),
            # Rectified linear unit activation function
            ReLU(),
            # Dropout layer for regularization
            Dropout(dropout_fc_1)
        )

        # Define the classification block as a sequential module
        self.clf = Sequential(
            # Linear layer with specified input and output features
            Linear(in_features=fc_out_1, out_features=clf_output_classes),
            # Softmax activation function for classification probabilities
            Softmax(dim=1))

    def forward(self,
                x: Tensor) -> Tensor:
        # Check if input tensor is 4D (batch_size, channels, height, width).
        # If it is not, add an extra dimension at index 1 to make it 4D.
        h = x if x.ndimension() == 4 else x.unsqueeze(1)

        # Pass the input tensor through the convolutional layers, followed by ReLU activation,
        # batch normalization, max pooling, and dropout in each layer.
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)

        # Reshape the output tensor to have shape (batch_size, num_features),
        # where num_features is the product of the dimensions of the output tensor from the
        # last convolutional layer.
        h = h.view(h.size()[0], -1)

        # Pass the flattened output tensor through a fully connected layer,
        # followed by ReLU activation and dropout.
        h = self.fc1(h)

        # Pass the output of the fully connected layer through the final classification layer,
        # which produces class probabilities using the Softmax activation function.
        return self.clf(h)


if __name__ == '__main__':
    pass
