import torch
import numpy as np
from crnn_system import MyCRNNSystem
from copy import deepcopy
from torch import Tensor, rand, cuda, no_grad, flatten, float32
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from typing import Union, MutableMapping, Optional, Tuple
from sed_data_loading import get_dataset, get_data_loader
from pathlib import Path


def main():
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Setup hyper-parameters

    epochs = 300
    CRNN_params = {
        "cnn_channels_out_1": 8,
        "cnn_kernel_1": 3,
        "cnn_stride_1": 1,
        "cnn_padding_1": 1,
        "pooling_kernel_1": (1, 3),
        "pooling_stride_1": 1,
        "cnn_channels_out_2": 4,
        "cnn_kernel_2": 3,
        "cnn_stride_2": 1,
        "cnn_padding_2": 1,
        "pooling_kernel_2": (1, 3),
        "pooling_stride_2": 1,
        "rnn_input_size": 144,
        "rnn_hidden_size": 4,
        "classifier_input_features": 4,
        "output_classes": 6,
        "dropout": 0.2
    }

    # Setup data loaders
    training_loader = get_data_loader('training', batch_size=4, shuffle=True)
    validation_loader = get_data_loader('validation', batch_size=4, shuffle=True)
    testing_loader = get_data_loader('testing', batch_size=4, shuffle=False)

    # Create CNN
    CRNN = MyCRNNSystem(**CRNN_params)
    CRNN = CRNN.to(device)

    optimizer = Adam(params=CRNN.parameters(), lr=1e-3)
    loss_function = BCEWithLogitsLoss()

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 15
    patience_counter = 0

    best_model = None

    # Training
    for epoch in range(epochs):

        # Lists to hold the corresponding losses of each epoch.
        epoch_loss_training = []
        epoch_loss_validation = []

        CRNN.train()

        # Go over each batch.
        for data in training_loader:
            optimizer.zero_grad()
            x, y = data
            x = x.to(float32)
            y = y.to(float32)
            x = x.to(device)
            y = y.to(device)
            y_hat = CRNN(x)
            loss = loss_function(input=y_hat, target=y)
            loss.backward()
            optimizer.step()
            epoch_loss_training.append(loss.item())

        CRNN.eval()

        with no_grad():
            for data in validation_loader:
                x, y = data
                x = x.to(float32)
                y = y.to(float32)
                x = x.to(device)
                y = y.to(device)
                y_hat = CRNN(x)
                loss = loss_function(input=y_hat, target=y)
                epoch_loss_validation.append(loss.item())

        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()

        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(CRNN.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                # Process similar to validation.
                print('Starting testing', end=' | ')
                testing_loss = []
                CRNN.eval()
                with no_grad():
                    for data in testing_loader:
                        x, y = data
                        x = x.to(float32)
                        y = y.to(float32)
                        x = x.to(device)
                        y = y.to(device)
                        y_hat = CRNN(x)
                        loss = loss_function(input=y_hat, target=y)
                        testing_loss.append(loss.item())

                testing_loss = np.array(testing_loss).mean()
                print(f'Testing loss: {testing_loss:7.4f}')
                break
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f}')


if __name__ == '__main__':
    main()