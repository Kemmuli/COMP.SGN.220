# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:23:48 2022

@author: qcirma
"""

import torch
import torch.nn as nn
import utils
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from my_cnn_system import MyCNNSystem
from getting_and_init_the_data import get_data_loader, get_dataset
from copy import deepcopy
from sklearn.metrics import confusion_matrix, accuracy_score


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define hyper-parameters to be used.
    batch_size = 4
    epochs = 30
    learning_rate = 1e-4

    cnn_1_channels = 16
    cnn_1_kernel = 5
    cnn_1_stride = 2
    cnn_1_padding = 2

    pooling_1_kernel = 3
    pooling_1_stride = 1
    # Output size: b_size x 16 x 214 x 18

    cnn_2_channels = 32
    cnn_2_kernel = 5
    cnn_2_stride = 2
    cnn_2_padding = 2

    pooling_2_kernel = 3
    pooling_2_stride = 2
    # Output size: b_size x 32 x 53 x 4

    classifier_input = 6784
    classifier_output = 4

    dropout = .20

    # Instantiate our DNN
    model = MyCNNSystem(cnn_channels_out_1=cnn_1_channels,
                        cnn_kernel_1=cnn_1_kernel,
                        cnn_stride_1=cnn_1_stride,
                        cnn_padding_1=cnn_1_padding,
                        pooling_kernel_1=pooling_1_kernel,
                        pooling_stride_1=pooling_1_stride,
                        cnn_channels_out_2=cnn_2_channels,
                        cnn_kernel_2=cnn_2_kernel,
                        cnn_stride_2=cnn_2_stride,
                        cnn_padding_2=cnn_2_padding,
                        pooling_kernel_2=pooling_2_kernel,
                        pooling_stride_2=pooling_2_stride,
                        classifier_input_features=classifier_input,
                        output_classes=classifier_output,
                        dropout=dropout)
    model = model.to(device)

    data_path = Path("./")
    labels = ['rain', 'sea_waves', 'chainsaw', 'helicopter']

    training_dataset = get_dataset('training')
    train_iterator = get_data_loader(dataset=training_dataset, batch_size=batch_size,
                                     shuffle=True, drop_last=True)
    tr_data_sz = len(training_dataset)

    validation_dataset = get_dataset('validation')
    val_iterator = get_data_loader(dataset=get_dataset('validation'), batch_size=batch_size,
                                   shuffle=True, drop_last=True)
    val_data_sz = len(validation_dataset)
    testing_dataset = get_dataset('testing')
    test_iterator = get_data_loader(dataset=testing_dataset, batch_size=batch_size,
                                    shuffle=False, drop_last=True)
    test_data_sz = len(testing_dataset)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 10
    patience_counter = 0

    best_model = None
    for epoch in range(epochs):
        epoch_loss_training = []
        epoch_loss_validation = []
        model.train()

        for i, batch in enumerate(train_iterator):
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            y_hat = model(x)
            loss = loss_function(input=y_hat, target=y)
            loss.backward()
            optimizer.step()
            epoch_loss_training.append(loss.item())

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_iterator):
                x, y = batch
                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float)
                y_hat = model(x)
                loss = loss_function(input=y_hat, target=y)
                epoch_loss_validation.append(loss.item())

        # Calculate mean losses.
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()

        print(f"epoch {epoch} Training loss {epoch_loss_training:.4f}")
        print(f"Validation loss {epoch_loss_validation:.4f}")

        # Check early stopping conditions.
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(model.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        # If we have to stop, do the testing.
        if (patience_counter >= patience) or (epoch==epochs-1):
            print('\nExiting training', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                print('Starting testing', end=' | ')
                testing_loss = []
                model.eval()
                y_pred = torch.zeros(test_data_sz, 4)
                y_true = torch.zeros(test_data_sz, 4)

                with torch.no_grad():
                    for i, batch in enumerate(test_iterator):
                        x, y = batch
                        x = x.to(device=device, dtype=torch.float)
                        y = y.to(device=device, dtype=torch.float)
                        y_true[i*batch_size:i*batch_size+batch_size, :] = y
                        y_hat = model(x)
                        loss = loss_function(input=y_hat, target=y)
                        y_pred[i*batch_size:i*batch_size+batch_size, :] = y_hat
                        testing_loss.append(loss.item())

                testing_loss = np.array(testing_loss).mean()
                cm = confusion_matrix(y_true=y_true.argmax(dim=1), y_pred=y_pred.argmax(dim=1))
                testing_acc = cm.diagonal().sum() / test_data_sz
                utils.plot_confusion_matrix(cm, labels)
                print(f'Size of test dataset: {test_data_sz}')
                print(f'Testing loss: {testing_loss:.4f} acc: {testing_acc:.4f}')
                break

            # All tests ran with normal features.
            # Accuracy without augmentations: 46.88%
            # Accuracy with only white noise, 5% noise: 53.12%
            # Accuracy white noise and pitch shift: 46.53%, 50.35% - pitch shifting lowered the result
            # Accuracy with reverberation and noise: 44.44% - reverberation seems to lower the result also
            # Accuracy with noise and specAugmentation: 42.36% - specAugmentation seems to lower the result even more
            # Accuracy with all the augmentations: 43.75%


if __name__ == '__main__':
    main()
