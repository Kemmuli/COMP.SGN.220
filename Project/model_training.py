from copy import deepcopy
import os

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from genre_clf import GenreClf
from dataset import GenreDataset
from plotting import plot_confusion_matrix

# Make experiment tracking manageable
torch.manual_seed(52)
torch.backends.cudnn.deterministic = True

def main():
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define data directory and access label names
    dir = './genres/'
    labels = [f.name for f in os.scandir(dir) if f.is_dir()]

    # Define training parameters
    config = {
        "epochs": 30,
        "learning_rate": 1e-4,
        "batch_size": 8
    }

    # Define model parameters
    n_mels = 80
    model_params = {
        'cnn_channels_out_1': 32,
        'cnn_kernel_1': 3,
        'cnn_stride_1': 1,
        'cnn_padding_1': 0,
        'pooling_kernel_1': 2,
        'pooling_stride_1': 2,
        'cnn_channels_out_2': 32,
        'cnn_kernel_2': 3,
        'cnn_stride_2': 1,
        'cnn_padding_2': 0,
        'pooling_kernel_2': 2,
        'pooling_stride_2': 2,
        'cnn_channels_out_3': 64,
        'cnn_kernel_3': 3,
        'cnn_stride_3': 1,
        'cnn_padding_3': 0,
        'pooling_kernel_3': 2,
        'pooling_stride_3': 2,
        'fc_out_1': 128,
        'clf_output_classes': len(labels),
        'dropout_conv_1': 0.,
        'dropout_conv_2': 0.,
        'dropout_conv_3': 0.,
        'dropout_fc_1': 0.4,
    }

    # Load datasets and generate iterators
    training_dataset = GenreDataset(data_dir='training', n_mels=n_mels)
    training_iterator = DataLoader(dataset=training_dataset,
                                   batch_size=config['batch_size'],
                                   shuffle=True,
                                   drop_last=False)

    test_dataset = GenreDataset(data_dir='testing', n_mels=n_mels)
    test_iterator = DataLoader(dataset=test_dataset,
                               batch_size=config['batch_size'],
                               shuffle=False,
                               drop_last=False)

    # Instantiate model
    model = GenreClf(**model_params)
    model = model.to(device)

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Start training loop
    for epoch in range(config['epochs']):
        epoch_loss_train = []
        epoch_acc_train = 0

        model.train()

        for i, (x, y) in enumerate(training_iterator):
            # Zero out gradients, move data to device and calculate predictions
            optimizer.zero_grad()
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            y_hat = model(x)

            # Calculate loss, backpropagate and update weights
            loss = loss_function(input=y_hat, target=y)
            loss.backward()
            optimizer.step()

            # Record loss and accuracy for each batch
            epoch_loss_train.append(loss.item())
            epoch_acc_train += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().item()

        # Calculate average loss and accuracy for the epoch
        epoch_loss_train = np.array(epoch_loss_train).mean()

        print(f"epoch {epoch} Training loss {epoch_loss_train:7.4f} acc {epoch_acc_train/len(training_dataset):7.4f}%")

    # Start testing and initialize some variables
    print('Starting testing', end=' | ')
    testing_loss = []
    epoch_acc_test = 0
    model.eval()
    y_pred = torch.zeros(len(test_dataset), len(labels))
    y_true = torch.zeros(len(test_dataset), len(labels))

    # Disable gradient computation since we are just testing
    with torch.no_grad():
        # Loop over the test set
        for i, (x, y) in enumerate(test_iterator):
            # Move input and target to the device
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            # Save the true labels for later use
            y_true[i * config['batch_size']:i * config['batch_size'] + config['batch_size'], :] = y
            # Compute the predicted labels and loss
            y_hat = model(x)
            loss = loss_function(input=y_hat, target=y)
            # Save the predicted labels and testing loss
            y_pred[i * config['batch_size']:i * config['batch_size'] + config['batch_size'], :] = y_hat
            testing_loss.append(loss.item())
            # Compute the testing accuracy
            epoch_acc_test += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().item()

    # Compute the confusion matrix and plot it
    cm = confusion_matrix(y_true.argmax(dim=1), y_pred.argmax(dim=1))
    plot_confusion_matrix(cm, classes=labels)

    # Compute the mean testing loss and print the testing accuracy
    testing_loss = np.array(testing_loss).mean()
    print(f'Testing loss: {testing_loss:.4f} acc {epoch_acc_test / len(test_dataset):7.4f}%')

    # Save the model's state dictionary to a file
    torch.save(model.state_dict(), 'genre-classifier.pt')


if __name__ == '__main__':
    main()

# EOF
