import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from dataset_class import CsvDataset
from torch.utils.data import DataLoader
from language_model_system import LanguageSystem
from aux_functions import get_word_from_one_hot_encoding


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    batch_sz = 1

    dataset = CsvDataset()
    data_loader = DataLoader(dataset=dataset, batch_size=batch_sz, shuffle=False)
    uniques = dataset.get_uniques()
    linear_in = len(uniques)
    linear_out = 64
    n_hidden = 64
    epochs = 200
    lr = 1e-4

    model = LanguageSystem(linear_in=linear_in, linear_out=linear_out, n_hidden=n_hidden).to(device)

    loss_function = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        print('Epoch: ', epoch, '\n')
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            y_hat = model(x)

            # Show the true value and prediction for the first 10 captions.
            if epoch % 20 and 0 <= i <= 9:
                true = y.cpu().detach().numpy()
                predict = y_hat.cpu().detach().numpy()
                true = np.squeeze(true, axis=0)
                predict = np.squeeze(predict, axis=0)
                print('Caption: ', i, '\n')
                for t, p in zip(true, predict):
                    print(get_word_from_one_hot_encoding(t, uniques), ' : true')
                    p_one_hot = np.zeros_like(p)
                    p_one_hot[np.argmax(p)] = 1
                    print(get_word_from_one_hot_encoding(p_one_hot, uniques), ' : predicted \n')
                print('\n')

            loss = loss_function(input=y_hat, target=y)
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main()
