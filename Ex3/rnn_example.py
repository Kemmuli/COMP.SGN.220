import torch
from torch.nn import MSELoss, RNNBase, Sigmoid, LazyLinear, LSTM, GRU, BCEWithLogitsLoss
from torch.nn.utils.rnn import PackedSequence
from torch import Tensor, rand, bernoulli
from typing import Union, Tuple, Optional
from torch.optim import Adam

import numpy as np


class MyRNN(RNNBase):

    def __init__(self,
                 mode: str,
                 input_size: Union[int, Tuple[int, int]],
                 hidden_size: Union[int, Tuple[int, int]],
                 hidden_size_2: Union[int, Tuple[int, int]],
                 n_classes: int) -> None:
        """

        :param mode: mode of the RNN, "LSTM", "GRU", "RNN_TANH", "RNN_RELU"
        :param input_size:
        :param hidden_size:
        :param n_classes:
        """

        super().__init__(mode=mode, input_size=input_size, hidden_size=hidden_size)

        if self.mode == 'LSTM':
            self.rl_1 = LSTM(input_size=input_size, hidden_size=hidden_size)
            self.rl_2 = LSTM(input_size=hidden_size, hidden_size=hidden_size_2)
        else:
            self.rl_1 = GRU(input_size=input_size, hidden_size=hidden_size)
            self.rl_2 = GRU(input_size=hidden_size, hidden_size=hidden_size_2)
        self.lin_layer = LazyLinear(out_features=n_classes)
        self.non_lin = Sigmoid()

    def forward(self,
                input: Union[Tensor, PackedSequence],
                hx: Optional[Tensor] = None) -> Tuple[Union[Tensor, PackedSequence], Tensor]:

        # Hidden-to-Hidden vectors seem to be the same for GRU and LSTM, LSTM has a bigger gate size than GRU
        x, t = self.rl_1(input)
        x, t = self.rl_2(x)
        x = self.lin_layer(x)

        return self.non_lin(x)


def task1():

    # Define hyper-parameters.
    epochs = 100
    n_examples = 20
    T = 64
    input_size = 8
    hs_1 = 4
    hs_2 = 2
    Z = 2

    lstm_rnn = MyRNN(mode='LSTM',
                     input_size=input_size,
                     hidden_size=hs_1,
                     hidden_size_2=hs_2,
                     n_classes=Z)

    gru_rnn = MyRNN(mode='GRU',
                    input_size=input_size,
                    hidden_size=hs_1,
                    hidden_size_2=hs_2,
                    n_classes=Z)

    lstm_optim = Adam(params=lstm_rnn.parameters(), lr=1e-3)
    gru_optim = Adam(params=gru_rnn.parameters(), lr=1e-3)

    loss_function = MSELoss()
    loss_function_2 = MSELoss()

    x_train = rand(n_examples, T, input_size)
    y_train = rand(n_examples, T, Z)

    for epoch in range(epochs):

        lstm_rnn.train()
        gru_rnn.train()
        epoch_loss_lstm = []
        epoch_loss_gru = []

        for i in range(0, n_examples):

            lstm_optim.zero_grad()
            gru_optim.zero_grad()

            x_input = x_train[i, :, :]
            x_input = x_input.unsqueeze(0)
            y_output = y_train[i, :, :]
            y_output = y_output.unsqueeze(0)

            y_lstm = lstm_rnn(x_input)
            y_gru = gru_rnn(x_input)

            loss_lstm = loss_function(input=y_lstm, target=y_output)
            loss_gru = loss_function_2(input=y_gru, target=y_output)

            loss_lstm.backward()
            loss_gru.backward()

            lstm_optim.step()
            gru_optim.step()

            epoch_loss_lstm.append(loss_lstm.item())
            epoch_loss_gru.append(loss_gru.item())

        epoch_loss_lstm = np.array(epoch_loss_lstm).mean()
        epoch_loss_gru = np.array(epoch_loss_gru).mean()

        print(f'Epoch: {epoch:03d} | '
              f'Mean LSTM loss: {epoch_loss_lstm:7.4f} | '
              f'Mean GRU loss {epoch_loss_gru:7.4f}')


class KHotRNN(RNNBase):

    def __init__(self,
                 mode: str,
                 input_size: Union[int, Tuple[int, int]],
                 hidden_size: Union[int, Tuple[int, int]],
                 hidden_size_2: Union[int, Tuple[int, int]],
                 n_classes: int) -> None:
        """

        :param mode: mode of the RNN, "LSTM", "GRU", "RNN_TANH", "RNN_RELU"
        :param input_size:
        :param hidden_size:
        :param n_classes:
        """

        super().__init__(mode=mode, input_size=input_size, hidden_size=hidden_size)

        if self.mode == 'LSTM':
            self.rl_1 = LSTM(input_size=input_size, hidden_size=hidden_size)
            self.rl_2 = LSTM(input_size=hidden_size, hidden_size=hidden_size_2)
        else:
            self.rl_1 = GRU(input_size=input_size, hidden_size=hidden_size)
            self.rl_2 = GRU(input_size=hidden_size, hidden_size=hidden_size_2)
        self.lin_layer = LazyLinear(out_features=n_classes)

    def forward(self,
                input: Union[Tensor, PackedSequence],
                hx: Optional[Tensor] = None) -> Tuple[Union[Tensor, PackedSequence], Tensor]:

        x, t = self.rl_1(input)
        x, t = self.rl_2(x)
        x = self.lin_layer(x)

        return x


def task2():

    # Define hyper-parameters.
    epochs = 100
    n_examples = 20
    T = 64
    input_size = 8
    hs_1 = 4
    hs_2 = 2
    Z = 4

    lstm_rnn = KHotRNN(mode='LSTM',
                       input_size=input_size,
                       hidden_size=hs_1,
                       hidden_size_2=hs_2,
                       n_classes=Z)

    lstm_optim = Adam(params=lstm_rnn.parameters(), lr=1e-3)

    loss_function = BCEWithLogitsLoss()

    x_train = rand(n_examples, T, input_size)
    y_train = rand(n_examples, T, Z)
    y_train = bernoulli(y_train)

    for epoch in range(epochs):

        lstm_rnn.train()
        epoch_loss_lstm = []
        epoch_loss_gru = []

        for i in range(0, n_examples):

            lstm_optim.zero_grad()

            x_input = x_train[i, :, :]
            x_input = x_input.unsqueeze(0)
            y_output = y_train[i, :, :]
            y_output = y_output.unsqueeze(0)

            y_lstm = lstm_rnn(x_input)

            loss_lstm = loss_function(input=y_lstm, target=y_output)

            loss_lstm.backward()

            lstm_optim.step()

            epoch_loss_lstm.append(loss_lstm.item())

        epoch_loss_lstm = np.array(epoch_loss_lstm).mean()

        print(f'Epoch: {epoch:03d} | '
              f'Mean LSTM loss: {epoch_loss_lstm:7.4f}')


def main():
    task1()
    task2()


if __name__ == '__main__':
    main()

# EOF
