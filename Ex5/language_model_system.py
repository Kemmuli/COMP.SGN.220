import torch
from torch.nn import GRU, Softmax, Tanh, Linear, Module, Sequential
from torch import Tensor


class LanguageSystem(Module):

    def __init__(self,
                 linear_in: int,
                 linear_out: int,
                 n_hidden: int,
                 ) -> None:
        super().__init__()

        self.linear_block = Sequential(Linear(in_features=linear_in, out_features=linear_out),
                                       Tanh())
        self.gru_block = Sequential(GRU(input_size=linear_out, hidden_size=n_hidden, num_layers=3))

        self.clf = Sequential(Linear(in_features=n_hidden, out_features=linear_in),
                              Softmax(dim=2))

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_block(x)
        x, _ = self.gru_block(x)
        x = self.clf(x)
        return x


def main():
    pass


if __name__ == '__main__':
    main()
