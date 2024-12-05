from torch import nn, Tensor
import torch

from .base import RNNBase

class LSTM(RNNBase):
    def __init__(self, seed: int=42, **kwargs):
        """LSTM neural network"""
        self.__set_global_seed__(seed)
        self.input_size: int = kwargs.get('input_size')
        self.hidden_size: int = kwargs.get('hidden_size')
        self.n_LSTM: int = kwargs.get('n_LSTM')
        self.bidirectional: bool = kwargs.get('bidirectional')
        self.n_LSTM_dim = 2 * self.n_LSTM if self.bidirectional else self.n_LSTM
        self.fc_out_size: list = kwargs.get('fc_out_size')
        self.output_size: int = kwargs.get('output_size')
        self.dropout: float = kwargs.get('dropout')
        self.n_fc: int = len(self.fc_out_size) + 1    # self.n_fc --> number of fully connected layers + output_layer
        self.fc_out_size.insert(0, 2 * self.hidden_size if self.bidirectional else self. hidden_size)
        self.fc_out_size.append(self.output_size)
        super().__init__()
        self.device = torch.device(self.get_device_type())
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_LSTM, bidirectional=self.bidirectional, batch_first=True, dropout=self.dropout, device=self.device)
        self.fc_linear = nn.Sequential()
        for i in range(self.n_fc):
            self.fc_linear.add_module(f"Linear_{i+1}", nn.Linear(in_features=self.fc_out_size[i], out_features=self.fc_out_size[i+1], device=self.device))

    def __repr__(self) -> str:
        name = 'biLSTM' if self.bidirectional else 'LSTM'
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'{name}({n_params},{self.input_size},{self.output_size})-{self.device}'

    def forward(self, x: Tensor) -> Tensor:
        batch_len = x.size(0)   # since batch_first
<<<<<<< Updated upstream
        # Initialize hidden and cell states with zeros (optional)
        n_LSTM = 2 * self.n_LSTM if self.bidirectional else self.n_LSTM
        h0 = torch.zeros(n_LSTM, batch_len, self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(n_LSTM, batch_len, self.hidden_size).requires_grad_().to(self.device)

        x, _ = self.lstm(x, (h0, c0))
        x = self.fc_linear(x[:, -1, :])
        return x
=======
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.n_LSTM_dim, batch_len, self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(self.n_LSTM_dim, batch_len, self.hidden_size).requires_grad_().to(self.device)
        _, (out, _) = self.lstm(x, (h0, c0))
        output = self.fc_linear(out[-1])
        return output
>>>>>>> Stashed changes


