from torch import nn, Tensor, optim
import torch.nn.functional as F
import torch

from .base import LitBase

class TFModel(LitBase):
    def __init__(self, seed: int=42, **kwargs):
        """TraedeForecast (TF) Model"""
        super().__init__()
        self.__set_global_seed__(seed)
        self.input_size: int = kwargs.get('input_size')
        self.conv_out_size: int = kwargs.get('conv_out_size')
        self.kernel_size: int = kwargs.get('kernel_size')
        self.hidden_size: int = kwargs.get('hidden_size')
        self.n_LSTM: int = kwargs.get('n_LSTM')
        self.bidirectional: bool = kwargs.get('bidirectional')
        self.fc_out_size: list = kwargs.get('fc_out_size')
        self.output_size: int = kwargs.get('output_size')
        self.dropout: float = kwargs.get('dropout')
        self.criterion = kwargs.get('criterion')
        self.optimizer: optim.Optimizer = kwargs.get('optimizer')
        self.initial_lr: float = kwargs.get('initial_lr')
        self.min_lr: float = kwargs.get('min_lr')
        self.n_fc: int = len(self.fc_out_size) + 1    # self.n_fc --> number of fully connected layers + output_layer
        self.fc_out_size.insert(0, self.hidden_size)
        self.fc_out_size.append(self.output_size)
        self.conv1d = nn.Conv1d(in_channels=self.input_size, out_channels=self.conv_out_size, kernel_size=self.kernel_size, stride=1, padding=0)
        self.bnorm = nn.BatchNorm1d(self.conv_out_size)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size=self.conv_out_size, hidden_size=self.hidden_size, num_layers=self.n_LSTM, bidirectional=self.bidirectional, batch_first=True, dropout=self.dropout)
        self.fc_linear = nn.Sequential()
        for i in range(self.n_fc):
            self.fc_linear.add_module(f"Linear_{i+1}", nn.Linear(in_features=self.fc_out_size[i], out_features=self.fc_out_size[i+1]))
    
    def __repr__(self) -> str:
        name = 'biTF' if self.bidirectional else 'TF'
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'{name}({n_params}_{self.input_size}_{self.output_size})'

    def forward(self, x: Tensor) -> Tensor:
        batch_len = x.size(0)   # since batch_first
        # Initialize hidden and cell states with zeros (optional)
        n_LSTM = 2 * self.n_LSTM if self.bidirectional else self.n_LSTM
        h = torch.zeros(n_LSTM, batch_len, self.hidden_size).requires_grad_().to(self.device)
        c = torch.zeros(n_LSTM, batch_len, self.hidden_size).requires_grad_().to(self.device)

        # Conv1D expects (batch, features, sequence_length)
        x = x.permute(0, 2, 1)  # arrange to (batch, features, sequence)
        x = F.relu(self.conv1d(x))
        x = self.bnorm(x)
        x = self.avg_pool(x)

        x = x.permute(0, 2, 1)  # re-arrange back to (batch, sequence, features)
        _, (h, _) = self.lstm(x, (h, c))
        x = self.fc_linear(h[-1])
        return x

