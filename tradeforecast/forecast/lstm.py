import torch.nn.functional as F
from torch import nn, Tensor
import torch

from .base import RNNBase

class LSTM(RNNBase):
    def __init__(self, **kwargs):
        """
        LSTM neural network
        kwargs must include: 
        """
        self.input_size: int = kwargs.get('input_size')
        self.hidden_size: int = kwargs.get('hidden_size')
        self.n_LSTM: int = kwargs.get('n_LSTM')
        self.fc_out_size: list = kwargs.get('fc_out_size')
        self.output_size: int = kwargs.get('output_size')
        self.dropout: float = kwargs.get('dropout')
        self.n_fc: int = len(self.fc_out_size) + 1    # self.n_fc --> number of fully connected layers + output_layer
        self.fc_out_size.insert(0, self.hidden_size)
        self.fc_out_size.append(self.output_size)
        super().__init__()
        self.device = torch.device(self.get_device_type())
        print(f"The device is set to --- '{self.device}'")
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_LSTM, batch_first=True, dropout=self.dropout, device=self.device)
        self.fc_linear = nn.Sequential()
        for i in range(self.n_fc):
            self.fc_linear.add_module(f"Linear_{i+1}", nn.Linear(in_features=self.fc_out_size[i], out_features=self.fc_out_size[i+1], device=self.device))
        pass

    def forward(self, x: Tensor):
        pass


