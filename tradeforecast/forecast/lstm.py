import torch.nn.functional as F
from torch import nn
import torch

from .base import BaseModel

class LSTM(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        LSTM neural network
        kwargs must include: ['input_size': int, 'hidden_size': int, 'n_LSTM': int,
                             'output_size': int, 'batch_first': bool, 'dropout': float]
        """
        super().__init__()
        if kwargs:
            for arg in kwargs.keys():
                setattr(self, arg, kwargs[arg])
        self.lstm_net = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_LSTM, batch_first=self.batch_first, dropout=self.dropout)
        self.linear_net = nn.Sequential()
        self.linear_net.add_module('Linear_0', nn.Linear(in_features=self.hidden_size, out_features=128))
        self.linear_net.add_module('Linear_1', nn.Linear(in_features=128, out_features=self.output_size))
        pass

    def forward(self, x: torch.Tensor, forecast: int):

        pass

    def train_model(self):
        self.train()
        return
    
    @torch.inference_mode
    def test_model(self):
        #assert torch.is_inference_mode_enabled(), "torch is not in inference_mode!"
        print(self.input_size)
        self.eval()
        return
    
    @torch.inference_mode
    def predict(self):
        #assert torch.is_inference_mode_enabled(), "torch is not in inference_mode!"
        return


