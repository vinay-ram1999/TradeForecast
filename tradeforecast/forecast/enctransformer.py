from torch import nn, Tensor, optim
import torch

import math

from .base import LitBase
    
class PositionalEncoding(LitBase):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        """ref: https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308/3"""
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).to(self.device)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(self.device)
        pe = torch.zeros(max_len, 1, d_model).to(self.device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EncTransformer(LitBase):
    def __init__(self, seed: int=42, **kwargs):
        """EncTransformer Model"""
        super().__init__()
        self.__set_global_seed__(seed)
        self.input_size: int = kwargs.get('input_size')
        self.nhead: int = kwargs.get('nhead')   # Number of attention heads
        self.d_model: int = kwargs.get('d_model')   # Embedding dimension
        self.num_layers: int = kwargs.get('num_layers')   # Number of transformer encoder layers
        self.output_size: int = kwargs.get('output_size')
        self.dropout: float = kwargs.get('dropout')
        self.criterion = kwargs.get('criterion')
        self.optimizer: optim.Optimizer = kwargs.get('optimizer')
        self.lr: float = kwargs.get('lr')
        self.input_layer = nn.Linear(self.input_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True, dropout=self.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_layer = nn.Linear(self.d_model, self.output_size)
    
    def __repr__(self) -> str:
        return 'EncTransformer'

    def forward(self, x: Tensor):
        x = self.input_layer(x)  # (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model) for PyTorch transformer
        x = self.encoder(x)     # (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, d_model)
        x = self.output_layer(x)  # Predict next N steps for each timestep
        return x[:, -1, :]
