import torch.nn as nn

from tradeforecast import LSTM

kwargs = {'input_size': 10,
          'hidden_size': 4,
          'n_LSTM': 2,
          'n_Linear': 4,
          'output_size': 1,
          'batch_first': True,
          'dropout': 0.2}

lstm_model = LSTM(**kwargs)
assert isinstance(lstm_model, nn.Module), "Not an instance of nn.Module!"
lstm_model.test_model()