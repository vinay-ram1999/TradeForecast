from torch.utils.data import DataLoader
from torch import nn, optim

from tradeforecast import LSTM

kwargs = {'input_size': 10,
          'hidden_size': 4,
          'n_LSTM': 2,
          'fc_out_size':[32,64,64,32],
          'output_size': 1,
          'dropout': 0.2}

lstm_model = LSTM(**kwargs)
assert isinstance(lstm_model, nn.Module), "Not an instance of nn.Module!"


#lstm_model.train_model(None, nn.HuberLoss, optim.Adam, 10, DataLoader, 0.0001)
lstm_model.predict()

