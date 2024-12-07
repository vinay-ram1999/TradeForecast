from torch.utils.data import DataLoader
from torch import nn, optim, Tensor
import torch.nn.functional as F
from lightning import Trainer

from tradeforecast.augmentation import DataEntryPoint, Indicators, FeatureEngg, RNNDataset, train_val_test_split
from tradeforecast import LSTM

fpath = 'AAPL_1d_max_(None-None).csv'

data_entry = DataEntryPoint(fpath)

indicators = Indicators(data_entry)
indicators.add_macd_sl().add_rsi().add_atr()

features = FeatureEngg(data_entry)
features.add_quarters().add_weeks().add_hours()

lf = data_entry.data.drop_nulls()

dataset_kwargs = {'lf': lf,
                 'non_temporal': data_entry.non_temporal,
                 'temporal': data_entry.temporal,
                 'target': 'Close',
                 'look_back_len': 30,
                 'forecast_len': 7}

rnn_dataset = RNNDataset(**dataset_kwargs)

train_dataset, val_dataset, test_dataset = train_val_test_split(rnn_dataset, val_size=0.1, test_size=0.1)
print(len(train_dataset), len(val_dataset), len(test_dataset))

batch_size = 128
num_workers = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

lstm_kwargs = {'input_size': len(rnn_dataset.features),
              'hidden_size': 5,
              'n_LSTM': 2,
              'bidirectional': True,
              'fc_out_size':[],
              'output_size': rnn_dataset.forecast_len,
              'dropout': 0.1,
              'criterion': F.mse_loss,
              'lr': 1.0,
              'optimizer': optim.Adam}

lstm_model = LSTM(**lstm_kwargs)

trainer = Trainer(fast_dev_run=True, max_epochs=3)

trainer.fit(lstm_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(lstm_model, test_loader)

model_fname = lstm_model.save_model_state(ticker_interval='AAPL_1d')
print(model_fname)
