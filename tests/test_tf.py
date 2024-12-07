from torch.utils.data import DataLoader
from torch import nn, optim, Tensor
import torch.nn.functional as F
from lightning import Trainer

from tradeforecast.augmentation import DataEntryPoint, Indicators, FeatureEngg, RNNDataset, train_val_test_split
from tradeforecast import TFModel

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
                 'look_back_len': 50,
                 'forecast_len': 7}

rnn_dataset = RNNDataset(**dataset_kwargs)

train_dataset, val_dataset, test_dataset = train_val_test_split(rnn_dataset, val_size=0.1, test_size=0.1)
print(len(train_dataset), len(val_dataset), len(test_dataset))

batch_size = 128
num_workers = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

tf_kwargs = {'input_size': len(rnn_dataset.features),
            'conv_out_size': len(rnn_dataset.features)*2,
            'kernel_size': 3,
            'hidden_size': 5,
            'n_LSTM': 2,
            'bidirectional': True,
            'fc_out_size':[15],
            'output_size': rnn_dataset.forecast_len,
            'dropout': 0.1,
            'criterion': F.mse_loss,
            'lr': 1.0,
            'optimizer': optim.SGD}

tf_model = TFModel(**tf_kwargs)

trainer = Trainer(fast_dev_run=False, max_epochs=3)

trainer.fit(tf_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(tf_model, test_loader)

model_fname = tf_model.save_model_state(ticker_interval='AAPL_1d')
