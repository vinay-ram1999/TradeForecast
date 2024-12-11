from torch.utils.data import DataLoader
import torch.nn.functional as F
from lightning import Trainer
from torch import optim
import pandas as pd

import os

from tradeforecast.augmentation import RNNDataset, DataEntryPoint, Indicators, FeatureEngg, train_val_test_split
from tradeforecast.forecast import LSTM, TFModel, TFTransformer
from tradeforecast.constants import data_dir

from tradeforecast import Scrapper

ticker = 'NVDA'
scrapper = Scrapper(ticker)

df_dict = scrapper.fetch_historic_data(interval='1d', start='2015-01-01', end='2024-12-06')

data_entry = DataEntryPoint(df=df_dict[ticker])

indicators = Indicators(data_entry)
indicators.add_moving_average().add_moving_average(n=30).add_macd_sl().add_rsi().add_atr()

features = FeatureEngg(data_entry)
features.add_quarters().add_weeks()

lf = data_entry.data.drop_nulls()

look_back_len = 60
forecast_len = 5
batch_size = 256
num_workers = 8

dataset_kwargs = {'lf': lf,
                'non_temporal': data_entry.non_temporal,
                'temporal': data_entry.temporal,
                'target': 'Close',
                'look_back_len': look_back_len,
                'forecast_len': forecast_len}

rnn_dataset = RNNDataset(**dataset_kwargs)

train_dataset, test_dataset = train_val_test_split(rnn_dataset, test_size=0.1)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

max_epoch = 500
hidden_size_opts = [32, 64]
n_LSTM_opts = [2]
dropout_opts = [0]
criterion_opts = [F.l1_loss, F.mse_loss]
lr_opts = [0.1]
conv_out_size_opts = [len(rnn_dataset.features)*2]
kernel_size_opts = [rnn_dataset.forecast_len, rnn_dataset.forecast_len*2]
nhead_opts = [2, 4]
d_model_opts = [64]
num_layers_opts = [2, 4]

lstm_metrics = {'hidden_size':[], 'n_LSTM':[], 'dropout':[], 'criterion':[], 'lr':[], 'train_loss':[], 'test_loss':[], 'n_params':[]}
tf_metrics = {'conv_out_size': [], 'kernel_size': [], 'hidden_size': [], 'n_LSTM': [], 'dropout': [], 'criterion': [], 'lr': [], 'train_loss':[], 'test_loss':[], 'n_params':[]}
tft_metrics = {'nhead':[], 'd_model':[], 'num_layers': [], 'dropout':[], 'criterion':[], 'lr':[], 'train_loss':[], 'test_loss':[], 'n_params':[]}

for dropout in dropout_opts:
    for criterion in criterion_opts:
        for lr in lr_opts:

            for hidden_size in hidden_size_opts:
                for n_LSTM in n_LSTM_opts:

                        lstm_kwargs = {'input_size': len(rnn_dataset.features),
                                    'hidden_size': hidden_size,
                                    'n_LSTM': n_LSTM,
                                    'bidirectional': False,
                                    'fc_out_size': [],
                                    'output_size': rnn_dataset.forecast_len,
                                    'dropout': dropout,
                                    'criterion': criterion,
                                    'lr': lr,
                                    'optimizer': optim.SGD}

                        lstm_model = LSTM(**lstm_kwargs)
                        lstm_trainer = Trainer(fast_dev_run=False, max_epochs=max_epoch, log_every_n_steps=10, check_val_every_n_epoch=100)
                        lstm_trainer.fit(lstm_model, train_dataloaders=train_loader)
                        for param in lstm_metrics.keys():
                            if not param in ['train_loss', 'test_loss', 'n_params']:
                                val = getattr(lstm_model, param)
                                lstm_metrics[param].append(val if not callable(val) else val.__name__)
                        train_loss = lstm_trainer.test(lstm_model, train_loader)
                        test_loss = lstm_trainer.test(lstm_model, test_loader)
                        lstm_metrics['n_params'].append(sum(x.numel() for x in lstm_model.parameters() if x.requires_grad))
                        lstm_metrics['train_loss'].append(train_loss[0]['test/loss'])
                        lstm_metrics['test_loss'].append(test_loss[0]['test/loss'])

                        for conv_out_size in conv_out_size_opts:
                            for kernel_size in kernel_size_opts:

                                tf_kwargs = {'input_size': len(rnn_dataset.features),
                                            'conv_out_size': conv_out_size,
                                            'kernel_size': kernel_size,
                                            'hidden_size': hidden_size,
                                            'n_LSTM': n_LSTM,
                                            'bidirectional': False,
                                            'fc_out_size': [],
                                            'output_size': rnn_dataset.forecast_len,
                                            'dropout': dropout,
                                            'criterion': criterion,
                                            'lr': lr,
                                            'optimizer': optim.SGD}

                                tf_model = TFModel(**tf_kwargs)
                                tf_trainer = Trainer(fast_dev_run=False, max_epochs=max_epoch, log_every_n_steps=10, check_val_every_n_epoch=100)
                                tf_trainer.fit(tf_model, train_dataloaders=train_loader)
                                for param in tf_metrics.keys():
                                    if not param in ['train_loss', 'test_loss', 'n_params']:
                                        val = getattr(tf_model, param)
                                        tf_metrics[param].append(val if not callable(val) else val.__name__)
                                train_loss = tf_trainer.test(tf_model, train_loader)
                                test_loss = tf_trainer.test(tf_model, test_loader)
                                tf_metrics['n_params'].append(sum(x.numel() for x in tf_model.parameters() if x.requires_grad))
                                tf_metrics['train_loss'].append(train_loss[0]['test/loss'])
                                tf_metrics['test_loss'].append(test_loss[0]['test/loss'])

            for nhead in nhead_opts:
                for d_model in d_model_opts:
                    for num_layers in num_layers_opts:

                        tft_kwargs = {'input_size': len(rnn_dataset.features),
                                    'nhead': nhead,
                                    'd_model': d_model,
                                    'num_layers': num_layers,
                                    'output_size': rnn_dataset.forecast_len,
                                    'dropout': dropout,
                                    'criterion': criterion,
                                    'lr': lr,
                                    'optimizer': optim.SGD}

                        tft_model = TFTransformer(**tft_kwargs)
                        tft_trainer = Trainer(fast_dev_run=False, max_epochs=max_epoch, log_every_n_steps=10, check_val_every_n_epoch=100)
                        tft_trainer.fit(tft_model, train_dataloaders=train_loader)
                        for param in tft_metrics.keys():
                            if not param in ['train_loss', 'test_loss', 'n_params']:
                                val = getattr(tft_model, param)
                                tft_metrics[param].append(val if not callable(val) else val.__name__)
                        train_loss = tft_trainer.test(tft_model, train_loader)
                        test_loss = tft_trainer.test(tft_model, test_loader)
                        tft_metrics['n_params'].append(sum(x.numel() for x in tft_model.parameters() if x.requires_grad))
                        tft_metrics['train_loss'].append(train_loss[0]['test/loss'])
                        tft_metrics['test_loss'].append(test_loss[0]['test/loss'])

lstm_metrics = pd.DataFrame(lstm_metrics)
lstm_metrics.index.name = 'model'
print(lstm_metrics)
lstm_metrics.to_csv(os.path.join(data_dir, 'lstm_metrics.csv'))

tf_metrics = pd.DataFrame(tf_metrics)
tf_metrics.index.name = 'model'
print(tf_metrics)
tf_metrics.to_csv(os.path.join(data_dir, 'tf_metrics.csv'))

tft_metrics = pd.DataFrame(tft_metrics)
tft_metrics.index.name = 'model'
print(tft_metrics)
tft_metrics.to_csv(os.path.join(data_dir, 'tft_metrics.csv'))
