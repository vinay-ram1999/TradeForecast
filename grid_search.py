from torch.utils.data import DataLoader
import torch.nn.functional as F
from lightning import Trainer
from torch import optim, Tensor
import pandas as pd

import os

from tradeforecast.augmentation import RNNDataset, DataEntryPoint, Indicators, FeatureEngg, train_val_test_split
from tradeforecast.forecast import LSTM, ConvLSTM, EncTransformer, calc_metrics
from tradeforecast.constants import data_dir
from tradeforecast.scrape import Scrapper

ticker = 'GOOG'
scrapper = Scrapper(ticker)

df_dict = scrapper.fetch_historic_data(interval='1d', start='2015-01-01', end='2024-12-06')

data_entry_base = DataEntryPoint(df=df_dict[ticker])

data_entry_feat_engg = DataEntryPoint(df=df_dict[ticker])

indicators = Indicators(data_entry_feat_engg)
indicators.add_moving_average().add_moving_average(n=30).add_macd_sl().add_rsi().add_atr()

features = FeatureEngg(data_entry_feat_engg)
features.add_quarters().add_weeks()

data_entries = {'base': data_entry_base, 'feat_engg': data_entry_feat_engg}

lstm_params = {'data_version':[], 'n_feat':[], 'hidden_size':[], 'n_LSTM':[], 'dropout':[], 'criterion':[], 'init_lr':[], 'final_lr':[], 'train_loss':[], 'test_loss':[], 'n_params':[], 'MAE':[], 'MSE':[], 'RMSE':[], 'R-squared':[]}
clstm_params = {'data_version':[], 'n_feat':[], 'conv_out_size': [], 'kernel_size': [], 'hidden_size': [], 'n_LSTM': [], 'dropout': [], 'criterion': [], 'init_lr': [], 'final_lr':[], 'train_loss':[], 'test_loss':[], 'n_params':[], 'MAE':[], 'MSE':[], 'RMSE':[], 'R-squared':[]}
et_params = {'data_version':[], 'n_feat':[], 'nhead':[], 'd_model':[], 'num_layers': [], 'dropout':[], 'criterion':[], 'init_lr':[], 'final_lr':[], 'train_loss':[], 'test_loss':[], 'n_params':[], 'MAE':[], 'MSE':[], 'RMSE':[], 'R-squared':[]}

for data_version, data_entry in data_entries.items():

    lf = data_entry.data.drop_nulls()

    look_back_len = 60
    forecast_len = 5
    batch_size = 128
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
    dropout_opts = [0, 0.05]
    criterion_opts = [F.l1_loss, F.mse_loss]
    lr_opts = [0.1]
    conv_out_size_opts = [len(rnn_dataset.features)*2]
    kernel_size_opts = [rnn_dataset.forecast_len, rnn_dataset.forecast_len*2]
    nhead_opts = [2, 4]
    d_model_opts = [64]
    num_layers_opts = [2, 4]

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
                            lstm_trainer.fit(lstm_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
                            train_loss: Tensor = lstm_trainer.callback_metrics.get('train/loss', None)
                            final_lr: Tensor = lstm_trainer.callback_metrics.get('lr', None)
                            lstm_trainer.test(lstm_model, test_loader)
                            test_loss: Tensor = lstm_trainer.callback_metrics.get('test/loss', None)
                            for param in lstm_params.keys():
                                try:
                                    val = getattr(lstm_model, param)
                                    lstm_params[param].append(val if not callable(val) else val.__name__)
                                except AttributeError:
                                    pass
                            lstm_params['data_version'].append(data_version)
                            lstm_params['n_feat'].append(len(rnn_dataset.features))
                            lstm_params['n_params'].append(sum(x.numel() for x in lstm_model.parameters() if x.requires_grad))
                            lstm_params['train_loss'].append(train_loss.item())
                            lstm_params['init_lr'].append(lr)
                            lstm_params['final_lr'].append(final_lr.item())
                            lstm_params['test_loss'].append(test_loss.item())
                            y, y_pred = lstm_model.predict(test_loader)
                            lstm_metrics = calc_metrics(y, y_pred)
                            for metric in lstm_metrics.keys():
                                lstm_params[metric].append(lstm_metrics[metric])

                            for conv_out_size in conv_out_size_opts:
                                for kernel_size in kernel_size_opts:

                                    clstm_kwargs = {'input_size': len(rnn_dataset.features),
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

                                    clstm_model = ConvLSTM(**clstm_kwargs)
                                    clstm_trainer = Trainer(fast_dev_run=False, max_epochs=max_epoch, log_every_n_steps=10, check_val_every_n_epoch=100)
                                    clstm_trainer.fit(clstm_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
                                    train_loss: Tensor = clstm_trainer.callback_metrics.get('train/loss', None)
                                    final_lr: Tensor = clstm_trainer.callback_metrics.get('lr', None)
                                    clstm_trainer.test(clstm_model, test_loader)
                                    test_loss: Tensor = clstm_trainer.callback_metrics.get('test/loss', None)
                                    for param in clstm_params.keys():
                                        try:
                                            val = getattr(clstm_model, param)
                                            clstm_params[param].append(val if not callable(val) else val.__name__)
                                        except AttributeError:
                                            pass
                                    clstm_params['data_version'].append(data_version)
                                    clstm_params['n_feat'].append(len(rnn_dataset.features))
                                    clstm_params['n_params'].append(sum(x.numel() for x in clstm_model.parameters() if x.requires_grad))
                                    clstm_params['train_loss'].append(train_loss.item())
                                    clstm_params['init_lr'].append(lr)
                                    clstm_params['final_lr'].append(final_lr.item())
                                    clstm_params['test_loss'].append(test_loss.item())
                                    y, y_pred = clstm_model.predict(test_loader)
                                    clstm_metrics = calc_metrics(y, y_pred)
                                    for metric in clstm_metrics.keys():
                                        clstm_params[metric].append(clstm_metrics[metric])

                for nhead in nhead_opts:
                    for d_model in d_model_opts:
                        for num_layers in num_layers_opts:

                            et_kwargs = {'input_size': len(rnn_dataset.features),
                                        'nhead': nhead,
                                        'd_model': d_model,
                                        'num_layers': num_layers,
                                        'output_size': rnn_dataset.forecast_len,
                                        'dropout': dropout,
                                        'criterion': criterion,
                                        'lr': lr,
                                        'optimizer': optim.SGD}

                            et_model = EncTransformer(**et_kwargs)
                            et_trainer = Trainer(fast_dev_run=False, max_epochs=max_epoch, log_every_n_steps=10, check_val_every_n_epoch=100)
                            et_trainer.fit(et_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
                            train_loss: Tensor = et_trainer.callback_metrics.get('train/loss', None)
                            final_lr: Tensor = et_trainer.callback_metrics.get('lr', None)
                            et_trainer.test(et_model, test_loader)
                            test_loss: Tensor = et_trainer.callback_metrics.get('test/loss', None)
                            for param in et_params.keys():
                                try:
                                    val = getattr(et_model, param)
                                    et_params[param].append(val if not callable(val) else val.__name__)
                                except AttributeError:
                                    pass
                            et_params['data_version'].append(data_version)
                            et_params['n_feat'].append(len(rnn_dataset.features))
                            et_params['n_params'].append(sum(x.numel() for x in et_model.parameters() if x.requires_grad))
                            et_params['train_loss'].append(train_loss.item())
                            et_params['init_lr'].append(lr)
                            et_params['final_lr'].append(final_lr.item())
                            et_params['test_loss'].append(test_loss.item())
                            y, y_pred = et_model.predict(test_loader)
                            et_metrics = calc_metrics(y, y_pred)
                            for metric in et_metrics.keys():
                                et_params[metric].append(et_metrics[metric])

lstm_params = pd.DataFrame(lstm_params).round(5).sort_values(by='train_loss', ascending=True).reset_index(drop=True)
lstm_params.index.name = 'model'
print(lstm_params)
lstm_params.to_csv(os.path.join(data_dir, 'lstm_params.csv'))

clstm_params = pd.DataFrame(clstm_params).round(5).sort_values(by='train_loss', ascending=True).reset_index(drop=True)
clstm_params.index.name = 'model'
print(clstm_params)
clstm_params.to_csv(os.path.join(data_dir, 'clstm_params.csv'))

et_params = pd.DataFrame(et_params).round(5).sort_values(by='train_loss', ascending=True).reset_index(drop=True)
et_params.index.name = 'model'
print(et_params)
et_params.to_csv(os.path.join(data_dir, 'et_params.csv'))
