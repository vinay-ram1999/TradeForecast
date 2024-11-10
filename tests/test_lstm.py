from torch.utils.data import DataLoader
from torch import nn, optim

from tradeforecast.augmentation import DataEntryPoint, Indicators, FeatureEngg, RNNDataset, train_test_split
from tradeforecast import LSTM

fpath = 'AAPL_1d_max_(None-None).csv'

data_entry = DataEntryPoint(fpath)

indicators = Indicators(data_entry)
indicators.add_macd_sl().add_rsi().add_atr()

features = FeatureEngg(data_entry)
features.add_quarters().add_weeks()

lf = data_entry.data.drop_nulls()

dataset_kwargs = {'lf': lf,
                 'non_temporal': data_entry.non_temporal,
                 'temporal': data_entry.temporal,
                 'target': 'Close',
                 'look_back_len': 30,
                 'forecast_len': 7}

rnn_dataset = RNNDataset(**dataset_kwargs)

train_dataset, test_dataset = train_test_split(rnn_dataset, 0.2)
print(len(train_dataset), len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, drop_last=False)

lstm_kwargs = {'input_size': len(rnn_dataset.features),
              'hidden_size': 5,
              'n_LSTM': 2,
              'fc_out_size':[],
              'output_size': rnn_dataset.forecast_len,
              'dropout': 0.3}

lstm_model = LSTM(**lstm_kwargs)

lstm_model.train_model(nn.HuberLoss, optim.Adam, 2, train_loader, 0.001)

y, y_preds = lstm_model.test_model(test_loader)
print(y.size(), y_preds.size())

model_fname = lstm_model.save_model_state(ticker_interval='AAPL_1d')

lstm_kwargs = {'input_size': len(rnn_dataset.features),
              'hidden_size': 5,
              'n_LSTM': 2,
              'fc_out_size':[10],
              'output_size': rnn_dataset.forecast_len,
              'dropout': 0.3}

lstm_loaded_model = LSTM(**lstm_kwargs)
lstm_loaded_model.load_model_state(model_fname)
y_loaded, y_preds_loaded = lstm_model.test_model(test_loader)
print(y_loaded.size(), y_preds_loaded.size())
