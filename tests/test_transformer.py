from torch.utils.data import DataLoader
from torch import nn, optim, Tensor

from tradeforecast.augmentation import DataEntryPoint, Indicators, FeatureEngg, RNNDataset, train_test_split
from tradeforecast import TFTransformer

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

train_dataset, test_dataset = train_test_split(rnn_dataset, 0.2)
print(len(train_dataset), len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, drop_last=False)

tft_kwargs = {'input_size': len(rnn_dataset.features),
             'nhead': 4,
             'd_model': 64,
             'num_layers': 5,
             'output_size': rnn_dataset.forecast_len,
             'dropout': 0.2}

tft_model = TFTransformer(**tft_kwargs)

tft_model.train_model(criterion=nn.MSELoss, optimizer=optim.SGD, n_epochs=2, data_loader=train_loader, min_learning_rate=0.0001)

y: Tensor; y_preds: Tensor
y, y_preds = tft_model.test_model(test_loader)
print(y.size(), y_preds.size())

model_fname = tft_model.save_model_state(ticker_interval='AAPL_1d')
