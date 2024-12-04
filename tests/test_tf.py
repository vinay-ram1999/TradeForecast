from torch.utils.data import DataLoader
from torch import nn, optim, Tensor

from tradeforecast.augmentation import DataEntryPoint, Indicators, FeatureEngg, RNNDataset, train_test_split
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

train_dataset, test_dataset = train_test_split(rnn_dataset, 0.2)
print(len(train_dataset), len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, drop_last=False)

tf_kwargs = {'input_size': len(rnn_dataset.features),
             'conv_out_size': len(rnn_dataset.features)*2,
             'kernel_size': 3,
             'hidden_size': 5,
             'n_LSTM': 2,
             'bidirectional': True,
             'fc_out_size':[15],
             'output_size': rnn_dataset.forecast_len,
             'dropout': 0.2}

tf_model = TFModel(**tf_kwargs)

tf_model.train_model(criterion=nn.HuberLoss, optimizer=optim.SGD, n_epochs=2, data_loader=train_loader, min_learning_rate=0.0001)

y: Tensor; y_preds: Tensor
y, y_preds = tf_model.test_model(test_loader)
print(y.size(), y_preds.size())

model_fname = tf_model.save_model_state(ticker_interval='AAPL_1d')
