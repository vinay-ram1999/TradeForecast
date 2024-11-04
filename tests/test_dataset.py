from torch.utils.data import DataLoader

from tradeforecast.augmentation import DataEntryPoint, Indicators, FeatureEngg, RNNDataset

fpath = 'AAPL_1d_max_(None-None).csv'
#fpath = 'AAPL_1h_max_(None-None).csv'


data_entry = DataEntryPoint(fpath)

indicators = Indicators(data_entry)
#indicators.add_moving_average(eval_var='Close', n=9, ma_type='SMA')
#indicators.add_moving_average(eval_var='Close', n=9, ma_type='EMA')
indicators.add_macd_sl().add_rsi().add_atr()

features = FeatureEngg(data_entry)
features.add_quarters().add_weeks()
#features.add_hours()

lf = data_entry.data.drop_nulls()
#print(lf.collect())

kwargs = {'lf': lf,
        'non_temporal': data_entry.non_temporal,
        'temporal': data_entry.temporal,
        'target': 'Close',
        'look_back_len': 10,
        'forecast_len': 3,
        'split': 0.2}

train_dataset = RNNDataset(train=True, **kwargs)
test_dataset = RNNDataset(train=False, **kwargs)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, drop_last=False)

X, y = next(iter(train_dataset))

print(X.size(), y.size())
