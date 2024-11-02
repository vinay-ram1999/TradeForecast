from torch.utils.data import DataLoader

from tradeforecast.augmentation import DataEntryPoint, Indicators, FeatureEngg, RNNDataset

#fpath = 'AAPL_1d_max_(None-None).csv'
fpath = 'AAPL_1h_max_(None-None).csv'


data_entry = DataEntryPoint(fpath)

indicators = Indicators(data_entry)
#indicators.add_moving_average(eval_var='Close', n=9, ma_type='SMA')
#indicators.add_moving_average(eval_var='Close', n=9, ma_type='EMA')
indicators.add_macd_sl().add_rsi().add_atr()

features = FeatureEngg(data_entry)
features.add_quarters().add_weeks()
features.add_hours()

print(data_entry.base_vars)
print(data_entry.temporal)
print(data_entry.non_temporal)

lf = data_entry.data.drop_nulls()
#print(lf.collect())

kwargs = {'lf': lf,
        'non_temporal': data_entry.non_temporal,
        'temporal': data_entry.temporal,
        'target': ['Open','High'],
        'seq_length': 5,
        'split': 0.2}

train_dataset = RNNDataset(train=True, **kwargs)
test_dataset = RNNDataset(train=False, **kwargs)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(train_loader.batch_size)
print(train_loader.dataset)



