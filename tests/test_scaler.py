from tradeforecast.scrape import Scrapper
from tradeforecast.augmentation import DataEntryPoint, Indicators, FeatureEngg, LSTMDataset
from sklego.preprocessing import RepeatingBasisFunction

#fpath = 'AAPL_1d_max_(None-None).csv'
fpath = 'AAPL_1h_max_(None-None).csv'


data_entry = DataEntryPoint(fpath)

indicators = Indicators(data_entry)
#indicators.add_moving_average(eval_var='Close', n=9, ma_type='SMA')
#indicators.add_moving_average(eval_var='Close', n=9, ma_type='EMA')
indicators.add_macd_sl()
indicators.add_rsi()
indicators.add_atr()

features = FeatureEngg(data_entry)
features.add_quarters()
features.add_weeks()
features.add_hours()

#print(data_entry.base_vars)
#print(data_entry.temporal_vars)
#print(data_entry.columns())

lf = data_entry.data

print(lf.drop(['Datetime','High']).collect())
#dataset = LSTMDataset(lf, seq_length=10)
#X, y = dataset[33]
#print(X.shape,y)



