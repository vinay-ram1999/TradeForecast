from tradeforecast.scrape import Scrapper
from tradeforecast.augmentation import Indicators, LSTMDataset


tckrs = 'AAPL'
scrapper = Scrapper(tckrs)

exported_fname = scrapper.export_historic_data(period='max')

for key in exported_fname.keys():

    indicators = Indicators(exported_fname[key])
    #indicators.add_moving_average(eval_var='Close', n=9, ma_type='SMA')
    #indicators.add_moving_average(eval_var='Close', n=9, ma_type='EMA')

    indicators.add_macd_sl()
    indicators.add_rsi()
    indicators.add_atr()
    lf = indicators.data.drop_nulls()

    dataset = LSTMDataset(lf, seq_length=10)
    X, y = dataset[33]
    print(X.shape,y)



