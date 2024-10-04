from tradeforecast import Indicators
from tradeforecast import Scrapper

tckrs = 'AAPL'
scrapper = Scrapper(tckrs)

exported_fname = scrapper.export_historic_data()

indicators = Indicators(exported_fname['AAPL'])
#indicators.add_moving_average(eval_var='Close', n=9, ma_type='SMA')
#indicators.add_moving_average(eval_var='Close', n=9, ma_type='EMA')

indicators.add_macd_sl()
indicators.add_rsi()
print(indicators.data.drop_nulls().collect())

