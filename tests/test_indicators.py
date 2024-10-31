from tradeforecast import Scrapper, DataEntryPoint, Indicators, FeatureEngg

tckrs = 'AAPL,NVDA'
scrapper = Scrapper(tckrs)

exported_fname = scrapper.export_historic_data()

for key in exported_fname.keys():

    data_entry = DataEntryPoint(exported_fname[key])
    
    indicators = Indicators(data_entry)
    #indicators.add_moving_average(eval_var='Close', n=9, ma_type='SMA')
    #indicators.add_moving_average(eval_var='Close', n=9, ma_type='EMA')
    indicators.add_macd_sl()
    indicators.add_rsi()
    indicators.add_atr()

    features = FeatureEngg(data_entry)
    features.add_quarters('Datetime')
    features.add_weeks('Datetime')

    print(data_entry.data.drop_nulls().collect())

