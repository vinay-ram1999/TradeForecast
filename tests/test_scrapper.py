from tradeforecast import Scrapper

tckrs = 'AAPL,NVDA'
scrapper = Scrapper(tckrs)
print(scrapper)
print(scrapper.ticker)

exported_fnames = scrapper.export_historic_data()
print(exported_fnames)

fetched_dfs = scrapper.fetch_historic_data()
print(fetched_dfs)
