import yfinance as yf
import pandas as pd

import os

from ..constants import data_dir

class Scrapper:
    def __init__(self, ticker: str | list=None) -> None:
        if ticker:
            if isinstance(ticker, str) and ',' not in ticker:
                self.yf_ticker_obj = yf.Ticker(ticker)
                self.ticker = [self.yf_ticker_obj.ticker]
            elif isinstance(ticker, list) or (isinstance(ticker, str) and ',' in ticker):
                self.yf_ticker_obj = yf.Tickers(ticker)
                self.ticker = self.yf_ticker_obj.symbols
            else:
                raise ValueError('The ticker must be a string or list of strings')
        self.check_data_dir()
        return
    
    def __repr__(self) -> str:
        return f'{__name__} object <{self.ticker}>'
    
    @staticmethod
    def check_data_dir():
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f'{data_dir}')
    
    def export_historic_data(self, interval: str='1d', period: str='max', start=None, end=None) -> dict:
        _fnames = {}
        for ticker in self.ticker:
            csv_fname = f'{ticker}_{interval}_{period}_({start}-{end}).csv'
            csv_fpath = os.path.join(data_dir, csv_fname)

            data: pd.DataFrame = yf.download(ticker, start=start, end=end, period=period, interval=interval, group_by='ticker')
            data.index.names = ['Datetime']
            data.drop(['Adj Close','Dividends','Stock Splits'], axis=1, inplace=True, errors='ignore') # drop 'Adj Close' column if exists
            data.to_csv(csv_fpath, index=True)
            _fnames[ticker] = csv_fname
        return _fnames
    
    def fetch_historic_data(self, interval: str='1d', period: str='max', start=None, end=None) -> dict[pd.DataFrame]:
        _dfs = {}
        yf_ticker_obj = [self.yf_ticker_obj]
        if isinstance(self.yf_ticker_obj, yf.Tickers):
            yf_ticker_obj = self.yf_ticker_obj.tickers.values()
        for obj in yf_ticker_obj:
            data: pd.DataFrame = obj.history(start=start, end=end, period=period, interval=interval)
            data.index.names = ['Datetime']
            data.drop(['Adj Close','Dividends','Stock Splits'], axis=1, inplace=True, errors='ignore') # drop 'Adj Close' column if exists
            _dfs[obj.ticker] = data
        return _dfs


