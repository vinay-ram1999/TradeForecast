import yfinance as yf
import pandas as pd

import datetime as dt
import os

from .constants import data_dir

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
        self._check_data_dir()
        return
    
    def __repr__(self) -> str:
        return f'{__name__} object <{self.ticker}>'
    
    @staticmethod
    def _check_data_dir():
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f'{data_dir}')
    
    def export_historic_data(self, interval: str='1h', period: str='max', start=None, end=None) -> dict:
        _fnames = {}
        for ticker in self.ticker:
            csv_fname = f'{ticker}_{interval}_{period}_({start}-{end}).csv'
            csv_fpath = os.path.join(data_dir, csv_fname)

            try:
                data = yf.download(ticker, start=start, end=end, period=period, interval=interval, group_by='ticker')
                data.to_csv(csv_fpath, index=False)
                _fnames[ticker] = csv_fname
            except Exception as e:
                print(e)
                pass
        return _fnames
    
    def fetch_historic_data(self, interval: str='1h', period: str='max', start=None, end=None) -> dict:
        _dfs = {}
        yf_ticker_obj = [self.yf_ticker_obj]
        if isinstance(self.yf_ticker_obj, yf.Tickers):
            yf_ticker_obj = self.yf_ticker_obj.tickers.values()
        
        try:
            for obj in yf_ticker_obj:
                data = obj.history(start=start, end=end, period=period, interval=interval)
                _dfs[obj.ticker] = data
        except Exception as e:
                print(e)
        return _dfs


