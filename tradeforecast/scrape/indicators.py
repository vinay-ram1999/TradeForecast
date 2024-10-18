import polars as pl

import os

from . import data_dir
from .utils import (
    ma_type_func_maping,
    calculate_macd_sl,
    calculate_rsi,
    calculate_atr
    )

class IndicatorBase:
    @staticmethod
    def moving_average(lf: pl.LazyFrame, eval_var: str, n: int, ma_type: str) -> pl.LazyFrame:
        try:
            assert ma_type in ma_type_func_maping.keys(), f"Currently available ma_types are {list(ma_type_func_maping.keys())}, '{ma_type}' is not available"
            ma_func = ma_type_func_maping[ma_type]
            lf = ma_func(lf, eval_var, n)
        except AssertionError as e:
            print(e, "\nMA variable is not added to the dataset!")
        return lf
    
    @staticmethod
    def macd_sl(lf: pl.LazyFrame, eval_var: str, a: int, b: int, c: int, ma_type: str) -> pl.LazyFrame:
        lf = calculate_macd_sl(lf, eval_var, a, b, c, ma_type)
        return lf
    
    @staticmethod
    def rsi(lf: pl.LazyFrame, eval_var: str, n: int, ma_type: str) -> pl.LazyFrame:
        lf = calculate_rsi(lf, eval_var, n, ma_type)
        return lf
    
    @staticmethod
    def atr(lf: pl.LazyFrame, eval_var: str, n: int, ma_type: str) -> pl.LazyFrame:
        lf = calculate_atr(lf, eval_var, n, ma_type)
        return lf


class Indicators(IndicatorBase):
    def __init__(self, csv_fpath: str) -> None:
        self.fpath = csv_fpath
        return
    
    @property
    def data(self) -> pl.LazyFrame:
        if hasattr(self, "_data"):
            return self._data
        else:
            try:
                self._data = pl.scan_csv(os.path.join(data_dir, self.fpath))
                return self._data
            except Exception as e:
                print(e)

    @data.setter
    def data(self, lf: pl.LazyFrame):
        self._data = lf
    
    def add_moving_average(self, eval_var: str="Close", n: int=9, ma_type: str="SMA") -> None:
        self.data = self.moving_average(self.data, eval_var, n, ma_type)
        return
    
    def add_macd_sl(self, eval_var: str="Close", a: int=12, b: int=26, c: int=9, ma_type: str="EMA") -> None:
        self.data = self.macd_sl(self.data, eval_var, a, b, c, ma_type)
        return
    
    def add_rsi(self, eval_var: str="Close", n: int=14, ma_type: str="EMA") -> None:
        self.data = self.rsi(self.data, eval_var, n, ma_type)
        return
    
    def add_atr(self, eval_var: str="Close", n: int=14, ma_type: str="EMA") -> None:
        self.data = self.atr(self.data, eval_var, n, ma_type)
        return
