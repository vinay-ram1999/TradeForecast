from typing import Self

from .data import DataEntryPoint
from .indicator_utils import (
    ma_type_func_maping,
    calculate_macd_sl,
    calculate_rsi,
    calculate_atr,
    )

class Indicators(object):
    def __init__(self, data_entry: DataEntryPoint) -> None:
        self.data_entry = data_entry
        pass

    def add_moving_average(self, eval_var: str="Close", n: int=9, ma_type: str="SMA") -> Self:
        assert ma_type in ma_type_func_maping.keys(), f"Currently available ma_types are {list(ma_type_func_maping.keys())}, '{ma_type}' is not available"
        ma_func = ma_type_func_maping[ma_type]
        self.data_entry.data = ma_func(self.data_entry.data, eval_var, n)
        return self
    
    def add_macd_sl(self, eval_var: str="Close", a: int=12, b: int=26, c: int=9, ma_type: str="EMA") -> Self:
        assert ma_type in ma_type_func_maping.keys(), f"Currently available ma_types are {list(ma_type_func_maping.keys())}, '{ma_type}' is not available"
        self.data_entry.data = calculate_macd_sl(self.data_entry.data, eval_var, a, b, c, ma_type)
        return self
    
    def add_rsi(self, eval_var: str="Close", n: int=14, ma_type: str="EMA") -> Self:
        assert ma_type in ma_type_func_maping.keys(), f"Currently available ma_types are {list(ma_type_func_maping.keys())}, '{ma_type}' is not available"
        self.data_entry.data = calculate_rsi(self.data_entry.data, eval_var, n, ma_type)
        return self
    
    def add_atr(self, eval_var: str="Close", n: int=14, ma_type: str="EMA") -> Self:
        assert ma_type in ma_type_func_maping.keys(), f"Currently available ma_types are {list(ma_type_func_maping.keys())}, '{ma_type}' is not available"
        self.data_entry.data = calculate_atr(self.data_entry.data, eval_var, n, ma_type)
        return self
