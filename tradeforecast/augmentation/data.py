import polars as pl
import pandas as pd

import os

from ..constants import data_dir

class DataEntryPoint(object):
    def __init__(self, csv_fpath: str=None, df: pd.DataFrame=None, datetime_var: str='Datetime') -> None:
        if csv_fpath is not None:
            csv_fpath = os.path.abspath(os.path.join(data_dir, csv_fpath))
            assert os.path.isfile(csv_fpath), f"'{csv_fpath}' does not exist"
            self.fpath = csv_fpath
        self.lf = pl.from_pandas(df, include_index=True).lazy()
        self.datetime_var = datetime_var
        self.base_vars = self.data.select(pl.all().exclude(datetime_var)).collect_schema().names()
        self.temporal = list()
        pass
    
    @property
    def data(self) -> pl.LazyFrame:
        if hasattr(self, "_data"):
            return self._data
        else:
            self._data = pl.scan_csv(self.fpath, try_parse_dates=True) if hasattr(self, 'fpath') else self.lf
            self._data = self._data.with_columns(pl.col(self.datetime_var).cast(pl.Datetime).dt.convert_time_zone(time_zone='EST'))   # WARNING: currently only converting to EST
            return self._data

    @data.setter
    def data(self, lf: pl.LazyFrame):
        self._data = lf
    
    @property
    def non_temporal(self) -> list:
        return [x for x in self.data.select(pl.all().exclude(self.datetime_var)).collect_schema().names() if x not in self.temporal]
