import polars as pl

import os

from ..constants import data_dir

class DataEntryPoint(object):
    def __init__(self, csv_fpath: str, datetime_var: str='Datetime') -> None:
        self.fpath = csv_fpath
        self.datetime_var = datetime_var
        self.base_vars = self.data.collect_schema().names()
        self.temporal_vars = list()
        pass
    
    @property
    def data(self) -> pl.LazyFrame:
        if hasattr(self, "_data"):
            return self._data
        else:
            self._data = pl.scan_csv(os.path.join(data_dir, self.fpath), try_parse_dates=True)
            self._data = self._data.with_columns(pl.col(self.datetime_var).dt.convert_time_zone(time_zone='EST'))
            return self._data

    @data.setter
    def data(self, lf: pl.LazyFrame):
        self._data = lf
