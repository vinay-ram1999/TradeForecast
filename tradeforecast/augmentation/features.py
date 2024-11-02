from typing import Self

from .data import DataEntryPoint
from .feature_utils import (
    date_to_quarters,
    date_to_weeks,
    datetime_to_hours,
)

class FeatureEngg(object):
    def __init__(self, data_entry: DataEntryPoint) -> None:
        self.data_entry = data_entry
        pass

    def add_quarters(self) -> Self:
        lf, col_names = date_to_quarters(self.data_entry.data, self.data_entry.datetime_var)
        self.data_entry.data = lf
        self.data_entry.temporal += col_names
        return self

    def add_weeks(self) -> Self:
        lf, col_names = date_to_weeks(self.data_entry.data, self.data_entry.datetime_var)
        self.data_entry.data = lf
        self.data_entry.temporal += col_names
        return self

    def add_hours(self) -> Self:
        lf, col_names = datetime_to_hours(self.data_entry.data, self.data_entry.datetime_var)
        self.data_entry.data = lf
        self.data_entry.temporal += col_names
        return self
    

