import polars as pl

import datetime as dt

def date_to_quarters(lf: pl.LazyFrame, var_name: str, col_name: str=None) -> pl.LazyFrame:
    col_name = 'Quarter' if col_name is None else col_name
    lf = lf.with_columns(pl.col(var_name).dt.quarter().alias(col_name))
    return lf, col_name

def date_to_weeks(lf: pl.LazyFrame, var_name: str, col_name: str=None) -> pl.LazyFrame:
    col_name = 'Week' if col_name is None else col_name
    lf = lf.with_columns(pl.col(var_name).dt.week().alias(col_name))
    return lf, col_name

def datetime_to_hours(lf: pl.LazyFrame, var_name: str, col_name: str=None) -> pl.LazyFrame:
    assert lf.collect_schema().to_python()[var_name] == dt.datetime, "To extract hours the variable must be of datetime type"
    col_name = 'Hour' if col_name is None else col_name
    lf = lf.with_columns(pl.col(var_name).dt.hour().alias(col_name))
    return lf, col_name
