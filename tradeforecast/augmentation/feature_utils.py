import polars as pl

import math

def date_to_quarters(lf: pl.LazyFrame, var_name: str, col_name: str=None) -> pl.LazyFrame:
    period = 4  # number of quarters in a year
    col_name = 'Quarter' if col_name is None else col_name
    lf = lf.with_columns(pl.col(var_name).dt.quarter().alias(col_name)
                         ).with_columns((2 * math.pi * (pl.col(col_name) / period)).sin().round(10).alias(f'{col_name}_sin'),
                                        (2 * math.pi * (pl.col(col_name) / period)).cos().round(10).alias(f'{col_name}_cos')).drop(col_name)
    return lf, [f'{col_name}_sin', f'{col_name}_cos']

def date_to_weeks(lf: pl.LazyFrame, var_name: str, col_name: str=None) -> pl.LazyFrame:
    period = 53 # number of weeks in a year
    col_name = 'Week' if col_name is None else col_name
    lf = lf.with_columns(pl.col(var_name).dt.week().alias(col_name)
                         ).with_columns((2 * math.pi * (pl.col(col_name) / period)).sin().round(10).alias(f'{col_name}_sin'),
                                        (2 * math.pi * (pl.col(col_name) / period)).cos().round(10).alias(f'{col_name}_cos')).drop(col_name)
    return lf, [f'{col_name}_sin', f'{col_name}_cos']

def datetime_to_hours(lf: pl.LazyFrame, var_name: str, col_name: str=None) -> pl.LazyFrame:
    assert lf.collect_schema()[var_name] == pl.Datetime, f"To extract hours the variable must be of 'Datetime' type not '{lf.collect_schema()[var_name]}' type"
    period = 24 # number of hours in a day
    col_name = 'Hour' if col_name is None else col_name
    lf = lf.with_columns(pl.col(var_name).dt.hour().alias(col_name)
                         ).with_columns((2 * math.pi * (pl.col(col_name) / period)).sin().round(10).alias(f'{col_name}_sin'),
                                        (2 * math.pi * (pl.col(col_name) / period)).cos().round(10).alias(f'{col_name}_cos')).drop(col_name)
    return lf, [f'{col_name}_sin', f'{col_name}_cos']

def datetime_to_minutes(lf: pl.LazyFrame, var_name: str, col_name: str=None) -> pl.LazyFrame:
    assert lf.collect_schema()[var_name] == pl.Datetime, f"To extract minutes the variable must be of 'Datetime' type not '{lf.collect_schema()[var_name]}' type"
    period = 60 # number of minutes in an hour
    col_name = 'Minute' if col_name is None else col_name
    lf = lf.with_columns(pl.col(var_name).dt.minute().alias(col_name)
                         ).with_columns((2 * math.pi * (pl.col(col_name) / period)).sin().round(10).alias(f'{col_name}_sin'),
                                        (2 * math.pi * (pl.col(col_name) / period)).cos().round(10).alias(f'{col_name}_cos')).drop(col_name)
    return lf, [f'{col_name}_sin', f'{col_name}_cos']
