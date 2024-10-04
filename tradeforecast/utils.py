import polars as pl

import math

def calculate_sma(lf: pl.LazyFrame, eval_var: str, n: int, col_name: str=None) -> pl.LazyFrame:
    """Calculate Simple Moving Average (SMA)"""
    col_name = f'{eval_var}_SMA_{n}' if col_name == None else col_name
    lf = lf.with_columns(pl.col(eval_var).rolling_mean(window_size=n).alias(col_name))
    return lf

def calculate_ema(lf: pl.LazyFrame, eval_var: str, n: int, col_name: str=None) -> pl.LazyFrame:
    """Calculate Exponential Moving Average"""
    col_name = f'{eval_var}_EMA_{n}' if col_name == None else col_name
    lf = lf.with_columns(pl.col(eval_var).ewm_mean(span=n, min_periods=n).alias(col_name))
    return lf

ma_type_func_maping = {"SMA": calculate_sma, "EMA": calculate_ema}

def calculate_macd_sl(lf: pl.LazyFrame, eval_var: str, a: int, b: int, c: int, ma_type: str) -> pl.LazyFrame:
    """
    Calculate Moving Average Convergence/Divergence (MACD) and Signal Line
    typical values  a(fast MA window) = 12;
                    b(slow MA window) = 26;
                    c(signal line MA window) = 9;
    """
    col_names = [f'{eval_var}_MACD_{a}-{b}-{c}', f'{eval_var}_MACD_SL']
    ma_func = ma_type_func_maping[ma_type]
    lf = ma_func(lf, eval_var, a, 'fast_ma')
    lf = ma_func(lf, eval_var, b, 'slow_ma')
    lf = lf.with_columns((pl.col('fast_ma') - pl.col('slow_ma')).alias(col_names[0])).drop('fast_ma','slow_ma')
    lf = ma_func(lf, col_names[0], c, col_names[1])
    return lf

def calculate_rsi(lf: pl.LazyFrame, eval_var: str, n: int, ma_type: str) -> pl.LazyFrame:
    """
    Calculate RSI (RSI value range is 0-100)
    Any number above 70 should be considered overbought (OB).
    Any number below 30 should be considered oversold (OS).
    """
    col_name = f'{eval_var}_RSI_{n}'
    ma_func = ma_type_func_maping[ma_type]
    lf = lf.with_columns((pl.col(eval_var) - pl.col(eval_var).shift(n=1)).alias('change'))
    lf = lf.with_columns(pl.when(pl.col('change') >= 0).then(pl.col('change')).otherwise(0).alias('gain'),
                        pl.when(pl.col('change') < 0).then(pl.col('change')*(-1)).otherwise(0).alias('loss'))
    lf = ma_func(lf, 'gain', n, 'avg_gain')
    lf = ma_func(lf, 'loss', n, 'avg_loss')
    lf = lf.with_columns((100 - (100/(1 + (pl.col('avg_gain')/pl.col('avg_loss'))))).alias(col_name)).drop('change','gain','loss','avg_gain','avg_loss')
    return lf
