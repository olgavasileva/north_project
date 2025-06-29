import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
import re


TARGET = 'mental_health_status'

BASE_FEATURES = [
    'location_id', 'temperature_celsius', 'humidity_percent', 'air_quality_index',
    'noise_level_db', 'lighting_lux', 'crowd_density',
    'stress_level', 'sleep_hours', 'mood_score'
]

INDEX = 'timestamp'

#
def remove_target_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces target=2 with the higher of its neighbors (assumed to be an outlier).
    """

    df = df.copy()
    outlier_indices = df[df[TARGET] == 2].index

    for ts in outlier_indices:
        idx = df.index.get_loc(ts)  # Get integer position of timestamp

        if 0 < idx < len(df) - 1: # make sure we exclude the last point (has no neighbors)
            before_val = df.iloc[idx - 1][TARGET]
            after_val = df.iloc[idx + 1][TARGET]
            df.at[ts, TARGET] = max(before_val, after_val)

    return df


#
def flip_outlier_sign(df: pd.DataFrame, col='mood_score', iqr_coef=1.5) -> pd.DataFrame:
    """
    Fixes possible sign-flip errors in a column based on IQR + neighbor values.
    For example, a few outliers (and their left and right neighbor) 
    in mood_score look like 1.4 -2.2 3.0 or 2.1 -1.7 0.5, which makes it appear as if the sign was flipped
    """
    df = df.copy()

    q25, q75 = df[col].quantile([0.25, 0.75])
    iqr      = q75 - q25
    lq, uq   = q25 - iqr_coef * iqr, q75 + iqr_coef * iqr

    outlier_mask = (df[col] < lq) | (df[col] > uq)
    mc_outlier_timestamps = df.index[outlier_mask]

    for ts in mc_outlier_timestamps:
        idx = df.index.get_loc(ts)
        val_before  = df.iloc[idx-1]['mood_score'] #if there were missing timepoints we should've handled this by time delta
        val_after   = df.iloc[idx+1]['mood_score'] #instead of idx-1 or idx+1
        val_current = df.iloc[idx]['mood_score'] 
        neighbor_sign = np.sign(val_before + val_after)
        if min(abs(val_after), abs(val_before)) <= abs(val_current) <= max(abs(val_after), abs(val_before)):
            if np.sign(val_current) != neighbor_sign and neighbor_sign != 0:
                df.iloc[idx, df.columns.get_loc('mood_score')] *= -1
    return df


#
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds hour, weekday, weekend, and cyclic encodings.
    """
    df = df.copy()

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df


#
def add_acf_lag_features(df, cols, period='', max_lag=48) -> pd.DataFrame:
    """
    Adds rolling average features based on ACF cutoff
    """
    df = df.copy()
    for col in cols:
        series = df[col].resample(period).mean() if period else df[col]
        series = series.dropna()

        adjusted_lag = min(max_lag, (len(series) // 3) - 1)
        if adjusted_lag < 1:
            continue 

        try:
            acf_vals = acf(series, nlags=adjusted_lag, fft=True)
        except Exception as e:
            print(f"ACF failed for {col}: {e}")
            continue

        threshold  = 1.96 / np.sqrt(len(series)) # == 95% CI
        cutoff_lag = next((i for i, v in enumerate(acf_vals[1:], start=1) if abs(v) < threshold), 4)

        if cutoff_lag > 1:
            if period:
                td = pd.Timedelta(period)
                mins = int(td.total_seconds() / 60)
                cutoff_lag = int(cutoff_lag * mins/15)
            new_col = f'{col}_ma_lag_{cutoff_lag}'
            df[new_col] = df[col].rolling(window=cutoff_lag, min_periods=1).mean()

    return df


#
def add_pacf_lag_features(df, cols, period='', max_lag=48) -> pd.DataFrame:
    """
    Adds direct lag features based on PACF significant lags.
    """
    df = df.copy()

    for col in cols:
        series = df[col].resample(period).mean() if period else df[col]
        series = series.dropna()

        adjusted_lag = min(max_lag, (len(series) // 3) - 1)
        if adjusted_lag < 1:
            continue 

        try:
            pacf_vals = pacf(series, nlags=adjusted_lag)
        except Exception as e:
            print(f"PACF failed for {col}: {e}")
            continue

        threshold = 1.96 / np.sqrt(len(series))  # 95% confidence
        sig_lags = [i for i, v in enumerate(pacf_vals[1:], start=1) if abs(v) > threshold]

        for lag in sig_lags:
            if period: 
                td = pd.Timedelta(period)
                mins = int(td.total_seconds() / 60)
                lag = int(lag * mins/15)
            new_col = f'{col}_lag_{lag}'
            df[new_col] = df[col].shift(lag)

    return df



#
def generate_required_lags(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """
    Ensures the DataFrame includes all expected lag features used in the model.
    Creates any missing lag features.
    Removes any features not in feature_list.
    """
    df = df.copy()
    import re

    lag_pattern = re.compile(r"(.+?)_lag_(\d+)$")

    for feature in feature_list:
        match = lag_pattern.match(feature)
        if match:
            base_col, lag = match.groups()
            lag = int(lag)
            if base_col in df.columns:
                df[feature] = df[base_col].shift(lag)
            else:
                df[feature] = np.nan
        elif feature not in df.columns:
            df[feature] = np.nan

    # Keep only model features
    keep_cols = feature_list + [TARGET] if TARGET in df.columns else feature_list
    return df[keep_cols]



#
def preprocess(df: pd.DataFrame, model_features=[], lag_cols=None) -> pd.DataFrame:
    """Main preprocessing pipeline: outlier fixing, time features, ACF/PACF-based lags."""
    #if lag_cols is None:
    #    lag_cols = [f for f in BASE_FEATURES if f != 'location_id'] + [TARGET]
    df = df.copy()

    df = df.drop(columns=['processed'], errors='ignore')

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    return (
        df
        .pipe(remove_target_outliers)
        .pipe(flip_outlier_sign)
        .pipe(add_time_features)
        .pipe(generate_required_lags, feature_list=model_features)
        #.pipe(add_acf_lag_features, cols=lag_cols, period='')
        #.pipe(add_acf_lag_features, cols=lag_cols, period='1h')
        #.pipe(add_pacf_lag_features, cols=lag_cols, period='')
        #.pipe(add_pacf_lag_features, cols=lag_cols, period='1h')
        .dropna()
    )