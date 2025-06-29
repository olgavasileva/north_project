import pandas as pd
from src.utils import data_manager as dm


def get_daily_df(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Filter df to just the rows matching the given date string (YYYY-MM-DD)."""
    return df[df.index.date == pd.to_datetime(date_str).date()]


def get_full_dataset_insights(df: pd.DataFrame, model=None):
    """Returns both correlation matrix and top SHAP features for full dataset."""
    X = df.drop(columns=["mental_health_status"])
    return {
        "shap_values": dm.get_shap_values(X, model),
        "correlation_matrix": dm.get_correlation_matrix(df).to_dict()
    }


def get_daily_insights(df: pd.DataFrame, date_str: str, model=None):
    """Same as above, scoped to a single day's data."""
    daily_df = get_daily_df(df, date_str)
    if daily_df.empty:
        return {"error": f"No data found for date: {date_str}"}
    X = daily_df.drop(columns=["mental_health_status"])
    return {
        "shap_values": dm.get_shap_values(X, model),
        "correlation_matrix": dm.get_correlation_matrix(daily_df).to_dict()
    }