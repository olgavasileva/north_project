import shap
import pandas as pd
import shap
from src.model.model_runner import load_model
import numpy as np


#
def get_shap_values(X: pd.DataFrame, model=None, n_feat=15):
    """
    Compute SHAP values for top N features.
    """
    if model is None:
        model = load_model()

    explainer    = shap.Explainer(model, X)
    shap_values  = explainer(X)
    shap_summary = np.abs(shap_values.values).mean(axis=0)

    top_idx = shap_summary.argsort()[-n_feat:][::-1]
    
    return {
        X.columns[i]: round(float(shap_summary[i]), 4)
        for i in top_idx
    }


#
def get_correlation_matrix(df: pd.DataFrame, n_feat=15) -> pd.DataFrame:
    """
    Correlation matrix of averaged time-of-day features.
    """
    df = df.copy()
    df['time_of_day'] = df.index.time
    avg_by_time = df.groupby('time_of_day').mean(numeric_only=True)
    corr_matrix = avg_by_time.corr()
    
    if 'mental_health_status' not in corr_matrix:
        return {}

    correlations = corr_matrix['mental_health_status'].drop('mental_health_status')
    top_corr = correlations.reindex(correlations.abs().sort_values(ascending=False).index).head(n_feat)

    return {k: round(v, 4) for k, v in top_corr.items()}