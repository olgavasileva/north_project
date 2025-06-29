import xgboost as xgb
import pandas as pd

MODEL_PATH = "assets/xgb_model.json"


def load_model(path) -> xgb.XGBClassifier:
    """
    Loads a trained XGBoost model from JSON.
    """
    path = path if path else MODEL_PATH
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def predict(model, X: pd.DataFrame) -> pd.Series:
    """
    Predicts the mental health status from features.
    """
    return model.predict(X)