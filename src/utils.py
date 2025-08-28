import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data_csv(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
    df = pd.read_csv(path)
    return df

def save_model_sklearn(model, name="isolation_forest"):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    return path

def load_model_sklearn(name="isolation_forest"):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    return joblib.load(path)

def save_dataframe_to_sql(df, table_name="transactions", db_url="sqlite:///../data/fraud.db"):
    """
    Saves dataframe (appends or replaces) to a SQL database. db_url examples:
    - sqlite:///../data/fraud.db
    - postgresql://user:pass@localhost:5432/fraudd
    """
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    return True

def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)
    return MODELS_DIR
