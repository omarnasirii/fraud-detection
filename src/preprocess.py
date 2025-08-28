import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from utils import PROJECT_ROOT

SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.joblib")

def preprocess_for_model(df, drop_time=True, fit_scaler=False):
    data = df.copy()
    y = data["Class"].copy() if "Class" in data.columns else None

    if drop_time and "Time" in data.columns:
        data = data.drop(columns=["Time"])

    X = data.drop(columns=["Class"]) if "Class" in data.columns else data

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)

    return X_scaled, y


def load_preprocessed(df):
    """
    Expects df with columns V1..V28, Amount (Time dropped)
    Returns scaled numpy array
    """
    scaler = joblib.load(SCALER_PATH)
    X = df.copy()
    X = X[['V'+str(i) for i in range(1,29)] + ['Amount']]  # ensure order
    X_scaled = scaler.transform(X)
    return X_scaled
