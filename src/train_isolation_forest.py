"""
Train and save an Isolation Forest anomaly detector on the credit card dataset.
Saves model to models/isolation_forest.joblib
"""
import os
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

from utils import load_data_csv, save_model_sklearn, ensure_models_dir
from preprocess import preprocess_for_model

ensure_models_dir()

def main():
    print("Loading data...")
    df = load_data_csv()
    X, y = preprocess_for_model(df, fit_scaler=True)    # Isolation Forest is unsupervised: we'll train on `X_train` that is mostly normal transactions
    # To reduce contamination, undersample fraud in training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Train IsolationForest on training set (assume training set contains mostly legit)
    print("Training IsolationForest...")
    contamination = y.mean()  # fraction of fraud in data (very small)
    iso = IsolationForest(n_estimators=200, contamination=y.mean(), random_state=42, n_jobs=-1)
    iso.fit(X_train)

    print("Predicting on test set...")
    y_pred_raw = iso.predict(X_test)
    # map iso output: -1 -> anomaly (fraud), 1 -> normal
    y_pred = np.array([1 if x == -1 else 0 for x in y_pred_raw])  # 1 = fraud to match y

    print("Evaluation:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model_path = save_model_sklearn(iso, name="isolation_forest")
    print(f"IsolationForest model saved to {model_path}")

if __name__ == "__main__":
    main()
