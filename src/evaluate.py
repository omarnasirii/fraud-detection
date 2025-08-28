"""
Utility script to load saved models and evaluate on test split (prints metrics).
"""
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from utils import load_data_csv, load_model_sklearn
from preprocess import preprocess_for_model
import tensorflow as tf

def evaluate_isolationforest():
    print("Evaluating Isolation Forest...")
    df = load_data_csv()
    X, y = preprocess_for_model(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    iso = load_model_sklearn("isolation_forest")
    y_pred_raw = iso.predict(X_test)
    y_pred = np.array([1 if x == -1 else 0 for x in y_pred_raw])
    print("IsolationForest Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

def evaluate_autoencoder():
    print("Evaluating Autoencoder...")
    df = load_data_csv()
    X, y = preprocess_for_model(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Correct variable name here
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    MODEL_FILE = os.path.join(MODEL_DIR, "autoencoder.keras")  # <-- use MODEL_DIR, not model_dir

    autoencoder = tf.keras.models.load_model(MODEL_FILE)

    # Evaluate
    recon = autoencoder.predict(X_test)
    mse = np.mean(np.square(X_test - recon), axis=1)

    # Compute threshold from training normal subset
    X_train_norm = X_train[y_train == 0]
    recon_train = autoencoder.predict(X_train_norm)
    mse_train = np.mean(np.square(X_train_norm - recon_train), axis=1)
    threshold = np.percentile(mse_train, 99)

    y_pred = (mse > threshold).astype(int)
    print("Autoencoder Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Threshold used: {threshold:.6f}")


if __name__ == "__main__":
    evaluate_isolationforest()
    evaluate_autoencoder()
