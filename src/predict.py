import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from preprocess import preprocess_for_model

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ISO_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
AE_PATH = os.path.join(MODEL_DIR, "autoencoder.keras")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")

# Load models
scaler = joblib.load(SCALER_PATH)
iso = joblib.load(ISO_PATH)
autoencoder = tf.keras.models.load_model(AE_PATH)

# Load data
df = pd.read_csv(DATA_PATH)
X, y = preprocess_for_model(df)

# Apply scaler
X_scaled = scaler.transform(X)

# Isolation Forest prediction
y_pred_iso = np.array([1 if x == -1 else 0 for x in iso.predict(X_scaled)])

# Autoencoder prediction
recon = autoencoder.predict(X_scaled)
mse = np.mean(np.square(X_scaled - recon), axis=1)
threshold = np.percentile(mse, 99)
y_pred_ae = (mse > threshold).astype(int)

# Results
results = pd.DataFrame({
    "TransactionID": df.index,
    "IsolationForest": y_pred_iso,
    "Autoencoder": y_pred_ae,
    "Label": y  # optional if you have true labels
})

print("Prediction summary:")
print(f"IsolationForest detected frauds: {y_pred_iso.sum()}")
print(f"Autoencoder detected frauds: {y_pred_ae.sum()}")

results.to_csv(os.path.join(PROJECT_ROOT, "data", "predictions.csv"), index=False)
print("Predictions saved to data/predictions.csv")
