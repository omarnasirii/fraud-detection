"""
Train a simple dense autoencoder as an anomaly detector.
Saves the model as models/autoencoder.keras
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from utils import load_data_csv
from preprocess import preprocess_for_model

# Save in models directory, not a subfolder
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def build_autoencoder(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(32, activation="relu")(x)
    encoded = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(encoded)
    x = layers.Dense(64, activation="relu")(x)
    decoded = layers.Dense(input_dim, activation="linear")(x)
    autoencoder = models.Model(inputs=inp, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

def main():
    print("Loading data...")
    df = load_data_csv()
    X, y = preprocess_for_model(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

    # Train autoencoder on normal transactions only (y==0)
    X_train_norm = X_train[y_train == 0]
    print("Training autoencoder on normal transactions only...")
    autoencoder = build_autoencoder(X_train_norm.shape[1])
    history = autoencoder.fit(
        X_train_norm, X_train_norm,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        shuffle=True
    )

    # Save model as .keras file
    save_path = os.path.join(MODEL_DIR, "autoencoder.keras")
    autoencoder.save(save_path)
    print(f"Autoencoder saved to {save_path}")

    # Evaluate on test set
    reconstructions = autoencoder.predict(X_test)
    mse = np.mean(np.square(X_test - reconstructions), axis=1)

    # Threshold from training errors
    recon_train = autoencoder.predict(X_train_norm)
    mse_train = np.mean(np.square(X_train_norm - recon_train), axis=1)
    threshold = np.percentile(mse_train, 99)

    y_pred = (mse > threshold).astype(int)
    print("Evaluation (autoencoder):")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Threshold used: {threshold:.6f}")

if __name__ == "__main__":
    main()
