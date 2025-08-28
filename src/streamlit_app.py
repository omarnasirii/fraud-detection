"""
Improved Streamlit app for fraud detection demo with better manual input handling.
Addresses the issue where manual entries always predict legit due to missing features.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
from utils import load_model_sklearn, PROJECT_ROOT
from preprocess import load_preprocessed
import joblib
import tensorflow as tf

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

st.title("🔍 Credit Card Fraud Detection Demo")
st.markdown("""
This app demonstrates fraud detection using IsolationForest and Autoencoder models.
**Note**: The models expect all 29 features (V1-V28 + Amount) to work properly.
""")

@st.cache_data
def load_models_and_stats():
    """Load models and compute statistics from training data for feature imputation."""
    iso = None
    ae = None
    scaler = None
    feature_means = None
    
    # Load models
    iso_path = os.path.join(PROJECT_ROOT, "models", "isolation_forest.joblib")
    if os.path.exists(iso_path):
        iso = joblib.load(iso_path)
    
    ae_path = os.path.join(PROJECT_ROOT, "models", "autoencoder.keras")
    if os.path.exists(ae_path):
        try:
            ae = tf.keras.models.load_model(ae_path)
        except Exception as e:
            st.warning(f"Could not load autoencoder: {e}")
            ae = None
    
    # Load scaler
    scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    # Compute feature means from training data for imputation
    data_path = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            # Get normal transactions only (Class == 0) for computing means
            normal_df = df[df['Class'] == 0]
            feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
            feature_means = normal_df[feature_cols].mean().values
        except Exception as e:
            st.warning(f"Could not compute feature statistics: {e}")
            feature_means = np.zeros(29)  # Fallback to zeros
    else:
        feature_means = np.zeros(29)  # Fallback to zeros
    
    return iso, ae, scaler, feature_means

# Load models and statistics
iso_model, ae_model, scaler, normal_means = load_models_and_stats()

# Display model status
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("IsolationForest", "✅ Loaded" if iso_model else "❌ Not Found")
with col2:
    st.metric("Autoencoder", "✅ Loaded" if ae_model else "❌ Not Found")
with col3:
    st.metric("Scaler", "✅ Loaded" if scaler else "❌ Not Found")

st.markdown("---")

# File upload section
st.subheader("📁 Upload Transaction Data")
uploaded = st.file_uploader(
    "Upload CSV with columns: V1..V28,Amount (Time and Class optional)", 
    type=["csv"],
    help="Upload a CSV file containing transaction data with the required features."
)

if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        st.write("**Input preview:**")
        st.dataframe(df_in.head())
        
        # Check if we have the required columns
        required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing_cols = [col for col in required_cols if col not in df_in.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            X_in = load_preprocessed(df_in)
            results_df = df_in.copy()
            
            # IsolationForest predictions
            if iso_model:
                pred_raw = iso_model.predict(X_in)
                pred = np.array([1 if p == -1 else 0 for p in pred_raw])
                results_df["IsolationForest_Prediction"] = pred
                results_df["IsolationForest_Label"] = ["FRAUD" if p == 1 else "LEGIT" for p in pred]
                
                fraud_count = pred.sum()
                st.write(f"**IsolationForest Results:** {fraud_count} out of {len(pred)} transactions flagged as fraud")
            
            # Autoencoder predictions
            if ae_model:
                recon = ae_model.predict(X_in)
                mse = np.mean(np.square(X_in - recon), axis=1)
                
                # Use a threshold (you might want to load this from training)
                threshold = np.percentile(mse, 99)  # Top 1% as anomalies
                ae_pred = (mse > threshold).astype(int)
                
                results_df["Autoencoder_MSE"] = mse
                results_df["Autoencoder_Prediction"] = ae_pred
                results_df["Autoencoder_Label"] = ["FRAUD" if p == 1 else "LEGIT" for p in ae_pred]
                
                ae_fraud_count = ae_pred.sum()
                st.write(f"**Autoencoder Results:** {ae_fraud_count} out of {len(ae_pred)} transactions flagged as fraud (threshold: {threshold:.6f})")
            
            # Display results
            st.write("**Prediction Results:**")
            display_cols = ["IsolationForest_Label"]
            if ae_model:
                display_cols.append("Autoencoder_Label")
                display_cols.append("Autoencoder_MSE")
            
            st.dataframe(results_df[display_cols])
            
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

st.markdown("---")

# Manual input section
st.subheader("✏️ Manual Transaction Entry")

input_method = st.radio(
    "Choose input method:",
    ["Basic (Amount + 2 key features)", "Advanced (All features)", "Smart Fill (Amount + key features with statistical imputation)"],
    help="Basic: Only Amount, V1, V2 - other features set to zero (may not work well)\nAdvanced: All 29 features\nSmart Fill: Key features with others filled using normal transaction statistics"
)

if input_method == "Basic (Amount + 2 key features)":
    st.warning("⚠️ This method sets most features to zero, which may cause inaccurate predictions!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=0.01)
    with col2:
        v1 = st.number_input("V1", value=0.0, step=0.01, help="Principal Component 1")
    with col3:
        v2 = st.number_input("V2", value=0.0, step=0.01, help="Principal Component 2")
    
    if st.button("🔍 Predict (Basic)", type="primary"):
        # Create sample with mostly zeros (problematic approach)
        sample = np.zeros(29)
        sample[0] = v1  # V1
        sample[1] = v2  # V2
        sample[28] = amount  # Amount is last
        
        # Create DataFrame for preprocessing
        col_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
        df_sample = pd.DataFrame([sample], columns=col_names)
        
        try:
            X_sample = load_preprocessed(df_sample)
            
            st.write("**Input Vector (first 10 features):**")
            st.write(f"V1={sample[0]:.3f}, V2={sample[1]:.3f}, V3-V27=0.000, Amount={amount:.2f}")
            
            # Predictions
            if iso_model:
                iso_pred_raw = iso_model.predict(X_sample)
                iso_pred = 1 if iso_pred_raw[0] == -1 else 0
                confidence = abs(iso_model.decision_function(X_sample)[0])
                
                if iso_pred == 1:
                    st.error(f"🚨 **IsolationForest: FRAUD** (confidence: {confidence:.3f})")
                else:
                    st.success(f"✅ **IsolationForest: LEGIT** (confidence: {confidence:.3f})")
            
            if ae_model:
                recon = ae_model.predict(X_sample)
                mse = np.mean(np.square(X_sample - recon))
                threshold = 0.1  # You might want to load this from training
                ae_pred = 1 if mse > threshold else 0
                
                if ae_pred == 1:
                    st.error(f"🚨 **Autoencoder: FRAUD** (MSE: {mse:.6f})")
                else:
                    st.success(f"✅ **Autoencoder: LEGIT** (MSE: {mse:.6f})")
                    
        except Exception as e:
            st.error(f"Error in prediction: {e}")

elif input_method == "Smart Fill (Amount + key features with statistical imputation)":
    st.info("🧠 This method fills missing features with statistics from normal transactions")
    
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=0.01, key="smart_amount")
        v1 = st.number_input("V1", value=0.0, step=0.01, key="smart_v1")
        v2 = st.number_input("V2", value=0.0, step=0.01, key="smart_v2")
        v3 = st.number_input("V3", value=0.0, step=0.01, key="smart_v3")
    
    with col2:
        v4 = st.number_input("V4", value=0.0, step=0.01, key="smart_v4")
        v14 = st.number_input("V14", value=0.0, step=0.01, key="smart_v14", help="Often important for fraud detection")
        v17 = st.number_input("V17", value=0.0, step=0.01, key="smart_v17", help="Often important for fraud detection")
        v12 = st.number_input("V12", value=0.0, step=0.01, key="smart_v12", help="Often important for fraud detection")
    
    if st.button("🔍 Predict (Smart Fill)", type="primary"):
        # Start with normal transaction means
        sample = normal_means.copy()
        
        # Override with user inputs
        sample[0] = v1   # V1
        sample[1] = v2   # V2
        sample[2] = v3   # V3
        sample[3] = v4   # V4
        sample[11] = v12 # V12 (index 11)
        sample[13] = v14 # V14 (index 13)
        sample[16] = v17 # V17 (index 16)
        sample[28] = amount  # Amount
        
        # Create DataFrame
        col_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
        df_sample = pd.DataFrame([sample], columns=col_names)
        
        try:
            X_sample = load_preprocessed(df_sample)
            
            st.write("**Input Summary:**")
            st.write(f"Amount: ${amount:.2f}")
            st.write(f"Specified features: V1={v1:.3f}, V2={v2:.3f}, V3={v3:.3f}, V4={v4:.3f}, V12={v12:.3f}, V14={v14:.3f}, V17={v17:.3f}")
            st.write("Other features filled with normal transaction averages")
            
            # Predictions
            results = []
            if iso_model:
                iso_pred_raw = iso_model.predict(X_sample)
                iso_pred = 1 if iso_pred_raw[0] == -1 else 0
                confidence = abs(iso_model.decision_function(X_sample)[0])
                results.append(("IsolationForest", "FRAUD" if iso_pred == 1 else "LEGIT", f"{confidence:.3f}"))
                
                if iso_pred == 1:
                    st.error(f"🚨 **IsolationForest: FRAUD** (anomaly score: {confidence:.3f})")
                else:
                    st.success(f"✅ **IsolationForest: LEGIT** (anomaly score: {confidence:.3f})")
            
            if ae_model:
                recon = ae_model.predict(X_sample)
                mse = np.mean(np.square(X_sample - recon))
                # Compute threshold from training normal samples (simplified)
                threshold = np.mean(normal_means) * 0.01  # Rough estimate
                ae_pred = 1 if mse > threshold else 0
                results.append(("Autoencoder", "FRAUD" if ae_pred == 1 else "LEGIT", f"{mse:.6f}"))
                
                if ae_pred == 1:
                    st.error(f"🚨 **Autoencoder: FRAUD** (MSE: {mse:.6f}, threshold: {threshold:.6f})")
                else:
                    st.success(f"✅ **Autoencoder: LEGIT** (MSE: {mse:.6f}, threshold: {threshold:.6f})")
            
            # Summary table
            if results:
                st.write("**Prediction Summary:**")
                results_df = pd.DataFrame(results, columns=["Model", "Prediction", "Score"])
                st.dataframe(results_df, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error in prediction: {e}")

else:  # Advanced - All features
    st.info("🔧 Enter all 29 features manually (for expert users)")
    
    # Create input fields for all V features
    v_features = {}
    
    st.write("**V Features (Principal Components):**")
    cols = st.columns(4)
    for i in range(1, 29):
        col_idx = (i-1) % 4
        with cols[col_idx]:
            v_features[f'V{i}'] = st.number_input(f"V{i}", value=0.0, step=0.01, key=f"adv_v{i}")
    
    amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=0.01, key="adv_amount")
    
    if st.button("🔍 Predict (All Features)", type="primary"):
        # Create sample from all inputs
        sample = np.array([v_features[f'V{i}'] for i in range(1, 29)] + [amount])
        
        # Create DataFrame
        col_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
        df_sample = pd.DataFrame([sample], columns=col_names)
        
        try:
            X_sample = load_preprocessed(df_sample)
            
            # Predictions
            if iso_model:
                iso_pred_raw = iso_model.predict(X_sample)
                iso_pred = 1 if iso_pred_raw[0] == -1 else 0
                confidence = abs(iso_model.decision_function(X_sample)[0])
                
                if iso_pred == 1:
                    st.error(f"🚨 **IsolationForest: FRAUD** (anomaly score: {confidence:.3f})")
                else:
                    st.success(f"✅ **IsolationForest: LEGIT** (anomaly score: {confidence:.3f})")
            
            if ae_model:
                recon = ae_model.predict(X_sample)
                mse = np.mean(np.square(X_sample - recon))
                threshold = 0.1  # You might want to compute this properly
                ae_pred = 1 if mse > threshold else 0
                
                if ae_pred == 1:
                    st.error(f"🚨 **Autoencoder: FRAUD** (MSE: {mse:.6f})")
                else:
                    st.success(f"✅ **Autoencoder: LEGIT** (MSE: {mse:.6f})")
                    
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Sidebar with information
st.sidebar.header("ℹ️ Information")
st.sidebar.markdown("""
**Model Requirements:**
- 29 features: V1-V28 + Amount
- Features must be scaled using trained scaler
- Time feature is dropped during preprocessing

**Feature Importance:**
- V14, V17, V12, V10: Often critical for fraud detection
- Amount: Transaction value
- V1-V28: PCA-transformed features from original data

**Why Manual Entry Often Shows 'LEGIT':**
- Missing features filled with zeros
- Models expect realistic feature combinations
- Use 'Smart Fill' for better results
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips for Testing:**")
st.sidebar.markdown("""
- Try large amounts (>$1000) with Smart Fill
- Experiment with V14, V17 values != 0
- Upload real data for best results
- Check model confidence scores
""")