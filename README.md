# 🔍 Fraud Detection in Transactions

An interactive **fraud detection system** for financial transactions using **machine learning anomaly detection models**.  
Built with **Python (scikit-learn, Pandas, Streamlit) and SQL**.  

---

## 🚀 Features
- Uses the popular **Credit Card Fraud Dataset** from Kaggle.
- Implements **Isolation Forest** and **Autoencoder** (deep learning) for anomaly detection.
- Stores transaction data in a **SQLite database**.
- Provides an **interactive Streamlit dashboard** for:
  - Viewing transaction distributions.
  - Highlighting anomalies (potential fraud cases).
  - Comparing model performance with precision, recall, and F1-score.
- Visualizes fraud vs. non-fraud transactions using **Plotly**.

## 📂 Project Structure

<img width="306" height="445" alt="Screenshot 2025-08-28 200147" src="https://github.com/user-attachments/assets/bca97c84-dbed-436b-9d8e-1e491d0d4cce" />

## ⚙️ Setup Instructions

### 1. Clone Repository

git clone https://github.com/omarnasirii/fraud-detection.git

cd fraud-detection

### 2. Create Virtual Environment

python -m venv venv

# Mac/Linux
source venv/bin/activate
# Windows
.\venv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

### 5. Download Dataset

Download the Credit Card Fraud Dataset
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?utm_source=chatgpt.com
 and place it in a folder named data/.

mkdir data
mv creditcard.csv data/

5. Preprocess Data & Train Models
python fetch_data.py
python train_model.py


This will store cleaned transactions in fraud.db and save trained models (isolation_forest.pkl, autoencoder.h5).

### 6. Run Dashboard
streamlit run app.py

## 📊 Example Dashboard

Upload or select transaction dataset.

Choose detection model: Isolation Forest or Autoencoder.

View fraud detection results in real time.

Analyze model performance with metrics and confusion matrix.

## 🛡️ Notes

.gitignore excludes fraud.db, model files, and dataset.

Database and models are generated locally, not stored in GitHub.

Suitable for experimenting with real-world fintech fraud detection workflows.

## 🔮 Future Improvements

Add real-time transaction scoring via API integration.

Support streaming data (Kafka, AWS Kinesis).

Extend dashboard to visualize time-based fraud trends.
