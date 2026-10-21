import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. DATA UNDERSTANDING: Membuat Dataset Sintetis Perusahaan E-Commerce
np.random.seed(42)
n_samples = 200

data_ecommerce = {
    'Customer_ID': [f"CUST-{i:03d}" for i in range(1, n_samples + 1)],
    'Total_Belanja_Bulanan': np.random.uniform(200000, 15000000, n_samples),
    'Frekuensi_Transaksi': np.random.randint(1, 30, n_samples),
    'Jumlah_Piutang': np.random.uniform(0, 5000000, n_samples),
    'Rata_Keterlambatan_Hari': np.random.randint(0, 60, n_samples),
    'Status_Gagal_Bayar': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
}

df_s1 = pd.DataFrame(data_ecommerce)

# Menambahkan Missing Value secara acak untuk mensimulasikan data kotor di lapangan
df_s1.loc[10:15, 'Total_Belanja_Bulanan'] = np.nan
df_s1.loc[25:28, 'Jumlah_Piutang'] = np.nan

print("=== 1. DATASET MENTAH (5 BARIS PERTAMA) ===")
print(df_s1.head())

# 2. DATA PREPARATION: Cleaning & Feature Engineering
# Mengisi nilai kosong (Imputasi)
df_s1['Total_Belanja_Bulanan'].fillna(df_s1['Total_Belanja_Bulanan'].median(), inplace=True)
df_s1['Jumlah_Piutang'].fillna(0, inplace=True)

# Feature Engineering: Menghitung Rasio Piutang terhadap Total Belanja
df_s1['Rasio_Piutang_Belanja'] = df_s1['Jumlah_Piutang'] / (df_s1['Total_Belanja_Bulanan'] + 1)

# Pemisahan Fitur dan Target
X_s1 = df_s1[['Total_Belanja_Bulanan', 'Frekuensi_Transaksi', 'Jumlah_Piutang', 'Rata_Keterlambatan_Hari', 'Rasio_Piutang_Belanja']]
y_s1 = df_s1['Status_Gagal_Bayar']

# Feature Scaling
scaler_s1 = StandardScaler()
X_s1_scaled = scaler_s1.fit_transform(X_s1)

# 3. MODELING: Split Data & Pelatihan Model Random Forest
X_train_s1, X_test_s1, y_train_s1, y_test_s1 = train_test_split(X_s1_scaled, y_s1, test_size=0.25, random_state=42)

model_s1 = RandomForestClassifier(n_estimators=100, random_state=42)
model_s1.fit(X_train_s1, y_train_s1)

# 4. EVALUATION
y_pred_s1 = model_s1.predict(X_test_s1)

print("\n=== EVALUASI MODEL PREDIKSI GAGAL BAYAR ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test_s1, y_pred_s1))
print("\nClassification Report:")
print(classification_report(y_test_s1, y_pred_s1))
