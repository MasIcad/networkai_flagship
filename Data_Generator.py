import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

print("1. Men-generate data mentah kolam nila (Resolusi 10 Detik)...")
np.random.seed(42)
n_days = 4
data_per_day = 24 * 60 * 6  # 8640 titik per hari (tiap 10 detik)
n_data = data_per_day * n_days
t = np.arange(n_data)

# Setup Basis Waktu Riil
start_time = datetime.now()
timestamps = [start_time + timedelta(seconds=int(i * 10)) for i in t]

def generate_realistic_pond_data(t):
    # 1. SUHU: Siklus Diurnal Alami yang Lembut
    suhu_base = 27.5 + 2.5 * np.sin(2 * np.pi * (t - 2160) / 8640) 
    suhu_raw = suhu_base + np.random.normal(0, 0.01, len(t))

    # 2. SIKLUS BIO-AMONIA HARIAN (Ikan makan & buang kotoran)
    # Limbah kumulatif naik perlahan tiap hari
    limbah_kumulatif = (t / n_data) * 0.25
    # Siklus harian: Amonia memuncak di sore hari (sekitar jam 14.00 - 16.00 / indeks siklus harian)
    # Karena fotosintesis siang hari menyerap CO2, pH naik di siang-sore, berpadu dengan amonia bebas
    siklus_amonia_harian = 0.15 * np.sin(2 * np.pi * (t - 3600) / 8640)

    # 3. pH: Pengaruh tren harian + akumulasi limbah
    ph_base = 7.4 + siklus_amonia_harian + limbah_kumulatif
    ph_raw = ph_base + np.random.normal(0, 0.005, len(t))

    # 4. DO: Terbalik dari suhu + beban respirasi biologi malam/subuh
    do_diurnal = 6.5 + 1.8 * np.cos(2 * np.pi * (t - 1080) / 8640)
    efek_suhu = -0.15 * (suhu_base - 27.5)
    # Amonia harian yang tinggi di sore hari meningkatkan beban kebutuhan oksigen bakteri pengurai (BOD)
    efek_respirasi_limbah = -0.2 * (siklus_amonia_harian + limbah_kumulatif)
    do_raw = do_diurnal + efek_suhu + efek_respirasi_limbah + np.random.normal(0, 0.01, len(t))
    
    # Anomali Kritis Tetap Dipertahankan pada Hari Ke-4 (Menit Kritis)
    ph_raw[29500:31500] += 0.8
    do_raw[29500:31500] -= 3.2

    # Penerapan Moving Average (Window=30 titik / 5 menit) untuk inersia air
    suhu = pd.Series(suhu_raw).rolling(window=30, min_periods=1).mean().values
    ph = pd.Series(ph_raw).rolling(window=30, min_periods=1).mean().values
    do = pd.Series(do_raw).rolling(window=30, min_periods=1).mean().values

    return suhu, ph, np.clip(do, 0.2, 10.0)

suhu, ph, do = generate_realistic_pond_data(t)

# DataFrame Lengkap dengan Timestamp untuk inspeksi manual Anda
df_view = pd.DataFrame({
    'TIMESTAMP': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
    'SUHU': suhu,
    'PH': ph,
    'DO': do
})
df_view.to_csv('aquaculture_data_view.csv', index=False)
print("-> File inspeksi angka & timestamp sukses diekspor ke 'aquaculture_data_view.csv'")

# ==========================================
# 2. NORMALISASI / SCALING DATA NUMERIK
# ==========================================
print("2. Melakukan normalisasi data (MinMaxScaler)...")
scaler_suhu, scaler_ph, scaler_do = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()

suhu_scaled = scaler_suhu.fit_transform(df_view[['SUHU']].values)
ph_scaled = scaler_ph.fit_transform(df_view[['PH']].values)
do_scaled = scaler_do.fit_transform(df_view[['DO']].values)

joblib.dump(scaler_suhu, 'scaler_suhu.pkl')
joblib.dump(scaler_ph, 'scaler_ph.pkl')
joblib.dump(scaler_do, 'scaler_do.pkl')

# ==========================================
# 3. WINDOWING (HORIZON T+30 LANGKAH = 5 MENIT KE DEPAN)
# ==========================================
lookback, horizon = 60, 30

def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback + horizon - 1])
    return np.array(X), np.array(y)

print("3. Memproses windowing ke bentuk matriks numerik...")
X_suhu, y_suhu = create_sequences(suhu_scaled, lookback, horizon)
X_ph, y_ph = create_sequences(ph_scaled, lookback, horizon)
X_do, y_do = create_sequences(do_scaled, lookback, horizon)

split = int(0.8 * len(X_suhu))

# Pematangan pemisahan train/test (.npy)
np.save('X_train_suhu.npy', X_suhu[:split])
np.save('y_train_suhu.npy', y_suhu[:split])
np.save('X_test_suhu.npy', X_suhu[split:])
np.save('y_test_suhu.npy', y_suhu[split:])

np.save('X_train_ph.npy', X_ph[:split])
np.save('y_train_ph.npy', y_ph[:split])
np.save('X_test_ph.npy', X_ph[split:])
np.save('y_test_ph.npy', y_ph[split:])

np.save('X_train_do.npy', X_do[:split])
np.save('y_train_do.npy', y_do[:split])
np.save('X_test_do.npy', X_do[split:])
np.save('y_test_do.npy', y_do[split:])

print("\n[SUKSES] Seluruh berkas fisik biner .npy dan .pkl siap digunakan!")