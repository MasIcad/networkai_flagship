import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. GENERATE DATA DUMMY RESOLUSI 1 MENIT
# ==========================================
print("1. Men-generate data mentah kolam nila (Resolusi 1 Menit)...")
np.random.seed(42)
n_days = 4
data_per_day = 24 * 60  # 1440 titik per hari (tiap 1 menit sesuai riil sensor)
n_data = data_per_day * n_days
t = np.arange(n_data)

# Setup Basis Waktu Riil per Menit
start_time = datetime.now()
timestamps = [start_time + timedelta(minutes=int(i)) for i in t]

def generate_realistic_pond_data(t):
    # 1. SUHU: Siklus Diurnal Alami (1 siklus = 1440 menit)
    suhu_base = 27.5 + 2.5 * np.sin(2 * np.pi * (t - 360) / 1440) 
    suhu_raw = suhu_base + np.random.normal(0, 0.01, len(t))

    # 2. SIKLUS BIO-AMONIA HARIAN (Efek sisa pakan & feses harian)
    limbah_kumulatif = (t / n_data) * 0.25
    # Siklus harian memuncak di sore hari
    siklus_amonia_harian = 0.15 * np.sin(2 * np.pi * (t - 600) / 1440)

    # 3. pH: Pengaruh tren harian + akumulasi limbah
    ph_base = 7.4 + siklus_amonia_harian + limbah_kumulatif
    ph_raw = ph_base + np.random.normal(0, 0.005, len(t))

    # 4. DO: Terbalik dari suhu + beban respirasi biologi
    do_diurnal = 6.5 + 1.8 * np.cos(2 * np.pi * (t - 180) / 1440)
    efek_suhu = -0.15 * (suhu_base - 27.5)
    efek_respirasi_limbah = -0.2 * (siklus_amonia_harian + limbah_kumulatif)
    do_raw = do_diurnal + efek_suhu + efek_respirasi_limbah + np.random.normal(0, 0.01, len(t))
    
    # Anomali Kritis Bom Amonia Parah pada Hari Ke-4 (Disesuaikan skala menitnya)
    # Menit ke 5000 s/d 5200 (berada di rentang Hari ke-4)
    ph_raw[5000:5200] += 0.8
    do_raw[5000:5200] -= 3.2

    # Penerapan Moving Average (Window=5 titik / 5 menit) untuk inersia air skala menit
    suhu = pd.Series(suhu_raw).rolling(window=5, min_periods=1).mean().values
    ph = pd.Series(ph_raw).rolling(window=5, min_periods=1).mean().values
    do = pd.Series(do_raw).rolling(window=5, min_periods=1).mean().values

    return suhu, ph, np.clip(do, 0.2, 10.0)

suhu, ph, do = generate_realistic_pond_data(t)

# DataFrame Lengkap dengan Timestamp untuk inspeksi manual
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
# 3. WINDOWING (HORIZON T+30 LANGKAH = 30 MENIT KE DEPAN)
# ==========================================
# Catatan: Karena sekarang 1 titik = 1 menit, horizon 30 artinya memprediksi 30 menit ke depan.
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

print(f"\n[SUKSES] Seluruh berkas biner diperbarui! Total titik data: {n_data} baris.")