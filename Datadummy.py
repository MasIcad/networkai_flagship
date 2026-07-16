import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. GENERATE DATA DUMMY RESOLUSI 10 DETIK
# ==========================================
print("1. Men-generate data mentah kolam nila...")
np.random.seed(42)
n_days = 4
data_per_day = 24 * 60 * 6  # 8640 titik per hari (tiap 10 detik)
n_data = data_per_day * n_days
t = np.arange(n_data)

def generate_realistic_pond_data(t):
    # 1. SUHU: Siklus Diurnal Alami yang Sangat Lembut
    suhu_base = 27.5 + 2.5 * np.sin(2 * np.pi * (t - 3600) / 8640) 
    # Noise diperkecil drastis menjadi 0.01 agar tidak merusak tren
    suhu_raw = suhu_base + np.random.normal(0, 0.01, len(t))

    # 2. pH: Tren harian akibat respirasi & fotosintesis
    limbah_akumulasi = (t / n_data) * 0.3 
    ph_base = 7.4 + 0.5 * np.sin(2 * np.pi * (t - 3600) / 8640) + limbah_akumulasi
    ph_raw = ph_base + np.random.normal(0, 0.005, len(t))

    # 3. DO: Siklus terbalik dari suhu + respirasi subuh
    do_diurnal = 6.5 + 1.8 * np.cos(2 * np.pi * (t - 1800) / 8640)
    efek_suhu = -0.15 * (suhu_base - 27.5) 
    do_raw = do_diurnal + efek_suhu + np.random.normal(0, 0.01, len(t))
    
    # Anomali Tetap Dipertahankan pada Hari Ke-4
    ph_raw[29500:31500] += 0.8
    do_raw[29500:31500] -= 3.2

    # INDIKASI REALISTIS: Terapkan Moving Average (Window=30 titik / 5 menit) 
    # untuk mensimulasikan inersia termal & kimiawi air yang lambat berubah
    suhu = pd.Series(suhu_raw).rolling(window=30, min_periods=1).mean().values
    ph = pd.Series(ph_raw).rolling(window=30, min_periods=1).mean().values
    do = pd.Series(do_raw).rolling(window=30, min_periods=1).mean().values

    return suhu, ph, np.clip(do, 0.2, 10.0)

suhu, ph, do = generate_realistic_pond_data(t)
df = pd.DataFrame({'SUHU': suhu, 'PH': ph, 'DO': do})

# ==========================================
# 2. NORMALISASI / SCALING DATA NUMERIK
# ==========================================
print("2. Melakukan normalisasi data (MinMaxScaler)...")
scaler_suhu = MinMaxScaler()
scaler_ph = MinMaxScaler()
scaler_do = MinMaxScaler()

suhu_scaled = scaler_suhu.fit_transform(df[['SUHU']].values)
ph_scaled = scaler_ph.fit_transform(df[['PH']].values)
do_scaled = scaler_do.fit_transform(df[['DO']].values)

# Simpan objek scaler untuk dipakai di Jetson Nano nanti
joblib.dump(scaler_suhu, 'scaler_suhu.pkl')
joblib.dump(scaler_ph, 'scaler_ph.pkl')
joblib.dump(scaler_do, 'scaler_do.pkl')

# ==========================================
# 3. WINDOWING (HORIZON T+30 LANGKAH = 5 MENIT)
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

# Pembagian data: 80% untuk Training, 20% untuk Testing
split = int(0.8 * len(X_suhu))

# Split Pasangan Suhu
X_train_suhu, X_test_suhu = X_suhu[:split], X_suhu[split:]
y_train_suhu, y_test_suhu = y_suhu[:split], y_suhu[split:]

# Split Pasangan pH
X_train_ph, X_test_ph = X_ph[:split], X_ph[split:]
y_train_ph, y_test_ph = y_ph[:split], y_ph[split:]

# Split Pasangan DO
X_train_do, X_test_do = X_do[:split], X_do[split:]
y_train_do, y_test_do = y_do[:split], y_do[split:]

# ==========================================
# 4. MENYIMPAN MATRIKS NUMERIK KE FILE PHYSICAL (.npy)
# ==========================================
print("4. Menyimpan matriks numerik ke dalam file npy (Termasuk Suhu)...")
# Save file data Suhu
np.save('X_train_suhu.npy', X_train_suhu)
np.save('y_train_suhu.npy', y_train_suhu)
np.save('X_test_suhu.npy', X_test_suhu)
np.save('y_test_suhu.npy', y_test_suhu)

# Save file data pH
np.save('X_train_ph.npy', X_train_ph)
np.save('y_train_ph.npy', y_train_ph)
np.save('X_test_ph.npy', X_test_ph)
np.save('y_test_ph.npy', y_test_ph)

# Save file data DO
np.save('X_train_do.npy', X_train_do)
np.save('y_train_do.npy', y_train_do)
np.save('X_test_do.npy', X_test_do)
np.save('y_test_do.npy', y_test_do)

print("\n[SUKSES] Seluruh file .npy (Suhu, pH, DO) berhasil diperbarui dan disimpan!")