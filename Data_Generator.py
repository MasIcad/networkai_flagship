import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

print("1. Men-generate data mentah kolam nila biologis (Resolusi 1 Menit)...")
np.random.seed(42)
n_days = 4
data_per_day = 24 * 60  # 1440 titik per hari
n_data = data_per_day * n_days
t = np.arange(n_data)

# Setup Basis Waktu Riil per Menit (Asumsi data dimulai dari tengah malam 00:00)
start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
timestamps = [start_time + timedelta(minutes=int(i)) for i in t]

def generate_biologically_accurate_data(t):
    # Menit ke- berulang dalam satu hari (0 s/d 1439)
    menit_hari = t % 1440

    # 1. SUHU: Puncak jam 14:00 (menit 840), terendah jam 05:00 subuh
    suhu_base = 27.5 + 2.5 * np.sin(2 * np.pi * (menit_hari - 480) / 1440)
    suhu_raw = suhu_base + np.random.normal(0, 0.01, len(t))

    # TREN AKUMULASI LIMBAH HARIAN (Efek feses harian yang menumpuk perlahan)
    limbah_kumulatif = (t / n_data) * 0.3

    # 2. pH: Memuncak di sore hari (jam 15.30 / menit 930) akibat serapan CO2 oleh fotosintesis
    ph_diurnal = 0.4 * np.sin(2 * np.pi * (menit_hari - 570) / 1440)
    ph_base = 7.4 + ph_diurnal + limbah_kumulatif
    ph_raw = ph_base + np.random.normal(0, 0.005, len(t))

    # 3. DO: Memuncak di sore hari karena fotosintesis, drop parah di subuh hari akibat respirasi
    do_diurnal = 6.5 + 2.0 * np.sin(2 * np.pi * (menit_hari - 570) / 1440)
    
    # EFEK PAKAN BERSANDARKAN WAKTU (BOD memuncak 2-3 jam pasca pemberian pakan jam 09:00 dan 16:00)
    # Konsumsi oksigen oleh bakteri pembongkar amonia feses/sisa pakan
    efek_pakan_pagi = -0.35 * np.exp(-((menit_hari - 750) / 150) ** 2)   # Reaksi puncak jam 12.30 siang
    efek_pakan_sore = -0.50 * np.exp(-((menit_hari - 1170) / 180) ** 2)  # Reaksi puncak jam 19.30 malam
    
    do_base = do_diurnal + efek_pakan_pagi + efek_pakan_sore - 0.1 * (suhu_base - 27.5)
    do_raw = do_base + np.random.normal(0, 0.01, len(t))
    
    # ==========================================================
    # ANOMALI REALISTIS: Ledakan Amonia / Kotoran di Hari ke-4
    # Rentang indeks menit 5000 s/d 5200 (Hari ke-4, pukul 07:20 s/d 10:40 pagi)
    # ==========================================================
    ph_raw[5000:5200] += 0.85  # Amonia bebas melonjakkan kebasaan air
    do_raw[5000:5200] -= 3.40  # Decomposers menyedot DO habis-habisan hingga drop kritis

    # Filter Moving Average (Window=5 Menit) untuk inersia air
    suhu = pd.Series(suhu_raw).rolling(window=5, min_periods=1).mean().values
    ph = pd.Series(ph_raw).rolling(window=5, min_periods=1).mean().values
    do = pd.Series(do_raw).rolling(window=5, min_periods=1).mean().values

    return suhu, ph, np.clip(do, 0.1, 10.0)

suhu, ph, do = generate_biologically_accurate_data(t)

# DataFrame Hasil Akhir
df_view = pd.DataFrame({
    'TIMESTAMP': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
    'SUHU': suhu,
    'PH': ph,
    'DO': do
})
df_view.to_csv('aquaculture_data_view.csv', index=False)
print("-> Berkas inspeksi angka & timestamp sukses diekspor ke 'aquaculture_data_view.csv'")

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
# 3. WINDOWING (HORIZON T+30 MENIT KE DEPAN)
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

# Menyimpan Data dengan Pembagian Matriks secara Akurat (Fix Typo DO)
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

print(f"\n[SUKSES] Seluruh berkas biner diperbarui secara ilmiah dengan total {n_data} baris.")