import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Setup Path Absolut dan Load Pustaka Model Keras
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'live_telemetry.csv')

print("⏳ Memuat Model Inteligensia Buatan CNN-LSTM...")
model_suhu = tf.keras.models.load_model(os.path.join(BASE_DIR, 'best_model_suhu.h5'), compile=False)
model_ph = tf.keras.models.load_model(os.path.join(BASE_DIR, 'best_model_ph.h5'), compile=False)
model_do = tf.keras.models.load_model(os.path.join(BASE_DIR, 'best_model_do.h5'), compile=False)

scaler_suhu = joblib.load(os.path.join(BASE_DIR, 'scaler_suhu.pkl'))
scaler_ph = joblib.load(os.path.join(BASE_DIR, 'scaler_ph.pkl'))
scaler_do = joblib.load(os.path.join(BASE_DIR, 'scaler_do.pkl'))

lookback = 60
THRESHOLD_RESIDU_DO = 0.8
THRESHOLD_RESIDU_PH = 0.3

# Setup Kanvas Live Plotting Matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Dashboard Monitoring Dinamis MISO Kolam Nila - CNN-LSTM Real-Time", fontsize=14, fontweight='bold')

# List untuk menampung riwayat visualisasi plot bergerak
hist_aktual_suhu, hist_pred_suhu = [], []
hist_aktual_ph, hist_pred_ph = [], []
hist_aktual_do, hist_pred_do = [], []

def update_live_graph(frame):
    global hist_aktual_suhu, hist_pred_suhu, hist_aktual_ph, hist_pred_ph, hist_aktual_do, hist_pred_do
    
    if not os.path.exists(CSV_PATH):
        return
    
    # Membaca data log telemetri terbaru
    df = pd.read_csv(CSV_PATH)
    if len(df) < lookback:
        return
        
    # Ambil 60 data paling bontot (Sliding Window Buffer)
    recent_data = df.tail(lookback)
    
    val_suhu = recent_data['SUHU'].values
    val_ph = recent_data['PH'].values
    val_do = recent_data['DO'].values
    
    # Ambil titik data terkini (T)
    current_suhu = val_suhu[-1]
    current_ph = val_ph[-1]
    current_do = val_do[-1]
    
    # 2. Pemrosesan Bentuk Input Scaling untuk AI
    in_suhu = scaler_suhu.transform(val_suhu.reshape(-1, 1)).reshape(1, lookback, 1)
    in_ph = scaler_ph.transform(val_ph.reshape(-1, 1)).reshape(1, lookback, 1)
    in_do = scaler_do.transform(val_do.reshape(-1, 1)).reshape(1, lookback, 1)
    
    # 3. Eksekusi Prediksi T+10
    pred_suhu = scaler_suhu.inverse_transform(model_suhu.predict(in_suhu, verbose=0))[0][0]
    pred_ph = scaler_ph.inverse_transform(model_ph.predict(in_ph, verbose=0))[0][0]
    pred_do = scaler_do.inverse_transform(model_do.predict(in_do, verbose=0))[0][0]
    
    # 4. Hitung Analisis Kontrol (Offset & Residu)
    offset_do = current_do - pred_do
    residu_do = abs(offset_do)
    residu_ph = abs(current_ph - pred_ph)
    
    # Simpan histori pergerakan grafik dinamis (Batasi maksimal 40 data terakhir di layar agar tidak penuh)
    hist_aktual_suhu.append(current_suhu)
    hist_pred_suhu.append(pred_suhu)
    hist_aktual_ph.append(current_ph)
    hist_pred_ph.append(pred_ph)
    hist_aktual_do.append(current_do)
    hist_pred_do.append(pred_do)
    
    if len(hist_aktual_suhu) > 40:
        hist_aktual_suhu.pop(0); hist_pred_suhu.pop(0)
        hist_aktual_ph.pop(0); hist_pred_ph.pop(0)
        hist_aktual_do.pop(0); hist_pred_do.pop(0)
        
    # Clear sumbu grafik lama untuk menggambar garis baru
    ax1.clear(); ax2.clear(); ax3.clear()
    
    # Subplot 1: SUHU
    ax1.plot(hist_aktual_suhu, label="Suhu Aktual (T)", color='gray', marker='o', markersize=3)
    ax1.plot(hist_pred_suhu, label="Prediksi AI (T+10)", color='red', linestyle='--')
    ax1.set_ylabel("Suhu (°C)", fontweight='bold')
    ax1.legend(loc="upper left")
    
    # Subplot 2: pH
    ax2.plot(hist_aktual_ph, label="pH Aktual (T)", color='gray', marker='o', markersize=3)
    ax2.plot(hist_pred_ph, label="Prediksi AI (T+10)", color='green', linestyle='--')
    ax2.set_ylabel("Kadar pH", fontweight='bold')
    ax2.legend(loc="upper left")
    if residu_ph > THRESHOLD_RESIDU_PH:
        ax2.set_facecolor('#ffcccc') # Beri warna latar merah muda jika pH anomali
        
    # Subplot 3: Dissolved Oxygen (DO)
    ax3.plot(hist_aktual_do, label="DO Aktual (T)", color='gray', marker='o', markersize=3)
    ax3.plot(hist_pred_do, label="Prediksi AI (T+10)", color='blue', linestyle='--')
    ax3.set_ylabel("DO (mg/L)", fontweight='bold')
    ax3.set_xlabel("Siklus Pengiriman Data Sensor (Perlarian Waktu)", fontweight='bold')
    ax3.legend(loc="upper left")
    if residu_do > THRESHOLD_RESIDU_DO:
        ax3.set_facecolor('#ffcccc') # Beri warna latar merah muda jika DO drop/anomali

    # Beri anotasi status teks di grafik DO secara langsung
    status_text = f"Offset DO: {offset_do:+.2f} | Status: {'🚨 ANOMALI CRITICAL' if residu_do > THRESHOLD_RESIDU_DO else '✅ SYSTEM AMAN'}"
    ax3.set_title(status_text, color='red' if residu_do > THRESHOLD_RESIDU_DO else 'green', fontsize=11, fontweight='bold')
    
    plt.tight_layout()

# Loop Animasi Matplotlib: Memanggil fungsi update_live_graph setiap 2000 milidetik (2 detik)
ani = FuncAnimation(fig, update_live_graph, interval=2000, cache_frame_data=False)
plt.show()