import os
import pandas as pd
import numpy as np
import joblib
import gc
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

lookback = 20
THRESHOLD_RESIDU_DO = 0.8
THRESHOLD_RESIDU_PH = 0.3
THRESHOLD_RESIDU_SUHU = 1.0

# Batas Ambang Kritis Absolut Budidaya (Double Protection Limits)
LIMIT_BAWAH_DO = 5.0   
LIMIT_ATAS_PH  = 8.5    
LIMIT_BAWAH_PH = 6.5    
LIMIT_ATAS_SUHU = 32.0

# Setup Kanvas Live Plotting Matplotlib GUI
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
fig.suptitle("Dashboard Monitoring Dinamis MISO Kolam Nila - CNN-LSTM Real-Time", fontsize=14, fontweight='bold')

# List untuk menampung riwayat visualisasi plot bergerak
hist_aktual_suhu, hist_pred_suhu = [], []
hist_aktual_ph, hist_pred_ph = [], []
hist_aktual_do, hist_pred_do = [], []

def update_live_graph(frame):
    global hist_aktual_suhu, hist_pred_suhu, hist_aktual_ph, hist_pred_ph, hist_aktual_do, hist_pred_do
    
    # FORCE MEMORY CLEAR: Hancurkan sisa graf internal TensorFlow lawas di setiap menit
    tf.keras.backend.clear_session()
    
    if not os.path.exists(CSV_PATH):
        return
    
    # Membaca data log telemetri terbaru
    df = pd.read_csv(CSV_PATH)
    if len(df) < lookback:
        return
        
    # Ambil 20 data paling bontot (Sliding Window Buffer)
    recent_data = df.tail(lookback)
    
    val_suhu = recent_data['SUHU'].values
    val_ph = recent_data['PH'].values
    val_do = recent_data['DO'].values
    
    # Ambil titik data terkini (T)
    current_suhu = val_suhu[-1]
    current_ph = val_ph[-1]
    current_do = val_do[-1]
    
    # Pemrosesan Bentuk Input Scaling untuk AI
    in_suhu = scaler_suhu.transform(val_suhu.reshape(-1, 1)).reshape(1, lookback, 1)
    in_ph = scaler_ph.transform(val_ph.reshape(-1, 1)).reshape(1, lookback, 1)
    in_do = scaler_do.transform(val_do.reshape(-1, 1)).reshape(1, lookback, 1)
    
    # Inferensi Cepat & Aman tanpa memicu Memory Leak
    pred_suhu = scaler_suhu.inverse_transform(model_suhu(in_suhu, training=False).numpy())[0][0]
    pred_ph = scaler_ph.inverse_transform(model_ph(in_ph, training=False).numpy())[0][0]
    pred_do = scaler_do.inverse_transform(model_do(in_do, training=False).numpy())[0][0]
    
    # Hitung Analisis Kontrol Multi-Parameter (Offset & Residu)
    offset_suhu = current_suhu - pred_suhu
    offset_ph = current_ph - pred_ph
    offset_do = current_do - pred_do
    
    residu_suhu = abs(offset_suhu)
    residu_ph = abs(offset_ph)
    residu_do = abs(offset_do)
    
    # Hitung Akurasi Instan Berbasis Persentase Error
    acc_suhu = max(0.0, 100.0 - (residu_suhu / current_suhu * 100)) if current_suhu != 0 else 0
    acc_ph = max(0.0, 100.0 - (residu_ph / current_ph * 100)) if current_ph != 0 else 0
    acc_do = max(0.0, 100.0 - (residu_do / current_do * 100)) if current_do != 0 else 0
    
    # ==========================================
    # IMPLEMENTASI GERBANG LOGIKA DOUBLE PROTECTION (AND)
    # ==========================================
    status_suhu = "ANOMALI" if (residu_suhu > THRESHOLD_RESIDU_SUHU and current_suhu > LIMIT_ATAS_SUHU) else "OK"
    status_ph = "ANOMALI" if (residu_ph > THRESHOLD_RESIDU_PH and (current_ph > LIMIT_ATAS_PH or current_ph < LIMIT_BAWAH_PH)) else "OK"
    status_do = "ANOMALI" if (residu_do > THRESHOLD_RESIDU_DO and current_do < LIMIT_BAWAH_DO) else "OK"
    
    # Simpan histori pergerakan grafik dinamis (Maksimal 40 data terakhir)
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
        
    # CLEAR TOTAL: Bersihkan sumbu lama agar memori tidak menumpuk
    ax1.clear(); ax2.clear(); ax3.clear()
    
    # ----------------------------------------------------------------
    # Subplot 1: SUHU
    # ----------------------------------------------------------------
    ax1.plot(hist_aktual_suhu, label=f"Suhu Aktual: {current_suhu:.2f}°C", color='gray', marker='o', markersize=3)
    ax1.plot(hist_pred_suhu, label=f"Prediksi AI (T+10): {pred_suhu:.2f}°C", color='red', linestyle='--')
    ax1.set_ylabel("Suhu (°C)", fontweight='bold')
    ax1.legend(loc="upper left")
    ax1.set_facecolor('#ffcccc' if status_suhu == "ANOMALI" else '#ffffff')
    title_suhu = f"Suhu Aktual vs Prediksi | Offset: {offset_suhu:+.2f}°C | Akurasi: {acc_suhu:.1f}% | Status: [{status_suhu}]"
    ax1.set_title(title_suhu, color='red' if status_suhu == "ANOMALI" else 'black', fontsize=10, fontweight='bold')
    
    # ----------------------------------------------------------------
    # Subplot 2: KADAR pH
    # ----------------------------------------------------------------
    ax2.plot(hist_aktual_ph, label=f"pH Aktual: {current_ph:.2f}", color='gray', marker='o', markersize=3)
    ax2.plot(hist_pred_ph, label=f"Prediksi AI (T+10): {pred_ph:.2f}", color='green', linestyle='--')
    ax2.set_ylabel("Kadar pH", fontweight='bold')
    ax2.legend(loc="upper left")
    ax2.set_facecolor('#ffcccc' if status_ph == "ANOMALI" else '#ffffff')
    title_ph = f"Kadar pH Aktual vs Prediksi | Offset: {offset_ph:+.2f} | Akurasi: {acc_ph:.1f}% | Status: [{status_ph}]"
    ax2.set_title(title_ph, color='red' if status_ph == "ANOMALI" else 'black', fontsize=10, fontweight='bold')
    
    # ----------------------------------------------------------------
    # Subplot 3: DISSOLVED OXYGEN (DO)
    # ----------------------------------------------------------------
    ax3.plot(hist_aktual_do, label=f"DO Aktual: {current_do:.2f} mg/L", color='gray', marker='o', markersize=3)
    ax3.plot(hist_pred_do, label=f"Prediksi AI (T+10): {pred_do:.2f} mg/L", color='blue', linestyle='--')
    ax3.set_ylabel("DO (mg/L)", fontweight='bold')
    ax3.set_xlabel("Siklus Pengiriman Data Sensor (Resolusi 1 Menit)", fontweight='bold')
    ax3.legend(loc="upper left")
    ax3.set_facecolor('#ffcccc' if status_do == "ANOMALI" else '#ffffff')
    title_do = f"Kadar DO Aktual vs Prediksi | Offset: {offset_do:+.2f} mg/L | Akurasi: {acc_do:.1f}% | Status: [{status_do}]"
    ax3.set_title(title_do, color='red' if status_do == "ANOMALI" else 'black', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Paksa Garbage Collector menyapu RAM sisa variabel tak terpakai
    gc.collect()

# Loop Animasi GUI Matplotlib: Diperbarui setiap 60.000 milidetik (1 Menit)
ani = FuncAnimation(fig, update_live_graph, interval=60000, cache_frame_data=False)
plt.show()