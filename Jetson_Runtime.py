import time
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient

# Bypassing Pylance resolved warning untuk TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    load_model = keras.models.load_model
except ImportError:
    # Ini agar script tidak langsung crash saat di-load Pylance di laptop
    pass

# ==========================================
# 1. KONFIGURASI INTEGRASI & PARAMETER
# ==========================================
# Sesuaikan dengan credential InfluxDB di Jetson Nano laboratoriummu
INFLUX_URL = "http://10.26.48.177:8086" # Sesuaikan port InfluxDB-mu
INFLUX_TOKEN = "YOUR_INFLUXDB_TOKEN"
INFLUX_ORG = "YOUR_ORG"
INFLUX_BUCKET = "telemetry_bucket"

# Target URL API Website Hydrolla
WEB_API_URL = "https://hydrolla.fahhmyalmaliki.uk/api/dashboard-update"

# Parameter Hysteresis sesuai rancangan Colab kamu
THRESHOLD_RESIDU_DO = 0.8
THRESHOLD_RESIDU_PH = 0.3
CRITICAL_BAWAH_DO = 5.0
CRITICAL_ATAS_PH   = 8.5
ZONA_AMAN_DO = 7.0
ZONA_AMAN_PH = 6.5

lookback = 60 # 10 menit ke belakang (60 titik * 10 detik)

# ==========================================
# 2. LOAD MODEL .H5 DAN SCALER .PKL
# ==========================================
print("[INIT] Memuat model AI dan scaler di Jetson Nano...")
model_suhu = load_model('best_model_suhu.h5')
model_ph   = load_model('best_model_ph.h5')
model_do   = load_model('best_model_do.h5')

scaler_suhu = joblib.load('scaler_suhu.pkl')
scaler_ph   = joblib.load('scaler_ph.pkl')
scaler_do   = joblib.load('scaler_do.pkl')

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

# Buffer memori lokal untuk mencatat prediksi masa lalu (hitung residu real-time)
history_predictions_do = {}
history_predictions_ph = {}

relay_do_state = 0
relay_ph_state = 0

# ==========================================
# 3. FUNGSI PIPA DATA (InfluxDB & Web Forwarder)
# ==========================================
def get_latest_60_points():
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -15m)
      |> filter(fn: (r) => r["_measurement"] == "aquaculture")
      |> filter(fn: (r) => r["sensor_id"] == "esp32-kolam-01")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> tail(n: {lookback})
    '''
    result = query_api.query_data_frame(query)
    result = result.sort_values(by='_time', ascending=True)
    
    # Mapping field dari ESP32 (temperature, ph, do) ke label matriks AI
    df_clean = result[['temperature', 'ph', 'do']].rename(
        columns={'temperature': 'SUHU', 'ph': 'PH', 'do': 'DO'}
    )
    return df_clean

def send_to_website(payload):
    try:
        res = requests.post(WEB_API_URL, json=payload, timeout=5)
        if res.status_code == 200:
            print("[WEB] Data sukses terkirim ke website hydrolla.")
    except Exception as e:
        print(f"[WEB ERROR] Gagal forward data ke cloud website: {e}")

# ==========================================
# 4. RUNTIME LOOP INFERENSI KONTINU
# ==========================================
print("[READY] Sistem Kendali AI Hydrolla Aktif...")

while True:
    try:
        # a. Tarik 60 data teraktual dari InfluxDB
        df_live = get_latest_60_points()
        
        if len(df_live) < lookback:
            print(f"[WARN] Data belum cukup ({len(df_live)}/{lookback}). Menunggu...")
            time.sleep(10)
            continue
            
        aktual_suhu_now = df_live['SUHU'].iloc[-1]
        aktual_ph_now   = df_live['PH'].iloc[-1]
        aktual_do_now   = df_live['DO'].iloc[-1]
        
        timestamp_now = datetime.utcnow()
        timestamp_now_str = timestamp_now.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        
        # b. Preprocessing ke format 3D (1, 60, 1)
        suhu_scaled = scaler_suhu.transform(df_live[['SUHU']].values).reshape(1, lookback, 1)
        ph_scaled   = scaler_ph.transform(df_live[['PH']].values).reshape(1, lookback, 1)
        do_scaled   = scaler_do.transform(df_live[['DO']].values).reshape(1, lookback, 1)
        
        # c. Jalankan Prediksi AI 5 Menit ke Depan (T+30 langkah)
        pred_suhu_t30 = scaler_suhu.inverse_transform(model_suhu.predict(suhu_scaled, verbose=0))[0][0]
        pred_ph_t30   = scaler_ph.inverse_transform(model_ph.predict(ph_scaled, verbose=0))[0][0]
        pred_do_t30   = scaler_do.inverse_transform(model_do.predict(do_scaled, verbose=0))[0][0]
        
        timestamp_future = timestamp_now + timedelta(minutes=5)
        timestamp_future_str = timestamp_future.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        
        # Simpan prediksi saat ini ke buffer untuk evaluasi residu 5 menit lagi
        history_predictions_do[timestamp_future_str] = pred_do_t30
        history_predictions_ph[timestamp_future_str] = pred_ph_t30
        
        # d. Hitung Residu Real-Time (Aktual Sekarang vs Hasil Ramalan AI 5 Menit Lalu)
        predict_do_for_now = history_predictions_do.get(timestamp_now_str, aktual_do_now)
        predict_ph_for_now = history_predictions_ph.get(timestamp_now_str, aktual_ph_now)
        
        residu_do = abs(aktual_do_now - predict_do_for_now)
        residu_ph = abs(aktual_ph_now - predict_ph_for_now)
        
        # e. EKSEKUSI LOGIKA PERTAHANAN GANDA HYSTERESIS
        # Hysteresis DO
        if relay_do_state == 0:
            if (residu_do > THRESHOLD_RESIDU_DO) or (pred_do_t30 < CRITICAL_BAWAH_DO):
                relay_do_state = 1
                print(f"[{timestamp_now_str}] DO TRIPPED! Residu: {residu_do:.2f} | Prediksi T+30: {pred_do_t30:.2f}")
        else:
            if aktual_do_now >= ZONA_AMAN_DO:
                relay_do_state = 0

        # Hysteresis pH
        if relay_ph_state == 0:
            if (residu_ph > THRESHOLD_RESIDU_PH) or (pred_ph_t30 > CRITICAL_ATAS_PH):
                relay_ph_state = 1
                print(f"[{timestamp_now_str}] pH TRIPPED! Residu: {residu_ph:.2f} | Prediksi T+30: {pred_ph_t30:.2f}")
        else:
            if aktual_ph_now <= ZONA_AMAN_PH:
                relay_ph_state = 0

        # f. Cetak Log Monitoring di Terminal Jetson
        print(f"[MONITOR] Suhu: {aktual_suhu_now:.1f}°C | pH: {aktual_ph_now:.2f} (AI 5m: {pred_ph_t30:.2f}) | DO: {aktual_do_now:.2f} (AI 5m: {pred_do_t30:.2f})")
        
        # g. Susun JSON Payload & Tembak ke Website
        payload = {
            "sensor_id": "esp32-kolam-01",
            "timestamp_aktual": timestamp_now_str,
            "timestamp_prediksi": timestamp_future_str,
            "aktual": {"suhu": float(aktual_suhu_now), "ph": float(aktual_ph_now), "do": float(aktual_do_now)},
            "prediksi_5m": {"suhu": float(pred_suhu_t30), "ph": float(pred_ph_t30), "do": float(pred_do_t30)},
            "kontrol_relay": {"aerator_do": relay_do_state, "solenoid_ph": relay_ph_state}
        }
        send_to_website(payload)
        
        # Bersihkan buffer memori tua (>15 menit) agar RAM Jetson tidak penuh
        limit_time = datetime.utcnow() - timedelta(minutes=15)
        history_predictions_do = {k: v for k, v in history_predictions_do.items() if datetime.strptime(k, "%Y-%m-%dT%H:%M:%SZ") > limit_time}
        history_predictions_ph = {k: v for k, v in history_predictions_ph.items() if datetime.strptime(k, "%Y-%m-%dT%H:%M:%SZ") > limit_time}

    except Exception as e:
        print(f"[ERROR LOOP] Kendala runtime: {e}")
        
    time.sleep(10) # Loop berjalan sinkron per 10 detik mengikuti input data ESP32