import os
import pandas as pd
import numpy as np
import joblib
import gc
import tensorflow as tf
import time

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

LIMIT_BAWAH_DO = 5.0   
LIMIT_ATAS_PH  = 8.5    
LIMIT_BAWAH_PH = 6.5    
LIMIT_ATAS_SUHU = 32.0

print("🚀 Engine CNN-LSTM Siap. Menunggu data dari live_telemetry.csv...")

def run_ai_inference():
    tf.keras.backend.clear_session()
    
    if not os.path.exists(CSV_PATH):
        print("⚠️ File 'live_telemetry.csv' belum ditemukan. Menunggu data...")
        return
    
    try:
        df = pd.read_csv(CSV_PATH)
        if len(df) < lookback:
            print(f"⏳ Data belum cukup. Butuh {lookback} baris, saat ini baru ada {len(df)} baris.")
            return
            
        recent_data = df.tail(lookback)
        
        val_suhu = recent_data['SUHU'].values
        val_ph = recent_data['PH'].values
        val_do = recent_data['DO'].values
        
        current_suhu = val_suhu[-1]
        current_ph = val_ph[-1]
        current_do = val_do[-1]
        
        in_suhu = scaler_suhu.transform(val_suhu.reshape(-1, 1)).reshape(1, lookback, 1)
        in_ph = scaler_ph.transform(val_ph.reshape(-1, 1)).reshape(1, lookback, 1)
        in_do = scaler_do.transform(val_do.reshape(-1, 1)).reshape(1, lookback, 1)
        
        pred_suhu = scaler_suhu.inverse_transform(model_suhu(in_suhu, training=False).numpy())[0][0]
        pred_ph = scaler_ph.inverse_transform(model_ph(in_ph, training=False).numpy())[0][0]
        pred_do = scaler_do.inverse_transform(model_do(in_do, training=False).numpy())[0][0]
        
        offset_suhu = current_suhu - pred_suhu
        offset_ph = current_ph - pred_ph
        offset_do = current_do - pred_do
        
        residu_suhu = abs(offset_suhu)
        residu_ph = abs(offset_ph)
        residu_do = abs(offset_do)
        
        acc_suhu = max(0.0, 100.0 - (residu_suhu / current_suhu * 100)) if current_suhu != 0 else 0
        acc_ph = max(0.0, 100.0 - (residu_ph / current_ph * 100)) if current_ph != 0 else 0
        acc_do = max(0.0, 100.0 - (residu_do / current_do * 100)) if current_do != 0 else 0
        
        status_suhu = "ANOMALI" if (residu_suhu > THRESHOLD_RESIDU_SUHU and current_suhu > LIMIT_ATAS_SUHU) else "OK"
        status_ph = "ANOMALI" if (residu_ph > THRESHOLD_RESIDU_PH and (current_ph > LIMIT_ATAS_PH or current_ph < LIMIT_BAWAH_PH)) else "OK"
        status_do = "ANOMALI" if (residu_do > THRESHOLD_RESIDU_DO and current_do < LIMIT_BAWAH_DO) else "OK"
        
        print("\n=======================================================")
        print(f"[SUHU] Akt: {current_suhu:.2f}°C | Pred: {pred_suhu:.2f}°C | Off: {offset_suhu:+.2f} | Acc: {acc_suhu:.1f}% | Status: {status_suhu}")
        print(f"[ pH ] Akt: {current_ph:.2f}   | Pred: {pred_ph:.2f}   | Off: {offset_ph:+.2f} | Acc: {acc_ph:.1f}% | Status: {status_ph}")
        print(f"[ DO ] Akt: {current_do:.2f} mg/L| Pred: {pred_do:.2f} mg/L| Off: {offset_do:+.2f} | Acc: {acc_do:.1f}% | Status: {status_do}")
        print("=======================================================")
        
    except Exception as e:
        print(f"⚠️ Menunggu sinkronisasi penulisan file oleh backend: {e}")
        
    finally:
        gc.collect()

if __name__ == "__main__":
    while True:
        run_ai_inference()
        time.sleep(60)