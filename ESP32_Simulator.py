import time
import csv
import os
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'live_telemetry.csv')

print("🟢 Simulator ESP32 Terkalibrasi Aktif dengan Logika Pemulihan!")
np.random.seed(42)

print("⏳ Mengisi buffer awal dengan tren alami...")
with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['TIMESTAMP', 'SUHU', 'PH', 'DO'])
    
    for t in range(20):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        suhu = 27.5 + 2.5 * np.sin(2 * np.pi * (t - 480) / 1440) + np.random.normal(0, 0.01)
        ph = 7.4 + 0.4 * np.sin(2 * np.pi * (t - 570) / 1440) + np.random.normal(0, 0.005)
        do = 7.8 + 1.5 * np.sin(2 * np.pi * (t - 570) / 1440) + np.random.normal(0, 0.01)
        writer.writerow([ts, suhu, ph, do])

step = 20
try:
    while True:
        step += 1
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # ====================================================================
        # FASE 1: SIKLUS NORMAL (Step 21 hingga 60)
        # ====================================================================
        if step <= 60:
            suhu = 27.5 + 2.5 * np.sin(step / 2) + np.random.normal(0, 0.05)
            ph = 7.4 + 0.4 * np.sin(step / 2.5) + np.random.normal(0, 0.01)
            do = 7.8 + 1.5 * np.sin(step / 2.5) + np.random.normal(0, 0.05)
            status = "Normal"
            
        # ====================================================================
        # FASE 2: INJEKSI ANOMALI KEJUT (Step 61 hingga 90)
        # ====================================================================
        elif 60 < step <= 90:
            suhu = 28.5 + np.random.normal(0, 0.05)
            ph = 8.8 + np.random.normal(0, 0.02)  # Lonjakan pH ke basa kuat
            do = 4.2 + np.random.normal(0, 0.05)  # Drop DO kritis ke bawah
            status = "🚨 ANOMALI"
            
        # ====================================================================
        # FASE 3: OTOMATIS PEMULIHAN / RECOVERY (Step 91 ke atas)
        # ====================================================================
        else:
            # Menggunakan fungsi eksponensial melandai agar nilai kembali ke target aman secara halus
            faktor_pulih = min(1.0, (step - 90) / 15)  # Akan mencapai 100% pulih dalam 15 step
            
            suhu = 28.5 - (1.0 * faktor_pulih) + np.random.normal(0, 0.05)
            ph = 8.8 - (1.4 * faktor_pulih) + np.random.normal(0, 0.01)  # ph kembali mendekati 7.4
            do = 4.2 + (3.2 * faktor_pulih) + np.random.normal(0, 0.05)   # DO kembali naik ke ~7.4
            
            if faktor_pulih >= 1.0:
                status = "✅ RECOVERY SELESAI"
            else:
                status = "🔄 PROSES PEMULIHAN"

        # Tulis data ke CSV
        with open(CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ts, suhu, ph, do])
            
        print(f"[{status}] Data Ke-{step} -> Suhu: {suhu:.2f}°C, pH: {ph:.2f}, DO: {do:.2f} mg/L")
        time.sleep(60) # Sedikit dipercepat agar tidak bosan menunggu transisinya

except KeyboardInterrupt:
    print("\n🛑 Simulator dihentikan.")