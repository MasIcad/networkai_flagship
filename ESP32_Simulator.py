import time
import csv
import os
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'live_telemetry.csv')

print("🟢 Simulator ESP32 Terkalibrasi Aktif!")
np.random.seed(42)

# Menggenerasikan 60 data awal yang berpusat pada kondisi dasar model (Jam 00:00 tengah malam)
# Agar buffer awal sesuai dengan pola yang dipelajari CNN-LSTM
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
        
        # SIKLUS NORMAL (Dibuat bergerak lebih cepat dengan membagi step dengan angka lebih kecil, misal 2)
        if step <= 80:
            suhu = 27.5 + 2.5 * np.sin(step / 2) + np.random.normal(0, 0.05)
            ph = 7.4 + 0.4 * np.sin(step / 2.5) + np.random.normal(0, 0.01)
            do = 7.8 + 1.5 * np.sin(step / 2.5) + np.random.normal(0, 0.05)
            status = "Normal"
        # INJEKSI ANOMALI LANGSUNG (Setelah langkah ke-80)
        else:
            suhu = 28.5 + np.random.normal(0, 0.05)
            ph = 8.8 + np.random.normal(0, 0.02)  # Lonjakan pH ke basa kuat
            do = 4.2 + np.random.normal(0, 0.05)  # Drop DO kritis ke bawah
            status = "🚨 ANOMALI"

        with open(CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ts, suhu, ph, do])
            
        print(f"[{status}] Data Ke-{step} -> Suhu: {suhu:.2f}°C, pH: {ph:.2f}, DO: {do:.2f} mg/L")
        time.sleep(1.5) # Dipercepat sedikit biar lebih responsif

except KeyboardInterrupt:
    print("\n🛑 Simulator dihentikan.")