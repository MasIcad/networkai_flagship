import numpy as np
import joblib
import collections
import tensorflow as tf

# 1. Konfigurasi Ambang Batas Residu Kejut Sesuai Logika Riset Anda
THRESHOLD_RESIDU_DO = 0.8
THRESHOLD_RESIDU_PH = 0.3
CRITICAL_BAWAH_DO = 5.0
CRITICAL_ATAS_PH = 8.5

# 2. Load Model dan Scaler Hasil Ekstraksi Pembuat Data
print("⏳ Mengunduh arsitektur model pintar ke memori RAM Jetson...")
model_suhu = tf.keras.models.load_model('best_model_suhu.h5')
model_ph = tf.keras.models.load_model('best_model_ph.h5')
model_do = tf.keras.models.load_model('best_model_do.h5')

scaler_suhu = joblib.load('scaler_suhu.pkl')
scaler_ph = joblib.load('scaler_ph.pkl')
scaler_do = joblib.load('scaler_do.pkl')
print("🟢 Sistem Utama Mengudara! Seluruh Model AI Siap Mengeksekusi.")

# 3. Inisialisasi Sliding Window Buffer (Lookback=60 data / 10 menit ke belakang)
lookback = 60
buffer_suhu = collections.deque(maxlen=lookback)
buffer_ph = collections.deque(maxlen=lookback)
buffer_do = collections.deque(maxlen=lookback)

# Isi awal antrean dengan nilai acuan aman agar prediksi langsung jalan tanpa nunggu buffer kosong
for _ in range(lookback):
    buffer_suhu.append(27.5)
    buffer_ph.append(7.4)
    buffer_do.append(6.5)

print("\n" + "="*55)
print("📡 INTEGRASI SIMULASI TELEMETRI ESP32 (TERMINAL INPUT)")
print("Masukkan data terkini secara manual dengan pemisah koma.")
print("="*55)

try:
    while True:
        print("\n" + "-"*55)
        try:
            raw_input = input("Format -> [SUHU, pH, DO] (Contoh: 28.2, 7.5, 4.2): ")
            val_suhu, val_ph, val_do = map(float, raw_input.split(','))
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Format salah atau interupsi terdeteksi. Silakan coba lagi.")
            continue

        # Sisipkan data real-time saat ini ke dalam sliding window
        buffer_suhu.append(val_suhu)
        buffer_ph.append(val_ph)
        buffer_do.append(val_do)

        # 4. Normalisasi Buffer Data Masuk untuk Kebutuhan Bentuk Input Model
        in_suhu = scaler_suhu.transform(np.array(buffer_suhu).reshape(-1, 1)).reshape(1, lookback, 1)
        in_ph = scaler_ph.transform(np.array(buffer_ph).reshape(-1, 1)).reshape(1, lookback, 1)
        in_do = scaler_do.transform(np.array(buffer_do).reshape(-1, 1)).reshape(1, lookback, 1)

        # 5. Eksekusi Inferensi AI (Prediksi 30 Langkah / 5 Menit ke depan)
        pred_suhu_t30 = scaler_suhu.inverse_transform(model_suhu.predict(in_suhu, verbose=0))[0][0]
        pred_ph_t30 = scaler_ph.inverse_transform(model_ph.predict(in_ph, verbose=0))[0][0]
        pred_do_t30 = scaler_do.inverse_transform(model_do.predict(in_do, verbose=0))[0][0]

        # Kalkulasi Margin Deviasi Residu Kejut
        residu_do = abs(val_do - pred_do_t30)
        residu_ph = abs(val_ph - pred_ph_t30)

        # 6. Evaluasi Indikator Bahaya/Anomali Berdasarkan Prediksi & Residu
        anomali_do = (residu_do > THRESHOLD_RESIDU_DO) or (pred_do_t30 < CRITICAL_BAWAH_DO)
        anomali_ph = (residu_ph > THRESHOLD_RESIDU_PH) or (pred_ph_t30 > CRITICAL_ATAS_PH)

        # 7. Cetak Konsol Monitoring Output Secara Bersih
        print(f"\n📊 [REAL-TIME (T)]  Suhu: {val_suhu:.2f}°C | pH: {val_ph:.2f} | DO: {val_do:.2f} mg/L")
        print(f"🔮 [PREDIKSI (T+30)] Suhu: {pred_suhu_t30:.2f}°C | pH: {pred_ph_t30:.2f} | DO: {pred_do_t30:.2f} mg/L")
        print(f"📉 [MARGIN RESIDU]  Residu pH: {residu_ph:.3f} | Residu DO: {residu_do:.3f}")
        print(f"⚠️ [INDIKASI SISTEM] Oksigen (DO): {'❌ ANOMALI DETECTED' if anomali_do else '✅ AMAN'} | Kadar Asam (pH): {'❌ ANOMALI DETECTED' if anomali_ph else '✅ AMAN'}")

except KeyboardInterrupt:
    print("\n👋 Sistem dihentikan dengan aman. Keluar dari lingkungan Jetson.")