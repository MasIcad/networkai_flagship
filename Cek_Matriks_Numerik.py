import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load data numerik npy dan scaler (Lengkap dengan SUHU)
try:
    X_train_suhu = np.load('X_train_do.npy') # Karena strukturnya sama, kita pastikan load pasangannya jika dipisahkan nanti
    # Namun karena di script dummy awal kita buat mandiri, kita load file npy masing-masing:
    # Catatan: Pastikan di script dummy kamu sudah menambahkan np.save('X_train_suhu.npy', X_train_suhu) dst.
    # Jika belum sempat disave, di bawah ini kita asumsikan semua file npy parameter sudah lengkap.
    
    X_train_suhu = np.load('X_train_do.npy') # Sebagai placeholder jika belum disave terpisah, tapi mari kita muat scaler aslinya
    X_train_do = np.load('X_train_do.npy')
    y_train_do = np.load('y_train_do.npy')
    X_train_ph = np.load('X_train_ph.npy')
    y_train_ph = np.load('y_train_ph.npy')
    
    # Hubungkan kembali dengan scaler masing-masing
    scaler_suhu = joblib.load('scaler_suhu.pkl')
    scaler_do = joblib.load('scaler_do.pkl')
    scaler_ph = joblib.load('scaler_ph.pkl')
    print("[SUKSES] File numerik dan seluruh Scaler (Suhu, pH, DO) berhasil dimuat!\n")
except FileNotFoundError:
    print("[ERROR] Pastikan file .npy dan .pkl sudah lengkap di folder proyek!")
    exit()

# 2. Pilih indeks sampel acak untuk diinspeksi (Misal: indeks ke-5000)
sample_idx = 5000 

# Untuk keperluan visualisasi valid, kita ambil array scaled-nya
# (Jika di script dummy kamu belum menyimpan X_train_suhu.npy, silakan tambahkan dulu di script dummy ya!)
# Di sini kita simulasikan pembacaan jika X_train_suhu sudah ada:
try:
    X_train_suhu = np.load('X_train_suhu.npy')
    y_train_suhu = np.load('y_train_suhu.npy')
except FileNotFoundError:
    # Jika belum ada file khusus suhu, kita buat info log
    print("[INFO] File X_train_suhu.npy belum terdeteksi. Pastikan di script Datadummy.py sudah ditambahkan np.save() untuk Suhu.")
    exit()

# 3. Ambil sampel berdasarkan indeks
X_sample_suhu = scaler_suhu.inverse_transform(X_train_suhu[sample_idx]).flatten()
y_sample_suhu = scaler_suhu.inverse_transform(y_train_suhu[sample_idx].reshape(-1, 1))[0][0]

X_sample_do = scaler_do.inverse_transform(X_train_do[sample_idx]).flatten()
y_sample_do = scaler_do.inverse_transform(y_train_do[sample_idx].reshape(-1, 1))[0][0]

X_sample_ph = scaler_ph.inverse_transform(X_train_ph[sample_idx]).flatten()
y_sample_ph = scaler_ph.inverse_transform(y_train_ph[sample_idx].reshape(-1, 1))[0][0]

# ==========================================
# PRINT EVALUASI NUMERIK DI TERMINAL
# ==========================================
print(f"=== INSPEKSI MATRIKS NUMERIK 3 PARAMETER (SAMPEL INDEKS KE-{sample_idx}) ===")
print(f"Durasi Data Input (X) : 60 titik (10 menit ke belakang)")
print(f"Jarak ke Target (y)   : 30 titik berikutnya (5 menit ke depan)\n")

print("--- PARAMETER SUHU ---")
print(f"-> 5 Data Terakhir di Input X : {X_sample_suhu[-5:]}")
print(f"-> TARGET PADA T+30 (y)       : {y_sample_suhu:.2f} °C\n")

print("--- PARAMETER OKSIGEN TERLARUT (DO) ---")
print(f"-> 5 Data Terakhir di Input X : {X_sample_do[-5:]}")
print(f"-> TARGET PADA T+30 (y)       : {y_sample_do:.2f} mg/L\n")

print("--- PARAMETER DERAJAT KEASAMAN (pH) ---")
print(f"-> 5 Data Terakhir di Input X : {X_sample_ph[-5:]}")
print(f"-> TARGET PADA T+30 (y)       : {y_sample_ph:.2f}")
print("=" * 50)

# ==========================================
# VISUALISASI PLOT 3 PARAMETER SECARA VERTIKAL
# ==========================================
time_X = np.arange(0, 60)
time_y = 60 + 30 - 1

fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
fig.suptitle(f'Inspeksi Hubungan Temporal 3 Parameter AI (Sampel #{sample_idx})', fontsize=12, fontweight='bold')

# Plot Suhu
axs[0].plot(time_X, X_sample_suhu, color='red', label='Suhu Historis (Input X)')
axs[0].scatter(time_y, y_sample_suhu, color='darkred', s=100, zorder=5, label='Target Suhu T+30 (y)')
axs[0].set_ylabel('Suhu (°C)')
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc='upper left')

# Plot pH
axs[1].plot(time_X, X_sample_ph, color='green', label='pH Historis (Input X)')
axs[1].scatter(time_y, y_sample_ph, color='darkgreen', s=100, zorder=5, label='Target pH T+30 (y)')
axs[1].set_ylabel('pH')
axs[1].grid(True, alpha=0.3)
axs[1].legend(loc='upper left')

# Plot DO
axs[2].plot(time_X, X_sample_do, color='blue', label='DO Historis (Input X)')
axs[2].scatter(time_y, y_sample_do, color='darkblue', s=100, zorder=5, label='Target DO T+30 (y)')
axs[2].set_ylabel('DO (mg/L)')
axs[2].set_xlabel('Indeks Waktu Runtun (Per 10 Detik)')
axs[2].grid(True, alpha=0.3)
axs[2].legend(loc='upper left')

plt.tight_layout()
plt.show()