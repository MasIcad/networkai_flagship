import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# GENERATE DATA MAKRO 4 HARI UNTUK DILIHAT UTUH
# ==========================================
n_days = 4
data_per_day = 8640  # 24 jam * 60 menit * 6 (per 10 detik)
n_data = data_per_day * n_days
t = np.arange(n_data)

# Fungsi generator mulus yang sama dengan Datadummy.py
suhu_base = 27.5 + 2.5 * np.sin(2 * np.pi * (t - 3600) / 8640) 
suhu_raw = suhu_base + np.random.normal(0, 0.01, len(t))

limbah_akumulasi = (t / n_data) * 0.3 
ph_base = 7.4 + 0.5 * np.sin(2 * np.pi * (t - 3600) / 8640) + limbah_akumulasi
ph_raw = ph_base + np.random.normal(0, 0.005, len(t))

do_diurnal = 6.5 + 1.8 * np.cos(2 * np.pi * (t - 1800) / 8640)
efek_suhu = -0.15 * (suhu_base - 27.5) 
do_raw = do_diurnal + efek_suhu + np.random.normal(0, 0.01, len(t))

# Anomali Hari Ke-4 (Menit akhir kolam)
ph_raw[29500:31500] += 0.8
do_raw[29500:31500] -= 3.2

# Smoothing Moving Average
suhu = pd.Series(suhu_raw).rolling(window=30, min_periods=1).mean().values
ph = pd.Series(ph_raw).rolling(window=30, min_periods=1).mean().values
do = pd.Series(do_raw).rolling(window=30, min_periods=1).mean().values
do = np.clip(do, 0.2, 10.0)

# ==========================================
# VISUALISASI PLOT MAKRO 4 HARI
# ==========================================
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Tren Makro Kualitas Air Kolam Nila (4 Hari Penuh - Resolusi 10 Detik)', fontsize=14, fontweight='bold')

# Subplot Suhu
axs[0].plot(suhu, color='red', label='Suhu Air')
axs[0].axvline(5000, color='black', linestyle='--', linewidth=1.5, label='Posisi Jendela Sampel #5000')
axs[0].set_ylabel('Suhu (°C)')
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc='upper right')

# Subplot pH
axs[1].plot(ph, color='green', label='pH Air')
axs[1].axvline(5000, color='black', linestyle='--', linewidth=1.5)
axs[1].set_ylabel('pH')
axs[1].grid(True, alpha=0.3)
axs[1].legend(loc='upper right')

# Subplot DO
axs[2].plot(do, color='blue', label='Dissolved Oxygen (DO)')
axs[2].axvline(5000, color='black', linestyle='--', linewidth=1.5)
axs[2].set_ylabel('DO (mg/L)')
axs[2].set_xlabel('Indeks Data (Setiap 10 Detik)')
axs[2].grid(True, alpha=0.3)
axs[2].legend(loc='upper right')

# Tandai area bom amonia kotoran ikan di hari ke-4
for ax in axs:
    ax.axvspan(29500, 31500, color='orange', alpha=0.25, label='Bom Amonia (Anomali)')

plt.tight_layout()
plt.show()