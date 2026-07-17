import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

print("⏳ Memuat data uji & model untuk kompilasi dashboard grafik...")
# Load data test fisik
X_test_suhu = np.load('X_test_suhu.npy')
y_test_suhu = np.load('y_test_suhu.npy')
X_test_ph = np.load('X_test_ph.npy')
y_test_ph = np.load('y_test_ph.npy')
X_test_do = np.load('X_test_do.npy')
y_test_do = np.load('y_test_do.npy')

# Load Model dan Scaler
model_suhu = tf.keras.models.load_model('best_model_suhu.h5')
model_ph = tf.keras.models.load_model('best_model_ph.h5')
model_do = tf.keras.models.load_model('best_model_do.h5')

scaler_suhu = joblib.load('scaler_suhu.pkl')
scaler_ph = joblib.load('scaler_ph.pkl')
scaler_do = joblib.load('scaler_do.pkl')

# Lakukan prediksi skala penuh untuk plot data uji
print("🔮 Memproses kalkulasi prediksi visual grafik...")
pred_suhu = scaler_suhu.inverse_transform(model_suhu.predict(X_test_suhu, batch_size=256, verbose=0))
pred_ph   = scaler_ph.inverse_transform(model_ph.predict(X_test_ph, batch_size=256, verbose=0))
pred_do   = scaler_do.inverse_transform(model_do.predict(X_test_do, batch_size=256, verbose=0))

act_suhu  = scaler_suhu.inverse_transform(y_test_suhu.reshape(-1, 1))
act_ph    = scaler_ph.inverse_transform(y_test_ph.reshape(-1, 1))
act_do    = scaler_do.inverse_transform(y_test_do.reshape(-1, 1))

# Plotting Grafik secara Terpisah & Simpan Fisik Gambar
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Dashboard Multi-Parameter MISO - Analisis Respon Jaringan CNN-LSTM', fontsize=14, fontweight='bold')

# Subplot Suhu
axs[0].plot(act_suhu, color='black', alpha=0.3, label='Aktual')
axs[0].plot(pred_suhu, color='red', linestyle='--', label='Prediksi AI (T+30)')
axs[0].set_ylabel('Suhu (°C)', fontweight='bold')
axs[0].grid(True, alpha=0.2)
axs[0].legend(loc='upper right')

# Subplot pH
axs[1].plot(act_ph, color='black', alpha=0.3, label='Aktual')
axs[1].plot(pred_ph, color='green', linestyle='--', label='Prediksi AI (T+30)')
axs[1].set_ylabel('pH', fontweight='bold')
axs[1].grid(True, alpha=0.2)
axs[1].legend(loc='upper right')

# Subplot DO
axs[2].plot(act_do, color='black', alpha=0.3, label='Aktual')
axs[2].plot(pred_do, color='blue', linestyle='--', label='Prediksi AI (T+30)')
axs[2].set_ylabel('DO (mg/L)', fontweight='bold')
axs[2].set_xlabel('Indeks Titik Data Uji (Resolusi 10 Detik)', fontweight='bold')
axs[2].grid(True, alpha=0.2)
axs[2].legend(loc='upper right')

plt.tight_layout()
plt.savefig('dashboard_analisis.png', dpi=300)
print("✅ Dashboard visualisasi sukses diekspor ke file gambar -> 'dashboard_analisis.png'")