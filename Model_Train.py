import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

# Panggil kelasnya melalui namespace yang lebih bersih
Sequential = keras.Sequential
Conv1D = layers.Conv1D
BatchNormalization = layers.BatchNormalization
MaxPooling1D = layers.MaxPooling1D
LSTM = layers.LSTM
Bidirectional = layers.Bidirectional
Dropout = layers.Dropout
Dense = layers.Dense
EarlyStopping = callbacks.EarlyStopping
ModelCheckpoint = callbacks.ModelCheckpoint
Adam = optimizers.Adam

# ==========================================
# 1. LOAD DATA NUMERIK YANG SUDAH SIAP
# ==========================================
print("1. Memuat data numerik .npy...")
try:
    X_train_suhu = np.load('X_train_suhu.npy')
    y_train_suhu = np.load('y_train_suhu.npy')
    X_test_suhu  = np.load('X_test_suhu.npy')
    y_test_suhu  = np.load('y_test_suhu.npy')

    X_train_ph   = np.load('X_train_ph.npy')
    y_train_ph   = np.load('y_train_ph.npy')
    X_test_ph    = np.load('X_test_ph.npy')
    y_test_ph    = np.load('y_test_ph.npy')

    X_train_do   = np.load('X_train_do.npy')
    y_train_do   = np.load('y_train_do.npy')
    X_test_do    = np.load('X_test_do.npy')
    y_test_do    = np.load('y_test_do.npy')
    
    lookback = X_train_do.shape[1] # Akan bernilai 60
    print(f"[SUKSES] Data berhasil dimuat. Panjang sekuens input (lookback): {lookback} titik.")
except FileNotFoundError as e:
    print(f"[ERROR] File data training tidak ditemukan: {e}")
    exit()

# ==========================================
# 2. DEFINISI ARSITEKTUR & KOMPILASI TERISOLASI
# ==========================================
# Fungsi ini menjamin model dan optimizernya dilahirkan baru setiap kali dipanggil
def create_and_compile_model():
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(lookback, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Bidirectional(LSTM(64, return_sequences=True, activation='relu')),
        LSTM(32, activation='relu'),
        
        Dropout(0.2),
        Dense(1)
    ])
    
    # Optimizer benar-benar lokal dan baru di dalam fungsi ini
    local_opt = Adam(learning_rate=0.001)
    model.compile(optimizer=local_opt, loss='mae')
    
    return model

# Panggil fungsi terpisah untuk melahirkan tiap model beserta lokakarya optimizernya
print("[INIT] Membuat dan mengompilasi Model Suhu...")
model_suhu = create_and_compile_model()

print("[INIT] Membuat dan mengompilasi Model pH...")
model_ph   = create_and_compile_model()

print("[INIT] Membuat dan mengompilasi Model DO...")
model_do   = create_and_compile_model()


# ==========================================
# 3. KONFIGURASI CALLBACKS TRAINING
# ==========================================
callbacks_suhu = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint('best_model_suhu.h5', save_best_only=True, monitor='val_loss')
]

callbacks_ph = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint('best_model_ph.h5', save_best_only=True, monitor='val_loss')
]

callbacks_do = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint('best_model_do.h5', save_best_only=True, monitor='val_loss')
]

# ==========================================
# 4. EKSEKUSI TRAINING MODEL (Batch size & Epochs sesuai Colab)
# ==========================================
EPOCHS = 100
BATCH_SIZE = 64

print("\n--- Training Model SUHU CNN-LSTM ---")
history_suhu = model_suhu.fit(X_train_suhu, y_train_suhu, epochs=EPOCHS, batch_size=BATCH_SIZE,
                              validation_data=(X_test_suhu, y_test_suhu), callbacks=callbacks_suhu, verbose=1)

print("\n--- Training Model PH CNN-LSTM ---")
history_ph = model_ph.fit(X_train_ph, y_train_ph, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(X_test_ph, y_test_ph), callbacks=callbacks_ph, verbose=1)

print("\n--- Training Model DO CNN-LSTM ---")
history_do = model_do.fit(X_train_do, y_train_do, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(X_test_do, y_test_do), callbacks=callbacks_do, verbose=1)

print("\n[SUKSES] Semua model berhasil dilatih dan disimpan sebagai file '.h5'!")