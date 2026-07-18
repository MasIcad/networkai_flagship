import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

lookback = 20

# 1. Load Data Fisik .npy
print("⏳ Memuat data biner dari penyimpanan lokal...")
X_train_suhu = np.load('X_train_suhu.npy')
y_train_suhu = np.load('y_train_suhu.npy')
X_train_ph = np.load('X_train_ph.npy')
y_train_ph = np.load('y_train_ph.npy')
X_train_do = np.load('X_train_do.npy')
y_train_do = np.load('y_train_do.npy')

# 2. Definisikan Arsitektur CNN-LSTM Asli
def build_cnn_lstm_architecture():
    return Sequential([
        Conv1D(32, 3, padding='same', activation='relu', input_shape=(lookback, 1)),
        BatchNormalization(),
        MaxPooling1D(2),
        Bidirectional(LSTM(64, return_sequences=True, activation='relu')),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]

# 3. Training & Eksportasi Model Fisik
targets = {
    'suhu': (X_train_suhu, y_train_suhu),
    'ph': (X_train_ph, y_train_ph),
    'do': (X_train_do, y_train_do)
}

for name, (X_train, y_train) in targets.items():
    print(f"\n--- Memulai Training CNN-LSTM Parameter: {name.upper()} ---")
    model = build_cnn_lstm_architecture()
    model.compile(optimizer=Adam(0.001), loss='mae')
    
    # Catatan Jetson Nano: Set 3-5 epoch saja jika ingin training langsung di dev-board, 
    # atau biarkan 100 jika Anda me-running file ini di laptop/Colab terpisah.
    model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=1, callbacks=callbacks)
    
    model.save(f'best_model_{name}.h5')
    print(f"✅ Model 'best_model_{name}.h5' sukses diamankan.")