# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# ── STEP 1: FIX THE PATH FIRST ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

def ensure_import_path():
    sys.path.insert(0, str(ROOT_DIR))
ensure_import_path() 

import numpy as np

# ── STEP 2: NOW IMPORT YOUR PROJECT MODULES ──────────────────────────────────
from input_processing.input_setup import X_train, X_test, y_train, y_test
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential

# ── Model 2: LSTM (2 layers, 64 units each) ──────────────────────────────────
# Reshape data for LSTM
# ── Model 2: LSTM (2 layers, 64 units each) ──────────────────────────────────

# CRUCIAL FIX: Force all data to float32 so TensorFlow doesn't complain about 'object' types
X_train_numeric = X_train.astype('float32')
X_test_numeric = X_test.astype('float32')

# Reshape data for LSTM (using our clean numeric copies)
X_train_lstm = X_train_numeric.values.reshape((X_train_numeric.shape[0], X_train_numeric.shape[1], 1))
X_test_lstm = X_test_numeric.values.reshape((X_test_numeric.shape[0], X_test_numeric.shape[1], 1))

# Build model the modern Keras way
lstm_model = Sequential([
    # 1. Define the input shape explicitly first
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    
    # 2. Your first LSTM layer (no input_shape argument needed anymore!)
    LSTM(64, return_sequences=True),
    
    # 3. Your second LSTM layer
    LSTM(64),
    
    # 4. Final output layer
    Dense(1)
])

# Compile model
lstm_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train model
history = lstm_model.fit(
    X_train_lstm,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_lstm, y_test),
    verbose=0
)

# ── Evaluate using evaluate_model style ──────────────────────────────────────

# Predict
y_pred = lstm_model.predict(X_test_lstm)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n🤖 LSTM model")
print(f"   MAE  (Avg Error in units): {mae:.2f}")
print(f"   RMSE (Root Mean Sq Error): {rmse:.2f}")
print(f"   R²   (Accuracy Score):     {r2:.4f}  ({r2*100:.2f}% accuracy)")


# Store results
lstm_results = {
    'Model': 'LSTM',
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2,
    'Predictions': y_pred
}