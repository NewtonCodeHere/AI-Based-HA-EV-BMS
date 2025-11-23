# train_lstm_soc.py


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import warnings
from glob import glob

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)


# ---------------------------
# User: provide the 4 local .mat file paths here (absolute paths)
# Example Windows: r"C:\Users\you\Downloads\Cycle_1.mat"
# Example Linux: "/home/you/datasets/Cycle_1.mat"
MAT_FILES = [
    r"C:\Users\visha\OneDrive\Desktop\03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat",
    r"C:\Users\visha\OneDrive\Desktop\03-19-17_03.25 25degC_Cycle_2_Pan18650PF.mat",
    r"C:\Users\visha\OneDrive\Desktop\03-19-17_09.07 25degC_Cycle_3_Pan18650PF.mat",
    r"C:\Users\visha\OneDrive\Desktop\03-19-17_14.31 25degC_Cycle_4_Pan18650PF.mat"
]

# Verify files exist
missing = [p for p in MAT_FILES if not os.path.exists(p)]
if missing:
    print("ERROR: The following MAT files were not found:")
    for m in missing:
        print("  -", m)
    print("\nPlease update MAT_FILES with the correct absolute paths and re-run.")
    sys.exit(1)


# ---------------------------
# Strategy / GPU configuration for local GPU (CUDA)
# ---------------------------
def get_strategy_local():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for each GPU
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            # Use MirroredStrategy (works for single GPU also)
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas (GPU).")
            print("Physical GPUs:", gpus)
            return strategy
        except Exception as e:
            print("Could not initialize MirroredStrategy:", e)
    # fallback CPU
    print("No GPU available - using default strategy (CPU).")
    return tf.distribute.get_strategy()

STRATEGY = get_strategy_local()


# ---------------------------
# Data loader (robust for the Panasonic .mat structure used in the paper)
# ---------------------------
def load_and_combine_mat_files(file_paths, battery_capacity=2.9, downsample_max=None):
    """
    Load multiple .mat files in the Panasonic meas struct format and combine them.
    Returns a pandas DataFrame with columns: Voltage, Current, Temperature, SoC
    """
    all_dfs = []
    print("Loading .mat files ...")
    for path in file_paths:
        try:
            print(f" - {os.path.basename(path)}")
            mat = scipy.io.loadmat(path)
            # typical structure: mat['meas'][0,0] with dtype.names fields
            if 'meas' not in mat:
                print("   WARNING: 'meas' not present in file; skipping.")
                continue
            meas = mat['meas'][0, 0]
            fields = meas.dtype.names

            # voltage
            if 'Voltage' in fields:
                voltage = np.ravel(meas['Voltage'])
            elif 'voltage' in fields:
                voltage = np.ravel(meas['voltage'])
            else:
                print("   Voltage field not found; skipping file.")
                continue

            # current
            if 'Current' in fields:
                current = np.ravel(meas['Current'])
            elif 'current' in fields:
                current = np.ravel(meas['current'])
            else:
                print("   Current field not found; skipping file.")
                continue

            # temperature
            if 'Battery_Temp' in fields:
                temp = np.ravel(meas['Battery_Temp'])
            elif 'Temperature' in fields:
                temp = np.ravel(meas['Temperature'])
            elif 'Temp' in fields:
                temp = np.ravel(meas['Temp'])
            else:
                # fallback: assume 25C if not provided
                temp = np.full_like(voltage, 25.0)

            # SoC computation: prefer Ah field if present
            if 'Ah' in fields:
                ah = np.ravel(meas['Ah'])
                # Convert Ah -> SoC (approx). This assumes Ah goes from 0 (full) to -capacity (empty).
                # This calculation may need adjustment by inspecting the exact sign convention in files.
                ah_min = ah.min()
                ah_max = ah.max()
                if abs(ah_max - ah_min) > 1e-6:
                    soc = 100.0 * (ah - ah_min) / (ah_max - ah_min)
                else:
                    soc = 100.0 * (1.0 - np.abs(ah) / battery_capacity)
            elif 'SoC' in fields:
                soc_raw = np.ravel(meas['SoC'])
                # If in [0,1] convert to %
                if soc_raw.max() <= 1.1:
                    soc = soc_raw * 100.0
                else:
                    soc = soc_raw
            else:
                # last resort: crude voltage-based mapping (adjust ranges if needed)
                v_min, v_max = 2.5, 4.2
                soc = 100.0 * (voltage - v_min) / (v_max - v_min)
                soc = np.clip(soc, 0.0, 100.0)

            # Build DataFrame
            df = pd.DataFrame({
                'Voltage': voltage.astype(np.float32),
                'Current': current.astype(np.float32),
                'Temperature': temp.astype(np.float32),
                'SoC': soc.astype(np.float32)
            })

            # basic cleaning
            df = df.dropna().reset_index(drop=True)
            # basic sanity filters
            df = df[(df['Voltage'] > 2.0) & (df['Voltage'] < 5.0)]
            df = df[(df['Current'] > -50) & (df['Current'] < 50)]
            df = df.reset_index(drop=True)

            if downsample_max and len(df) > downsample_max:
                step = max(1, len(df) // downsample_max)
                df = df.iloc[::step].reset_index(drop=True)

            print(f"   Loaded {len(df)} samples.")
            all_dfs.append(df)
        except Exception as e:
            print(f"   Error loading {path}: {e}")
            continue

    if not all_dfs:
        raise RuntimeError("No valid data loaded. Check file contents and structure.")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataset shape: {combined.shape}")
    return combined


# ---------------------------
# LSTM predictor class (build in strategy scope)
# ---------------------------
class BatterySoCPredictor:
    def __init__(self, look_back=60, hidden_dim=5, learning_rate=1e-3, strategy=None):
        self.look_back = look_back
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.strategy = strategy or tf.distribute.get_strategy()
        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.history = None

    def preprocess(self, df):
        # Optional smoothing
        df = df.copy()
        df['Voltage'] = df['Voltage'].rolling(window=5, center=True, min_periods=1).mean()
        df['Current'] = df['Current'].rolling(window=5, center=True, min_periods=1).mean()
        df['Temperature'] = df['Temperature'].rolling(window=3, center=True, min_periods=1).mean()
        df['SoC'] = df['SoC'].rolling(window=5, center=True, min_periods=1).mean()
        df = df.dropna().reset_index(drop=True)

        X_all = df[['Voltage', 'Current', 'Temperature']].values
        y_all = df['SoC'].values.reshape(-1, 1)

        # train/val split (80/20)
        split = int(len(X_all) * 0.8)
        X_train_raw, X_test_raw = X_all[:split], X_all[split:]
        y_train_raw, y_test_raw = y_all[:split], y_all[split:]

        # fit scalers on training raw portion only
        self.scaler_X.fit(X_train_raw)
        self.scaler_y.fit(y_train_raw)

        X_train = self.scaler_X.transform(X_train_raw)
        X_test = self.scaler_X.transform(X_test_raw)
        y_train = self.scaler_y.transform(y_train_raw)
        y_test = self.scaler_y.transform(y_test_raw)

        X_tr_seq, y_tr_seq = self._create_sequences(X_train, y_train)
        X_te_seq, y_te_seq = self._create_sequences(X_test, y_test)

        if len(X_tr_seq) == 0 or len(X_te_seq) == 0:
            raise RuntimeError("Not enough data to create sequences. Consider lowering look_back or using more data.")

        print(f"Shapes -> X_train: {X_tr_seq.shape}, X_test: {X_te_seq.shape}")
        return X_tr_seq, y_tr_seq, X_te_seq, y_te_seq

    def _create_sequences(self, features, target):
        X, y = [], []
        for i in range(self.look_back, len(features)):
            X.append(features[i - self.look_back:i])
            y.append(target[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def build_model(self, input_shape):
        with self.strategy.scope():
            model = Sequential()
            model.add(LSTM(units=self.hidden_dim, input_shape=input_shape, return_sequences=False))
            model.add(Dense(units=1, activation='linear'))
            optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
            model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
            self.model = model
        print("Model built successfully under strategy scope.")
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=60, ckpt_path='best_model.h5'):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
        ]
        self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                      epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
        return self.history

    def evaluate(self, X_test, y_test_scaled):
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mask = y_true.flatten() != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        print(f"Evaluation -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}, y_true, y_pred

    def plot_results(self, y_true, y_pred):
        plt.figure(figsize=(12,5))
        n = min(500, len(y_true))
        plt.plot(y_true[:n], label='Actual SoC')
        plt.plot(y_pred[:n], '--', label='Predicted SoC')
        plt.xlabel('Time step'); plt.ylabel('SoC (%)'); plt.legend(); plt.title('SoC: actual vs predicted (first samples)')
        plt.show()


# ---------------------------
# Run everything
# ---------------------------
def main():
    print("TensorFlow version:", tf.__version__)
    print("Using strategy:", STRATEGY)

    # Load files
    data_df = load_and_combine_mat_files(MAT_FILES, downsample_max=200000)

    predictor = BatterySoCPredictor(look_back=60, hidden_dim=5, learning_rate=1e-3, strategy=STRATEGY)

    X_train, y_train, X_test, y_test = predictor.preprocess(data_df)

    input_shape = (X_train.shape[1], X_train.shape[2])
    predictor.build_model(input_shape)

    predictor.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=60, ckpt_path='best_model.h5')

    metrics, y_true, y_pred = predictor.evaluate(X_test, y_test)
    predictor.plot_results(y_true, y_pred)

    # Save final model
    predictor.model.save('final_lstm_soc_model_local.h5')
    print("Saved final model: final_lstm_soc_model_local.h5")
    return predictor, metrics

if __name__ == "__main__":
    main()
