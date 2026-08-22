import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "household_power_consumption.txt"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "electricity_lstm_model.keras"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)


print("=" * 60)
print("SMART ELECTRICITY FORECASTING")
print("LSTM MODEL EVALUATION")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading LSTM model...")

model = tf.keras.models.load_model(MODEL_FILE)

print("Model loaded successfully")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(SCALER_FILE)

print("Scaler loaded successfully")
print("Number of scaler features:", len(scaler.feature_names_in_))


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading electricity dataset...")

df = pd.read_csv(
    DATA_FILE,
    sep=";",
    na_values="?",
    low_memory=False
)

print("Dataset loaded")
print("Original rows:", len(df))


# ============================================================
# CLEAN DATA
# ============================================================

print("\nCleaning data...")

df["DateTime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

df["Global_reactive_power"] = pd.to_numeric(
    df["Global_reactive_power"],
    errors="coerce"
)

df["Voltage"] = pd.to_numeric(
    df["Voltage"],
    errors="coerce"
)

df["Global_intensity"] = pd.to_numeric(
    df["Global_intensity"],
    errors="coerce"
)

df["Sub_metering_1"] = pd.to_numeric(
    df["Sub_metering_1"],
    errors="coerce"
)

df["Sub_metering_2"] = pd.to_numeric(
    df["Sub_metering_2"],
    errors="coerce"
)

df["Sub_metering_3"] = pd.to_numeric(
    df["Sub_metering_3"],
    errors="coerce"
)


df = df.dropna().copy()

df = df.sort_values("DateTime")

print("Valid rows:", len(df))


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating time features...")

df["hour"] = df["DateTime"].dt.hour

df["day"] = df["DateTime"].dt.day

df["month"] = df["DateTime"].dt.month

df["day_of_week"] = df["DateTime"].dt.dayofweek


# ============================================================
# FEATURES
# ============================================================

features = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour",
    "day",
    "month",
    "day_of_week"
]


print("\nFeatures used by model:")

for i, feature in enumerate(features, start=1):
    print(f"{i:2}. {feature}")


# ============================================================
# PREPARE DATA
# ============================================================

data = df[features].values.astype(np.float32)


# ============================================================
# SCALE DATA
# ============================================================

print("\nScaling data...")

scaled_data = scaler.transform(data)


# ============================================================
# CREATE SEQUENCES
# ============================================================

WINDOW_SIZE = 24

print("\nCreating 24-step sequences...")

X = []
y = []


for i in range(WINDOW_SIZE, len(scaled_data)):

    X.append(
        scaled_data[i - WINDOW_SIZE:i]
    )

    y.append(
        scaled_data[i, 0]
    )


X = np.array(X, dtype=np.float32)

y = np.array(y, dtype=np.float32)


print("X shape:", X.shape)

print("y shape:", y.shape)


# ============================================================
# USE A SMALL TEST SET
# ============================================================

TEST_SIZE = 5000

if len(X) > TEST_SIZE:

    X_test = X[-TEST_SIZE:]

    y_test = y[-TEST_SIZE:]

else:

    X_test = X

    y_test = y


print("\nEvaluation samples:", len(X_test))


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning LSTM predictions...")

pred_scaled = model.predict(
    X_test,
    verbose=0
)


pred_scaled = pred_scaled.reshape(-1)


# ============================================================
# CONVERT POWER BACK TO ORIGINAL SCALE
# ============================================================

power_min = scaler.data_min_[0]

power_max = scaler.data_max_[0]


y_actual = (
    y_test * (power_max - power_min)
    + power_min
)


y_predicted = (
    pred_scaled * (power_max - power_min)
    + power_min
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_actual,
    y_predicted
)

rmse = np.sqrt(
    mean_squared_error(
        y_actual,
        y_predicted
    )
)


# Mean Absolute Percentage Error

non_zero = np.abs(y_actual) > 0.01

mape = np.mean(
    np.abs(
        (y_actual[non_zero] - y_predicted[non_zero])
        / y_actual[non_zero]
    )
) * 100


# Simple accuracy-style value

accuracy = max(
    0,
    100 - mape
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE      : {mae:.4f} kW")

print(f"RMSE     : {rmse:.4f} kW")

print(f"MAPE     : {mape:.2f}%")

print(f"Accuracy : {accuracy:.2f}%")

print("=" * 60)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\nSample predictions:")

print("-" * 50)

for i in range(min(10, len(y_actual))):

    error = abs(
        y_actual[i] - y_predicted[i]
    )

    print(
        f"{i+1:2}. "
        f"Actual: {y_actual[i]:.3f} kW   "
        f"Predicted: {y_predicted[i]:.3f} kW   "
        f"Error: {error:.3f} kW"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "Actual_Power_kW": y_actual,

    "Predicted_Power_kW": y_predicted,

    "Absolute_Error_kW":
        np.abs(
            y_actual - y_predicted
        )

})


results_file = os.path.join(
    BASE_DIR,
    "model_evaluation_results.csv"
)


results.to_csv(
    results_file,
    index=False
)


print("\nResults saved to:")

print(results_file)


print("\nEvaluation completed successfully.")