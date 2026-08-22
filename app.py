from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "data/household_power_consumption.txt"
RESULTS_PATH = "model_evaluation_results.csv"

# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_NAME = "LSTM"
INPUT_WINDOW = 24

MODEL_MAE = 0.3876
MODEL_RMSE = 0.5013
MODEL_MAPE = 54.51
MODEL_ACCURACY = 45.49
TEST_SAMPLES = 5000
# ============================================================
# LOAD SMALL DATASET FOR LIVE SIMULATION
# ============================================================

print("Loading electricity data...")

USECOLS = [
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

try:

    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        na_values="?",
        usecols=USECOLS,
        nrows=100000,
        low_memory=True
    )

    print("Dataset loaded")
    print("Rows:", len(df))

except Exception as e:

    print("Dataset loading error:")
    print(e)

    df = pd.DataFrame()


# ============================================================
# PREPROCESS
# ============================================================

if not df.empty:

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        errors="coerce"
    )

    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna().reset_index(drop=True)

    print("Valid rows:", len(df))


# ============================================================
# LOAD ACTUAL LSTM RESULTS
# ============================================================

print("Loading LSTM prediction results...")

try:

    results_df = pd.read_csv(
        RESULTS_PATH
    )

    print("LSTM results loaded")
    print("Prediction rows:", len(results_df))

except Exception as e:

    print("Could not load LSTM results:")
    print(e)

    results_df = pd.DataFrame()


# ============================================================
# SIMULATION INDEX
# ============================================================

current_index = 0


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# LIVE API
# ============================================================

@app.route("/api/live")
def live_data():

    global current_index

    if df.empty:

        return jsonify({
            "success": False,
            "error": "Electricity dataset unavailable"
        }), 500


    # Restart after reaching end
    if current_index >= len(df):

        current_index = 0


    row = df.iloc[current_index]


    # ========================================================
    # CURRENT REAL DATASET READING
    # ========================================================

    current_power = float(
        row["Global_active_power"]
    )

    reactive_power = float(
        row["Global_reactive_power"]
    )

    voltage = float(
        row["Voltage"]
    )

    current = float(
        row["Global_intensity"]
    )

    meter1 = float(
        row["Sub_metering_1"]
    )

    meter2 = float(
        row["Sub_metering_2"]
    )

    meter3 = float(
        row["Sub_metering_3"]
    )


    # ========================================================
    # ACTUAL LSTM PREDICTION
    # ========================================================

    if not results_df.empty:

        result_index = current_index % len(results_df)

        prediction = float(
            results_df.iloc[result_index]
            ["Predicted_Power_kW"]
        )

    else:

        # fallback
        prediction = current_power


    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = row["datetime"].strftime(
        "%d-%m-%Y %H:%M:%S"
    )


    # ========================================================
    # MOVE TO NEXT READING
    # ========================================================

    current_index += 1


    return jsonify({

        "success": True,

        "timestamp": timestamp,

        "current_power":
            round(current_power, 3),

        "predicted_power":
            round(prediction, 3),

        "reactive_power":
            round(reactive_power, 3),

        "voltage":
            round(voltage, 2),

        "current":
            round(current, 2),

        "sub_meter_1":
            round(meter1, 2),

        "sub_meter_2":
            round(meter2, 2),

        "sub_meter_3":
            round(meter3, 2),

        "readings_used":
            INPUT_WINDOW,

        "model":
            MODEL_NAME,

        "status":
            "LIVE SIMULATION"

    })


# ============================================================
# MODEL PERFORMANCE API
# ============================================================

@app.route("/api/performance")
def performance():

    try:

        results_path = "model_evaluation_results.csv"

        results = pd.read_csv(results_path)

        # Limit data sent to browser
        chart_data = results.head(100).copy()

        actual = chart_data["Actual_Power_kW"].tolist()
        predicted = chart_data["Predicted_Power_kW"].tolist()

        # Overall metrics from evaluation
        mae = float(
            results["Absolute_Error_kW"].mean()
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        results["Actual_Power_kW"]
                        -
                        results["Predicted_Power_kW"]
                    ) ** 2
                )
            )
        )

        return jsonify({

            "success": True,

            "model": "LSTM",

            "mae": round(mae, 4),

            "rmse": round(rmse, 4),

            "test_samples": len(results),

            "actual": actual,

            "predicted": predicted

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500
# ============================================================
# RESET
# ============================================================

@app.route("/api/reset")
def reset():

    global current_index

    current_index = 0

    return jsonify({

        "success": True,

        "message":
            "Simulation reset successfully"

    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({

        "success": True,

        "server": "online",

        "dataset_rows":
            len(df),

        "lstm_predictions":
            len(results_df),

        "model":
            MODEL_NAME,

        "mode":
            "LSTM PREDICTION STREAM"

    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )