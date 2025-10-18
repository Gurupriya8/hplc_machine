
# =============================================================
# HPLC Predictive Analytics & Maintenance Model (Enhanced)
#
# Purpose: Forecast utilization, detect anomalies, and compute
#          daily productivity + maintenance insights for HPLC.
# =============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ---------------- #
FILEPATH = r"C:\Users\adurt\Documents\hplc_machine\QLS_SHEET.xlsx"
EXPECTED_ROWS = 241
EXPECTED_COLUMNS = 16
# ---------------------------------------- #

def verify_and_load_excel(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")

    print(f"✅ File located successfully: {filepath}")
    df = pd.read_excel(filepath, nrows=EXPECTED_ROWS)
    if df.shape[1] > EXPECTED_COLUMNS:
        df = df.iloc[:, :EXPECTED_COLUMNS]
    print(f"📊 Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def preprocess(df):
    df = df.dropna(how="all").fillna(0)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found! Check your Excel file contents.")
    print(f"🔢 Numeric columns detected: {numeric_cols}")
    return df, numeric_cols


def descriptive_analytics(df, numeric_cols):
    desc = df[numeric_cols].describe().T
    print("\n--- 📈 Descriptive Analytics ---")
    print(desc)
    return desc


def detect_anomalies(df, numeric_cols):
    iso = IsolationForest(contamination=0.05, random_state=42)
    anomaly_scores = iso.fit_predict(df[numeric_cols])
    df["Anomaly"] = anomaly_scores
    anomaly_count = np.sum(anomaly_scores == -1)
    print(f"\n⚠️  Anomalies detected: {anomaly_count} ({anomaly_count/len(df)*100:.2f}% of data)")
    return df


def utilization_forecast(df, numeric_cols):
    """Estimate utilization % based on duration (minutes) trend."""
    if "duration (minutes)" not in numeric_cols:
        print("\n⚙️  Utilization forecast skipped (no 'duration (minutes)' column).")
        return None, None

    # Assume expected max duration is 60 min (100% utilization)
    df["Utilization_%"] = (df["duration (minutes)"] / 60) * 100
    avg_util = df["Utilization_%"].mean()
    print(f"\n📈 Average Machine Utilization: {avg_util:.2f}%")

    # Predict next 5 runs utilization using linear regression
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Utilization_%"].values
    model = LinearRegression()
    model.fit(X, y)
    next_runs = np.arange(len(df), len(df) + 5).reshape(-1, 1)
    forecast = model.predict(next_runs)
    next_avg = np.mean(forecast)
    print(f"🔮 Forecasted Utilization (next 5 runs): {next_avg:.2f}%")

    return avg_util, next_avg


def failure_prediction(df, numeric_cols):
    """Dummy binary failure predictor based on threshold of duration."""
    if "duration (minutes)" not in numeric_cols:
        print("\n⚙️  Failure prediction skipped (no 'duration (minutes)' column).")
        return None

    df["Failure"] = np.where(df["duration (minutes)"] > df["duration (minutes)"].median(), 0, 1)
    X = df[["duration (minutes)"]]
    y = df["Failure"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🧠 Failure Prediction Accuracy: {acc*100:.2f}%")

    return acc


def maintenance_prediction(df):
    """Predict how soon maintenance might be required."""
    anomaly_rate = (df["Anomaly"] == -1).mean()
    # Heuristic: higher anomaly rate → shorter time to failure
    if anomaly_rate == 0:
        days = "Stable (no anomalies)"
    else:
        days = max(1, int(10 / (anomaly_rate * 10)))  # arbitrary scaling
    print(f"\n🛠️  Estimated Maintenance Need: in ~{days} days")
    return days


def smart_recommendations(df, numeric_cols):
    """Find which numeric parameters correlate with anomalies."""
    print("\n💡 Smart Parameter Insights:")
    if "Anomaly" not in df.columns:
        print("No anomaly column found — skipping recommendations.")
        return

    corr_values = {}
    for col in numeric_cols:
        if col != "Anomaly":
            corr = abs(np.corrcoef(df[col], df["Anomaly"])[0, 1])
            corr_values[col] = corr

    sorted_corr = sorted(corr_values.items(), key=lambda x: x[1], reverse=True)
    top_corr = sorted_corr[:3]
    for param, corr in top_corr:
        print(f"   • {param} correlation with anomalies: {corr:.2f}")


def main():
    print("🚀 Starting Enhanced HPLC Predictive Maintenance System\n")
    df = verify_and_load_excel(FILEPATH)
    df, numeric_cols = preprocess(df)
    descriptive_analytics(df, numeric_cols)
    df = detect_anomalies(df, numeric_cols)
    failure_prediction(df, numeric_cols)
    avg_util, next_util = utilization_forecast(df, numeric_cols)
    maintenance_prediction(df)
    smart_recommendations(df, numeric_cols)
    print("\n✅ Process completed successfully.\n")


if __name__ == "__main__":
    main()
