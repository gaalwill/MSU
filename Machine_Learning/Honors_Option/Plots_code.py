#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

DATA_FILE = "GHCNh_USW00014836_por.psv"

FOG_THRESH = 1.0
HOURS_AHEAD = 3
THRESH = 0.20

TOP_N_FEATURES = 20   # ✅ NEW: top features to plot

# =========================================================
# LOAD
# =========================================================

def load_data(path):
    df = pd.read_csv(path, sep="|", low_memory=False)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    else:
        df["datetime"] = pd.to_datetime(
            df[["Year","Month","Day","Hour","Minute"]],
            utc=True
        )

    return df

# =========================================================
# FEATURES
# =========================================================

def build_features(df):

    df = df.sort_values(["Station_ID", "datetime"])

    for c in ["temperature", "dew_point_temperature", "wind_speed", "visibility"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["dew_spread"] = df["temperature"] - df["dew_point_temperature"]

    hour = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2*np.pi*hour/24)
    df["hour_cos"] = np.cos(2*np.pi*hour/24)

    g = df.groupby("Station_ID")["visibility"]

    df["vis_lag_3"]  = g.shift(3)
    df["vis_lag_6"]  = g.shift(6)
    df["vis_lag_12"] = g.shift(12)

    df["vis_trend_3"] = df["visibility"] - g.shift(3)

    return df

# =========================================================
# TARGET
# =========================================================

def build_target(df):

    df = df.sort_values(["Station_ID","datetime"])

    df["vis_future"] = df.groupby("Station_ID")["visibility"].shift(-HOURS_AHEAD)

    df["target"] = (df["vis_future"] < FOG_THRESH).astype(int)

    return df

# =========================================================
# BALANCE
# =========================================================

def balance(X, y):

    df = X.copy()
    df["y"] = y.values

    fog = df[df["y"] == 1]
    no_fog = df[df["y"] == 0]

    n = min(len(fog), len(no_fog))

    df = pd.concat([
        fog.sample(n=n, random_state=42),
        no_fog.sample(n=n, random_state=42)
    ]).sample(frac=1, random_state=42)

    return df.drop(columns=["y"]), df["y"]

# =========================================================
# PLOTS
# =========================================================

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Fog Detection Confusion Matrix")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()


def plot_feature_importance(model, features):

    importances = model.feature_importances_

    idx = np.argsort(importances)[::-1][:TOP_N_FEATURES]  # ✅ TOP 20 ONLY

    plt.figure(figsize=(9,6))

    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(range(len(idx)), np.array(features)[idx], rotation=45, ha="right")

    plt.ylabel("Importance")
    plt.title(f"Top {TOP_N_FEATURES} Feature Importance")

    plt.tight_layout()
    plt.savefig("feature_importance_top20.png", dpi=150)
    plt.close()

# =========================================================
# MAIN
# =========================================================

def main():

    print("Loading data...")
    df = load_data(DATA_FILE)

    if "Station_ID" not in df.columns:
        df["Station_ID"] = "STN"

    df = df[(df["datetime"] >= "2010-01-01") & (df["datetime"] <= "2025-12-31")]

    df = build_features(df)
    df = build_target(df)

    df = df.dropna(subset=["target"])

    features = [
        "temperature",
        "dew_point_temperature",
        "dew_spread",
        "wind_speed",
        "hour_sin",
        "hour_cos",
        "vis_lag_3",
        "vis_lag_6",
        "vis_lag_12",
        "vis_trend_3"
    ]

    X = df[features]
    y = df["target"].astype(int)

    print("Balancing dataset...")
    X, y = balance(X, y)

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_imp, y)

    # =====================================================
    # PREDICT
    # =====================================================

    proba = model.predict_proba(X_imp)[:, 1]
    y_pred = (proba > THRESH).astype(int)

    # =====================================================
    # PLOTS
    # =====================================================

    plot_confusion(y, y_pred)
    plot_feature_importance(model, features)

    print("\nSaved:")
    print(" - confusion_matrix.png")
    print(" - feature_importance_top20.png")


if __name__ == "__main__":
    main()
