#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import recall_score, precision_score, confusion_matrix

# =========================================================
# CONFIG
# =========================================================

DATA_FILE = "GHCNh_USW00014836_por.psv"  # <-- YOUR RAW DATA FILE

DATE_START = pd.Timestamp("2010-01-01T00:00:00Z")
DATE_END   = pd.Timestamp("2025-12-31T23:59:59Z")

FOG_THRESH = 1.0   # km (METAR standard)
HOURS_AHEAD = 3

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
# TARGET (BINARY FOG)
# =========================================================

def build_target(df):

    df = df.sort_values(["Station_ID","datetime"])

    df["vis_future"] = df.groupby("Station_ID")["visibility"].shift(-HOURS_AHEAD)

    df["target"] = (df["vis_future"] < FOG_THRESH).astype(int)

    return df

# =========================================================
# BALANCING (NO SMOTE)
# =========================================================

def balance(X, y):

    df = X.copy()
    df["y"] = y.values

    fog = df[df["y"] == 1]
    no_fog = df[df["y"] == 0]

    n = min(len(fog), len(no_fog))

    fog = fog.sample(n=n, random_state=42)
    no_fog = no_fog.sample(n=n, random_state=42)

    df_bal = pd.concat([fog, no_fog]).sample(frac=1, random_state=42)

    return df_bal.drop(columns=["y"]), df_bal["y"]

# =========================================================
# PLOT
# =========================================================

def plot_cm(y_true, y_pred, out_path):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Fog Detection Confusion Matrix")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# =========================================================
# MAIN
# =========================================================

def main():

    print("Loading:", DATA_FILE)
    df = load_data(DATA_FILE)

    df = df[(df["datetime"] >= DATE_START) & (df["datetime"] <= DATE_END)]

    if "Station_ID" not in df.columns:
        df["Station_ID"] = "STN"

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

    # =====================================================
    # BALANCE DATA
    # =====================================================
    print("\nBalancing dataset...")
    X, y = balance(X, y)

    print("Class distribution:")
    print(y.value_counts())

    # =====================================================
    # TIME SPLIT
    # =====================================================
    tscv = TimeSeriesSplit(n_splits=5)

    recalls = []

    for fold, (tr, te) in enumerate(tscv.split(X), 1):

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42
        )

        model.fit(X_tr, y_tr)

        # =================================================
        # PROBABILITY THRESHOLD (KEY FIX)
        # =================================================
        proba = model.predict_proba(X_te)[:, 1]

        THRESH = 0.20

        y_pred = (proba > THRESH).astype(int)

        recall = recall_score(y_te, y_pred)
        precision = precision_score(y_te, y_pred)

        print(f"\nFold {fold}")
        print(f"Recall:    {recall:.3f}")
        print(f"Precision: {precision:.3f}")

        recalls.append(recall)

    print("\n====================")
    print(f"MEAN RECALL: {np.mean(recalls):.3f}")
    print("====================")

    # =====================================================
    # FINAL MODEL + PLOT
    # =====================================================
    model.fit(X_tr, y_tr)
    final_pred = (model.predict_proba(X_te)[:,1] > THRESH).astype(int)

    plot_cm(y_te, final_pred, "fog_confusion_matrix.png")

    print("\nSaved: fog_confusion_matrix.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()