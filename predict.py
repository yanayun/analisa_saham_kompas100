"""Inferensi model: probabilitas tiap saham naik >= 3% dalam 5 hari.

Membaca bundle models/saham_model.joblib yang dihasilkan train.py,
lalu menerapkannya pada baris TERAKHIR tiap saham.

Keluaran:
    results/signals.csv
    results/signals.json
"""

import glob
import json
import os

import joblib
import numpy as np
import pandas as pd

DATA_DIR = "data"
RESULT_DIR = "results"
MODEL_PATH = os.path.join("models", "saham_model.joblib")

SIGNALS_CSV = os.path.join(RESULT_DIR, "signals.csv")
SIGNALS_JSON = os.path.join(RESULT_DIR, "signals.json")


def load_bundle(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model tidak ditemukan di {path}. Jalankan: python train.py"
        )
    bundle = joblib.load(path)
    if "model" not in bundle or "features" not in bundle:
        raise ValueError(
            "Model disimpan dalam format lama (objek mentah). "
            "Jalankan ulang: python train.py"
        )
    return bundle


def predict_proba(bundle, X):
    """Probabilitas terkalin: model + kalibrator (bila ada)."""
    raw = np.asarray(bundle["model"].predict_proba(X)[:, 1]).reshape(-1, 1)
    calibrator = bundle.get("calibrator")
    if calibrator is None:
        return raw.ravel()
    return calibrator.predict_proba(raw)[:, 1]


def load_latest_rows(features):
    """Baris terakhir tiap saham yang fiturnya lengkap."""
    rows = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_features.csv")))

    for path in files:
        ticker = os.path.basename(path).replace("_features.csv", "")

        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.loc[~df.index.isna()].sort_index()

        missing = [c for c in features if c not in df.columns]
        if missing:
            continue

        for col in features + ["Close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=features)

        if df.empty:
            continue

        last = df.iloc[[-1]].copy()
        last["ticker"] = ticker
        last["date"] = last.index
        rows.append(last)

    if not rows:
        return pd.DataFrame()

    data = pd.concat(rows, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


def predict_all(bundle=None, path=MODEL_PATH):
    """DataFrame berisi probabilitas + fitur terkini tiap saham."""
    if bundle is None:
        bundle = load_bundle(path)

    features = bundle["features"]
    threshold = float(bundle.get("threshold", 0.5))

    data = load_latest_rows(features)
    if data.empty:
        return data

    data["probability_up_3pct"] = predict_proba(bundle, data[features])
    data["signal"] = np.where(
        data["probability_up_3pct"] >= threshold, "PANTAU BUY",
        np.where(data["probability_up_3pct"] <= 1 - threshold, "PANTAU SELL", "HOLD"),
    )
    return data.sort_values("probability_up_3pct", ascending=False).reset_index(drop=True)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    bundle = load_bundle()
    data = predict_all(bundle)

    if data.empty:
        print("Tidak ada saham yang bisa diprediksi.")
        return

    out = data[[
        "date", "ticker", "Close", "probability_up_3pct", "signal"
    ]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["Close"] = out["Close"].round(2)
    out["probability_up_3pct"] = out["probability_up_3pct"].round(4)
    out.to_csv(SIGNALS_CSV, index=False)

    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_threshold": float(bundle.get("threshold", 0.5)),
        "n_stocks": int(len(out)),
        "signals": out.to_dict("records"),
    }
    with open(SIGNALS_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"=== SINYAL ({len(out)} saham) ===")
    print(out.to_string(index=False, formatters={
        "probability_up_3pct": "{:.1%}".format,
    }))
    print(f"\nTersimpan: {SIGNALS_CSV}")


if __name__ == "__main__":
    main()
