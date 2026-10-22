"""Training model klasifikasi: apakah saham naik >= 3% dalam 5 hari.

Alur:
  1. Muat semua data/<KODE>_features.csv
  2. Split kronologis 70/15/15 dengan purge + embargo (anti kebocoran)
  3. Latih XGBoost dengan early stopping pada validation
  4. Pilih ambang probabilitas optimal dari validation
  5. Evaluasi sekali di test, simpan metrik + prediksi

Model disimpan sebagai BUNDLE (bukan objek mentah):
    {"model": ..., "features": [...], "threshold": 0.62, ...}
bundle inilah yang dipakai predict.py, recommend.py, dan backtest.py.
"""

import glob
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from features import (
    FEATURE_COLUMNS,
    FUTURE_RETURN_COLUMN,
    TARGET_COLUMN,
)

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "models"
RESULT_DIR = "results"

MODEL_PATH = os.path.join(MODEL_DIR, "saham_model.joblib")
LEGACY_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_saham.joblib")
METRICS_PATH = os.path.join(RESULT_DIR, "metrics.json")
VALIDATION_PATH = os.path.join(RESULT_DIR, "validation_predictions.csv")
TEST_PATH = os.path.join(RESULT_DIR, "test_predictions.csv")
THRESHOLD_PATH = os.path.join(RESULT_DIR, "threshold_analysis.csv")

PURGE_DAYS = 5
TRAIN_SHARE = 0.70
VALID_SHARE = 0.15

# Minimal sinyal agar sebuah ambang layak dipilih.
MIN_SIGNALS = 150
MIN_SIGNAL_FRACTION = 0.10


def create_directories():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def load_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_features.csv")))

    if not files:
        raise FileNotFoundError(
            f"Tidak ada file *_features.csv di {DATA_DIR}/. Jalankan features.py dulu."
        )

    frames = []
    print("Memuat data...")

    for path in files:
        ticker = os.path.basename(path).replace("_features.csv", "")
        df = pd.read_csv(path)

        if "date" not in df.columns:
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            elif "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "date"})
            else:
                print(f"  melewati {ticker}: kolom tanggal tidak ada")
                continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["ticker"] = ticker

        missing = [
            c for c in FEATURE_COLUMNS + [TARGET_COLUMN, FUTURE_RETURN_COLUMN]
            if c not in df.columns
        ]
        if missing:
            print(f"  melewati {ticker}: kolom hilang {missing}")
            continue

        numeric = FEATURE_COLUMNS + ["Close", TARGET_COLUMN, FUTURE_RETURN_COLUMN]
        for col in numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(
            subset=["date", *FEATURE_COLUMNS, TARGET_COLUMN, FUTURE_RETURN_COLUMN]
        ).copy()

        if df.empty:
            print(f"  melewati {ticker}: tidak ada baris valid")
            continue

        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        df = df[df[TARGET_COLUMN].isin([0, 1])]

        frames.append(df)
        print(f"  {ticker}: {len(df):,} baris")

    if not frames:
        raise ValueError("Tidak ada data valid untuk training.")

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"\nTotal baris : {len(data):,}")
    print(f"Jumlah saham: {data['ticker'].nunique()}")
    print(f"Periode     : {data['date'].min().date()} - {data['date'].max().date()}")
    print(f"Target positif: {data[TARGET_COLUMN].mean():.2%}")

    return data


def split_data(data):
    """Split kronologis per tanggal unik + purge di sekitar batas."""
    dates = np.array(sorted(data["date"].unique()))
    n = len(dates)

    train_end = int(n * TRAIN_SHARE)
    valid_end = int(n * (TRAIN_SHARE + VALID_SHARE))

    if train_end <= PURGE_DAYS or valid_end <= train_end + PURGE_DAYS or valid_end >= n:
        raise ValueError("Data terlalu sedikit untuk split train/valid/test.")

    train_dates = dates[: train_end - PURGE_DAYS]
    valid_dates = dates[train_end: valid_end - PURGE_DAYS]
    test_dates = dates[valid_end:]

    train = data[data["date"].isin(train_dates)]
    valid = data[data["date"].isin(valid_dates)]
    test = data[data["date"].isin(test_dates)]

    if train.empty or valid.empty or test.empty:
        raise ValueError("Ada periode yang kosong setelah split.")

    print("\nPembagian data:")
    for name, part in (("Train", train), ("Validation", valid), ("Test", test)):
        print(f"  {name:<11}: {len(part):,} baris | "
              f"{part['date'].min().date()} - {part['date'].max().date()}")

    return train, valid, test


def prepare_xy(df):
    X = df[FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        raise ValueError("Masih ada fitur yang kosong.")
    return X, df[TARGET_COLUMN].astype(int).copy()


def save_predictions(df, probabilities, output_path):
    out = df[["date", "ticker", "Close", TARGET_COLUMN]].copy()
    out["probability_up_3pct"] = probabilities
    out["predicted_target"] = (out["probability_up_3pct"] >= 0.50).astype(int)
    out["actual_future_return_5d"] = df[FUTURE_RETURN_COLUMN].values
    out.to_csv(output_path, index=False)
    return out


def pick_threshold(valid, probabilities):
    """Ambang dipilih dari validation saja.

    Tujuannya alat penyaring: pada selektivitas yang masih praktis
    (minimal MIN_SIGNAL_FRACTION sinyal), pilih precision tertinggi.
    F1 tidak dipakai karena recall mendominasi dan mendorong ambang
    ke daerah yang hampir memilih semua saham.
    """
    y = valid[TARGET_COLUMN].astype(int).values
    min_signals = max(MIN_SIGNALS, int(len(valid) * MIN_SIGNAL_FRACTION))
    rows = []

    for threshold in np.arange(0.10, 0.91, 0.01):
        selected = probabilities >= threshold
        count = int(selected.sum())
        if count < min_signals:
            continue
        rows.append({
            "threshold": round(float(threshold), 4),
            "signals": count,
            "signal_fraction": count / len(valid),
            "precision": precision_score(y, selected, zero_division=0),
            "recall": recall_score(y, selected, zero_division=0),
            "f1": f1_score(y, selected, zero_division=0),
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return 0.50, pd.DataFrame()

    # Precision tertinggi; seri -> ambang lebih tinggi (lebih selektif).
    best = table.sort_values(
        ["precision", "threshold"], ascending=[False, False]
    ).iloc[0]
    return float(best["threshold"]), table


def main():
    create_directories()

    data = load_data()
    train, valid, test = split_data(data)

    X_train, y_train = prepare_xy(train)
    X_valid, y_valid = prepare_xy(valid)
    X_test, y_test = prepare_xy(test)

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = negatives / max(positives, 1)

    print(f"\nKelas train -> 0: {negatives:,} | 1: {positives:,} "
          f"| scale_pos_weight: {scale_pos_weight:.4f}")

    model = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=40,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    print("\nMelatih XGBoost...")
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    print(f"Training selesai. Best iteration: {model.best_iteration}")

    # --- validation ---------------------------------------------------
    # scale_pos_weight memperbaiki peringkat (IC lebih tinggi) tetapi
    # merusak kalibrasi probabilitas. Platt scaling memasang ulang
    # probabilitas pada validation TANPA mengubah peringkat.
    calibrator = LogisticRegression(C=1e6, solver="lbfgs")
    calibrator.fit(model.predict_proba(X_valid)[:, 1].reshape(-1, 1), y_valid)

    valid_prob = calibrator.predict_proba(model.predict_proba(X_valid)[:, 1].reshape(-1, 1))[:, 1]
    threshold, threshold_table = pick_threshold(valid, valid_prob)
    save_predictions(valid, valid_prob, VALIDATION_PATH)

    if not threshold_table.empty:
        threshold_table.to_csv(THRESHOLD_PATH, index=False)

    # --- test ---------------------------------------------------------
    test_prob = calibrator.predict_proba(model.predict_proba(X_test)[:, 1].reshape(-1, 1))[:, 1]
    predictions = (test_prob >= threshold).astype(int)
    save_predictions(test, test_prob, TEST_PATH)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_prob) if y_test.nunique() == 2 else None
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    importance = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    metrics = {
        "model_type": "XGBClassifier",
        "threshold": threshold,
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "train_period": [str(train["date"].min().date()), str(train["date"].max().date())],
        "validation_period": [str(valid["date"].min().date()), str(valid["date"].max().date())],
        "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
        "n_features": len(FEATURE_COLUMNS),
        "positive_rate_test": float(y_test.mean()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "feature_importance": [
            {"feature": f, "importance": round(float(i), 6)} for f, i in importance
        ],
        "features": FEATURE_COLUMNS,
    }

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "features": FEATURE_COLUMNS,
        "threshold": threshold,
        "target": "naik >= 3% dalam 5 hari",
        "horizon_days": 5,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PATH)

    print("\n" + "=" * 56)
    print("HASIL EVALUASI MODEL")
    print("=" * 56)
    print(f"Ambang probabilitas : {threshold:.3f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC   : n/a")
    print(f"Positive rate test: {y_test.mean():.2%}")

    print("\nConfusion matrix (baris aktual, kolom prediksi):")
    print(cm)

    print("\nFitur paling berpengaruh:")
    for name, score in importance[:10]:
        print(f"  {name:<18} {score:.4f}")

    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print(f"\nModel   : {MODEL_PATH}")
    print(f"Metrik  : {METRICS_PATH}")
    print(f"Validasi: {VALIDATION_PATH}")
    print(f"Test    : {TEST_PATH}")


if __name__ == "__main__":
    main()
