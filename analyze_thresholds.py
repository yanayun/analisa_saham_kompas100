"""Laporan pemilihan ambang probabilitas dan evaluasi akhir.

Membaca hasil yang dihasilkan train.py:
    results/threshold_analysis.csv   ( tabel ambang vs precision/recall/F1 )
    results/test_predictions.csv     ( prediksi satu kali di data test )

Ambang TIDAK dioptimalkan ulang di sini -- tabel hanya laporan.
Keputusan ambang diambil train.py dari data validation saja.
"""

import json
import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RESULT_DIR = "results"
THRESHOLD_FILE = os.path.join(RESULT_DIR, "threshold_analysis.csv")
TEST_FILE = os.path.join(RESULT_DIR, "test_predictions.csv")
METRICS_FILE = os.path.join(RESULT_DIR, "metrics.json")

PROB = "probability_up_3pct"
TARGET = "target"


def main():
    for path in (THRESHOLD_FILE, TEST_FILE):
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} tidak ada. Jalankan: python train.py"
            )

    table = pd.read_csv(THRESHOLD_FILE)
    test = pd.read_csv(TEST_FILE)

    with open(METRICS_FILE, encoding="utf-8") as fh:
        metrics = json.load(fh)

    threshold = float(metrics["threshold"])

    print("=" * 58)
    print("AMBANG YANG DIPILIH (dari validation)")
    print("=" * 58)
    print(table.to_string(index=False, formatters={
        "threshold": "{:.2f}".format,
        "precision": "{:.2%}".format,
        "recall": "{:.2%}".format,
        "f1": "{:.3f}".format,
    }))
    print(f"\nAmbang terpilih: {threshold:.2f}")

    print("\n" + "=" * 58)
    print("EVALUASI AKHIR (test, tanpa optimasi ulang)")
    print("=" * 58)
    print(f"Periode test : {metrics['test_period'][0]} - {metrics['test_period'][1]}")
    print(f"Jumlah baris : {len(test):,}")

    selected = test[PROB] >= threshold
    n_selected = int(selected.sum())
    print(f"Sinyal       : {n_selected:,} ({n_selected / len(test):.1%} dari test)")

    if n_selected:
        print(f"Precision    : {precision_score(test[TARGET], selected, zero_division=0):.2%}")
        print(f"Recall       : {recall_score(test[TARGET], selected, zero_division=0):.2%}")
        print(f"F1           : {f1_score(test[TARGET], selected, zero_division=0):.3f}")

    print(f"ROC-AUC      : {metrics['roc_auc']:.4f}" if metrics.get("roc_auc") else "ROC-AUC      : n/a")
    print(f"Base rate    : {test[TARGET].mean():.2%}")

    cm = confusion_matrix(
        test[TARGET], selected.astype(int), labels=[0, 1]
    )
    print("\nConfusion matrix (baris aktual, kolom prediksi):")
    print(cm)

    print("\nCatatan: precision adalah peluang sinyal benar, "
          "bukan jaminan return positif.")


if __name__ == "__main__":
    main()
