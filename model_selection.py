"""Pemilihan model: membandingkan beberapa varian secara empiris.

Yang diukur pada periode test (tidak pernah dipakai untuk training):
  * ROC-AUC   -- kualitas klasifikasi (bila berlaku)
  * IC        -- korelasi peringkat (Spearman) skor vs return 5 hari
  * IC harian -- IC rata-rata per tanggal (lebih jujur dari IC pooled)
  * Top-decile-- return rata-rata 10 saham peringkat teratas tiap tanggal,
                 dibandingkan rata-rata seluruh semesta

Skrip ini murni untuk memilih model; pipeline harian ada di daily.py.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from features import FEATURE_COLUMNS, FUTURE_RETURN_COLUMN, TARGET_COLUMN

DATA_DIR = "data"
BENCH_FILE = os.path.join(DATA_DIR, "JKSE.csv")
PURGE_DAYS = 5
TOP_N = 10


def load_all():
    frames = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_features.csv"))):
        ticker = os.path.basename(path).replace("_features.csv", "")
        df = pd.read_csv(path)
        if "date" not in df.columns:
            for candidate in ("Date", "Datetime"):
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "date"})
                    break
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["ticker"] = ticker
        for col in FEATURE_COLUMNS + [TARGET_COLUMN, FUTURE_RETURN_COLUMN, "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, FUTURE_RETURN_COLUMN])
        if not df.empty:
            frames.append(df)

    data = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])

    # Return IHSG 5 hari ke depan, untuk target excess return.
    bench = pd.read_csv(BENCH_FILE, index_col=0)
    bench.index = pd.to_datetime(bench.index, errors="coerce")
    bench = bench.loc[~bench.index.isna()].sort_index()
    bench_close = pd.to_numeric(bench["Close"], errors="coerce").dropna()
    bench_fwd = bench_close.shift(-5) / bench_close - 1
    bench_fwd.name = "bench_fwd_5d"

    data = data.merge(bench_fwd, left_on="date", right_index=True, how="left")
    data["excess_5d"] = data[FUTURE_RETURN_COLUMN] - data["bench_fwd_5d"]

    return data.reset_index(drop=True)


def split(data):
    dates = np.sort(data["date"].unique())
    n = len(dates)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train = data[data["date"].isin(dates[: train_end - PURGE_DAYS])]
    valid = data[data["date"].isin(dates[train_end: valid_end - PURGE_DAYS])]
    test = data[data["date"].isin(dates[valid_end:])]
    return train, valid, test


def _values(x):
    """Terima Series atau ndarray, kembalikan ndarray 1-D."""
    return np.asarray(x).ravel()


def ic_by_date(scores, returns, dates):
    """IC rata-rata per tanggal (hanya tanggal dengan >= 8 saham)."""
    scores = _values(scores)
    returns = _values(returns)
    ics = []
    for _, idx in pd.Series(range(len(dates))).groupby(dates.values).groups.items():
        s = scores[idx]
        r = returns[idx]
        mask = np.isfinite(s) & np.isfinite(r)
        if mask.sum() >= 8:
            rho, _ = spearmanr(s[mask], r[mask])
            if np.isfinite(rho):
                ics.append(rho)
    return float(np.mean(ics)) if ics else 0.0


def top_decile(scores, returns, dates, top_n=TOP_N):
    """Return rata-rata 10 saham teratas per tanggal vs semesta."""
    scores = _values(scores)
    returns = _values(returns)
    top_rets, universe_rets = [], []
    for _, idx in pd.Series(range(len(dates))).groupby(dates.values).groups.items():
        r = returns[idx]
        s = scores[idx]
        mask = np.isfinite(r) & np.isfinite(s)
        if mask.sum() < 20:
            continue
        order = np.argsort(-s[mask])
        r_valid = r[mask]
        top_rets.append(r_valid[order[:top_n]].mean())
        universe_rets.append(r_valid.mean())
    if not top_rets:
        return 0.0, 0.0
    return float(np.mean(top_rets) * 100), float(np.mean(universe_rets) * 100)


def evaluate(name, scores, test, extra=None):
    out = {
        "model": name,
        "ic_daily": ic_by_date(scores, test[FUTURE_RETURN_COLUMN], test["date"]),
    }

    if TARGET_COLUMN in test and test[TARGET_COLUMN].nunique() == 2:
        out["roc_auc"] = float(
            roc_auc_score(test[TARGET_COLUMN], scores)
        )

    top, univ = top_decile(scores, test[FUTURE_RETURN_COLUMN], test["date"])
    out["top10_avg_5d_pct"] = round(top, 3)
    out["universe_avg_5d_pct"] = round(univ, 3)
    out["top10_edge_pct"] = round(top - univ, 3)

    if extra:
        out.update(extra)

    print(f"{name:<26} IC {out['ic_daily']:+.4f} | "
          f"top10 {out['top10_avg_5d_pct']:+.3f}% vs semesta "
          f"{out['universe_avg_5d_pct']:+.3f}% (edge "
          f"{out['top10_edge_pct']:+.3f}%)"
          + (f" | AUC {out['roc_auc']:.4f}" if "roc_auc" in out else ""))
    return out


def main():
    data = load_all()
    train, valid, test = split(data)

    print(f"Train {len(train):,} | Valid {len(valid):,} | Test {len(test):,}")
    print(f"Periode test: {test['date'].min().date()} - {test['date'].max().date()}\n")

    X_train = train[FEATURE_COLUMNS]
    X_valid = valid[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    results = []

    # ---- baseline teknikal sederhana ---------------------------------
    print("--- Baseline ---")
    for col in ("return_20d", "rel_20d", "rsi_14"):
        results.append(evaluate(f"baseline:{col}", test[col], test))

    # ---- 1. klasifikasi tanpa scale_pos_weight -----------------------
    print("\n--- XGBoost klasifikasi ---")
    clf = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=40, random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, train[TARGET_COLUMN], eval_set=[(X_valid, valid[TARGET_COLUMN])], verbose=False)
    results.append(evaluate("clf natural", clf.predict_proba(X_test)[:, 1], test,
                            {"best_iter": int(clf.best_iteration)}))

    # ---- 2. klasifikasi dengan scale_pos_weight ----------------------
    spw = (train[TARGET_COLUMN] == 0).sum() / max((train[TARGET_COLUMN] == 1).sum(), 1)
    clf2 = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=40, scale_pos_weight=float(spw),
        random_state=42, n_jobs=-1,
    )
    clf2.fit(X_train, train[TARGET_COLUMN], eval_set=[(X_valid, valid[TARGET_COLUMN])], verbose=False)
    results.append(evaluate("clf scale_pos_weight", clf2.predict_proba(X_test)[:, 1], test,
                            {"scale_pos_weight": round(float(spw), 3)}))

    # ---- 2b. dikalibrasi ulang di validation -------------------------
    print("\n--- Kalibrasi probabilitas ---")
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    prob_valid = clf2.predict_proba(X_valid)[:, 1]
    y_valid = valid[TARGET_COLUMN].astype(int)
    prob_test = clf2.predict_proba(X_test)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(prob_valid, y_valid)
    results.append(evaluate("clf spw + isotonic", iso.predict(prob_test), test))

    platt = LogisticRegression(C=1e6, solver="lbfgs")
    platt.fit(np.asarray(prob_valid).reshape(-1, 1), y_valid)
    results.append(evaluate("clf spw + platt", platt.predict_proba(np.asarray(prob_test).reshape(-1, 1))[:, 1], test))

    # ---- 3. regresi return 5 hari ------------------------------------
    print("\n--- XGBoost regresi ---")
    reg = XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="reg:squarederror", early_stopping_rounds=40,
        random_state=42, n_jobs=-1,
    )
    reg.fit(X_train, train[FUTURE_RETURN_COLUMN], eval_set=[(X_valid, valid[FUTURE_RETURN_COLUMN])], verbose=False)
    results.append(evaluate("reg return_5d", reg.predict(X_test), test))

    # ---- 4. regresi excess return vs IHSG ---------------------------
    reg2 = XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="reg:squarederror", early_stopping_rounds=40,
        random_state=42, n_jobs=-1,
    )
    reg2.fit(X_train, train["excess_5d"], eval_set=[(X_valid, valid["excess_5d"])], verbose=False)
    results.append(evaluate("reg excess_vs_ihsg", reg2.predict(X_test), test))

    # ---- 5. regresi gabungan: return + excess ------------------------
    reg3 = XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="reg:squarederror", early_stopping_rounds=40,
        random_state=42, n_jobs=-1,
    )
    y3 = 0.5 * train[FUTURE_RETURN_COLUMN] + 0.5 * train["excess_5d"]
    y3v = 0.5 * valid[FUTURE_RETURN_COLUMN] + 0.5 * valid["excess_5d"]
    reg3.fit(X_train, y3, eval_set=[(X_valid, y3v)], verbose=False)
    results.append(evaluate("reg blend", reg3.predict(X_test), test))

    with open("results/model_selection.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    best = max(results, key=lambda r: r["top10_edge_pct"])
    print(f"\nTerbaik berdasarkan edge top-10: {best['model']} "
          f"({best['top10_edge_pct']:+.3f}% per 5 hari)")


if __name__ == "__main__":
    main()
