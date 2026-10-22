"""Analisa AI: saham KOMPAS100 yang berpotensi untuk dibeli.

Dua lapisan:
  1. MODEL  -- probabilitas XGBoost naik >= 3% dalam 5 hari
  2. KONFIRMASI TEKNIKAL -- skor 0..1 dari tren, kekuatan relatif,
     RSI, volume, dan volatilitas

Probabilitas model dikalibrasi sehingga nilainya terkompresi di sekitar
base rate (28%). Karena itu skor akhir memakai Z-SCORE lintas semesta
pada hari yang sama, bukan probabilitas mentah:

    skor = 50 + 15 * z_probabilitas + 12 * (teknikal_normal - 0.45)

  * 50  = rata-rata semesta
  * 70  = kira-kira 1,3 simpangan baku di atas rata-rata (daftar BELI)
  * 60  = batas PERTIMBANGKAN

Keluaran:
    results/recommendations.csv
    results/recommendations.json
    reports/latest_message.txt   <- siap kirim ke Telegram
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import predict
import sectors
from features import FUTURE_RETURN_COLUMN

RESULT_DIR = "results"
REPORT_DIR = "reports"
HISTORY_DIR = os.path.join(RESULT_DIR, "history")

REC_CSV = os.path.join(RESULT_DIR, "recommendations.csv")
REC_JSON = os.path.join(RESULT_DIR, "recommendations.json")
MESSAGE_TXT = os.path.join(REPORT_DIR, "latest_message.txt")

SCORE_BASE = 50.0
SCORE_MODEL_WEIGHT = 15.0
SCORE_TECH_WEIGHT = 20.0
SCORE_TECH_CENTER = 0.45
MAX_TECH = 0.80

BUY_SCORE = 72.0
WATCH_SCORE = 62.0

# Parameter manajemen risiko berbasis ATR harian.
STOP_ATR_MULT = 1.5
TARGET_ATR_MULT = 2.5


# ---------------------------------------------------------------------
# KONFIRMASI TEKNIKAL
# ---------------------------------------------------------------------

def technical_score(row):
    """Mengembalikan (skor 0..1, alasan, peringatan)."""
    score = 0.0
    reasons = []
    warnings = []

    # 1. Tren menengah -- harga di atas MA20
    if row["close_ma20"] > 0:
        score += 0.15
        reasons.append(f"di atas MA20 (+{row['close_ma20']:.1%})")
    else:
        warnings.append(f"di bawah MA20 ({row['close_ma20']:.1%})")

    # 2. Tren pendek di atas tren menengah
    if row["ma5_ma20"] > 0:
        score += 0.10
        reasons.append("MA5 di atas MA20")
    else:
        warnings.append("MA5 di bawah MA20")

    # 3. Struktur tren menengah
    if row["ma20_ma50"] > 0:
        score += 0.10
        reasons.append("MA20 di atas MA50")
    else:
        warnings.append("MA20 di bawah MA50")

    # 4. Kekuatan relatif terhadap IHSG
    if row["rel_20d"] > 0:
        score += 0.10
        reasons.append(f"lebih kuat dari IHSG ({row['rel_20d']:+.1%})")
    else:
        warnings.append(f"lemah vs IHSG ({row['rel_20d']:+.1%})")

    # 5. RSI dalam rentang sehat
    rsi = row["rsi_14"]
    if 40 <= rsi <= 72:
        score += 0.10
        reasons.append(f"RSI {rsi:.0f} sehat")
    elif rsi > 72:
        warnings.append(f"RSI {rsi:.0f} mulai overbought")
    else:
        warnings.append(f"RSI {rsi:.0f} masih lemah")

    # 6. Tidak terlalu jauh di bawah puncak 60 hari
    if row["dist_high_60d"] > -0.20:
        score += 0.10
    else:
        warnings.append(
            f"masih {abs(row['dist_high_60d']):.0%} di bawah puncak 60 hari"
        )

    # 7. Likuiditas masih normal
    volume_ratio = row["volume_ratio"]
    if 0.6 <= volume_ratio <= 3.0:
        score += 0.05
    elif volume_ratio < 0.6:
        warnings.append(f"volume tipis ({volume_ratio:.2f}x)")

    # 8. Volatilitas terkendali
    volatility = row["volatility_20"]
    if volatility <= 0.035:
        score += 0.10
    elif volatility > 0.06:
        warnings.append(f"volatilitas tinggi ({volatility:.1%})")

    return score, reasons, warnings


def risk_parameters(row):
    """Perkiraan level stop & target berdasarkan ATR harian."""
    close = float(row["Close"])
    atr_pct = float(row["atr_pct"])

    if not np.isfinite(atr_pct) or atr_pct <= 0:
        return None, None, None

    stop = close * (1 - STOP_ATR_MULT * atr_pct)
    target = close * (1 + TARGET_ATR_MULT * atr_pct)
    reward_risk = (target - close) / (close - stop) if close > stop else 0.0
    return stop, target, reward_risk


def classify(score, probability, threshold):
    if score >= BUY_SCORE:
        return "BELI"
    if score >= WATCH_SCORE:
        return "PERTIMBANGKAN"
    return "TAHAN"


def signal_type(prob_z, tech_norm):
    """Memberi nama jenis sinyal agar tidak salah tafsir."""
    if prob_z >= 0.5 and tech_norm >= 0.5:
        return "model & tren sepakat"
    if prob_z >= 0.5:
        return "mean-reversion (model)"
    if tech_norm >= 0.6:
        return "tren kuat, model netral"
    if prob_z >= 0:
        return "model agak positif"
    return "campuran"


# ---------------------------------------------------------------------
# PENYUSUNAN
# ---------------------------------------------------------------------

def build():
    bundle = predict.load_bundle()
    threshold = float(bundle.get("threshold", 0.5))

    data = predict.predict_all(bundle)
    if data.empty:
        raise ValueError("Tidak ada data untuk dianalisa.")

    # Z-score probabilitas terhadap semesta pada hari yang sama.
    prob = data["probability_up_3pct"].astype(float)
    std = prob.std(ddof=0)
    data["prob_z"] = (prob - prob.mean()) / (std if std > 0 else 1.0)
    data["prob_rank_pct"] = prob.rank(pct=True) * 100

    records = []
    for _, row in data.iterrows():
        tech, reasons, warnings = technical_score(row)

        tech_norm = tech / MAX_TECH
        score = (
            SCORE_BASE
            + SCORE_MODEL_WEIGHT * row["prob_z"]
            + SCORE_TECH_WEIGHT * (tech_norm - SCORE_TECH_CENTER)
        )
        stop, target, reward_risk = risk_parameters(row)

        records.append({
            "date": row["date"],
            "ticker": row["ticker"],
            "sector": sectors.sector_of(row["ticker"]),
            "close": float(row["Close"]),
            "probability_up_3pct": float(row["probability_up_3pct"]),
            "prob_rank_pct": round(float(row["prob_rank_pct"]), 1),
            "prob_z": round(float(row["prob_z"]), 3),
            "tech_score": round(tech_norm, 4),
            "score": round(float(score), 1),
            "status": classify(float(score), float(row["probability_up_3pct"]), threshold),
            "signal_type": signal_type(float(row["prob_z"]), tech_norm),
            "rsi_14": round(float(row["rsi_14"]), 1),
            "return_1d_pct": round(float(row["return_1d"]) * 100, 2),
            "rel_20d_pct": round(float(row["rel_20d"]) * 100, 2),
            "volatility_20_pct": round(float(row["volatility_20"]) * 100, 2),
            "stop": round(stop, 0) if stop else None,
            "target": round(target, 0) if target else None,
            "reward_risk": round(reward_risk, 2) if reward_risk else None,
            "reasons": reasons,
            "warnings": warnings,
        })

    rec = pd.DataFrame(records).sort_values(
        ["score", "probability_up_3pct"], ascending=False
    ).reset_index(drop=True)
    rec["rank"] = rec.index + 1
    return rec, bundle, threshold


def _snapshot(rec):
    return {
        r["ticker"]: {
            "score": r["score"],
            "status": r["status"],
            "close": r["close"],
        }
        for _, r in rec.iterrows()
    }


def save_history(rec):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    path = os.path.join(HISTORY_DIR, f"{stamp}.json")
    previous = {}

    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            previous = json.load(fh)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_snapshot(rec), fh, indent=2, ensure_ascii=False, sort_keys=True)

    return previous


def model_stats():
    """Statistik uji holdout dari metrics.json + backtest_metrics.json."""
    stats = {}
    metrics_path = os.path.join(RESULT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as fh:
            m = json.load(fh)
        stats.update({
            "precision": m.get("precision"),
            "base_rate": m.get("positive_rate_test"),
            "roc_auc": m.get("roc_auc"),
        })

    bt_path = os.path.join(RESULT_DIR, "backtest_metrics.json")
    if os.path.exists(bt_path):
        with open(bt_path, encoding="utf-8") as fh:
            b = json.load(fh)
        hold = b.get("holdout") or {}
        stats["holdout_return"] = hold.get("total_return_pct")
        stats["holdout_bh"] = b.get("holdout_buy_and_hold_pct")
        stats["holdout_alpha"] = (
            round(stats["holdout_return"] - stats["holdout_bh"], 2)
            if stats["holdout_return"] is not None and stats["holdout_bh"] is not None
            else None
        )
    return stats


def format_message(rec, bundle, threshold, previous=None):
    os.makedirs(REPORT_DIR, exist_ok=True)

    now = pd.Timestamp.now()
    latest_date = rec["date"].max().strftime("%d %b %Y")
    stats = model_stats()

    buys = rec[rec["status"] == "BELI"]
    watch = rec[rec["status"] == "PERTIMBANGKAN"]

    lines = []
    lines.append("📊 REKOMENDASI SAHAM KOMPAS100")
    lines.append(
        f"{now.strftime('%A, %d %B %Y')} · {now.strftime('%H.%M')} WIB"
    )
    lines.append(f"Model: XGBoost terkalibrasi · ambang {threshold:.0%} · data {latest_date}")
    lines.append(f"Saham dianalisa: {len(rec)}")

    if stats.get("precision"):
        lift = stats["precision"] / stats["base_rate"] if stats.get("base_rate") else 0
        lines.append(
            f"Uji holdout: precision {stats['precision']:.1%} vs dasar "
            f"{stats['base_rate']:.1%} (lift {lift:.2f}x)"
            + (f" · AUC {stats['roc_auc']:.3f}" if stats.get("roc_auc") else "")
        )
    if stats.get("holdout_alpha") is not None:
        lines.append(
            f"Backtest holdout: {stats['holdout_return']:.1f}% vs "
            f"beli-tahan {stats['holdout_bh']:.1f}% "
            f"(alpha {stats['holdout_alpha']:+.1f}%)"
        )
    lines.append("")

    if buys.empty:
        lines.append("🟡 Tidak ada saham yang mencapai skor BELI hari ini.")
        lines.append("Menunggu konfirmasi lebih kuat.")
    else:
        lines.append(f"🟢 BERPOTENSI BELI ({len(buys)})")
        lines.append("")

        for _, row in buys.head(10).iterrows():
            lines.append(f"{row['rank']}. {row['ticker']} — Rp {row['close']:,.0f}")
            lines.append(
                f"   Skor {row['score']:.0f}/100 · prob {row['probability_up_3pct']:.0%}"
                f" · {row['sector']} · {row['signal_type']}"
            )
            if row["reasons"]:
                lines.append(f"   ✅ {'; '.join(row['reasons'][:3])}")
            if row["warnings"]:
                lines.append(f"   ⚠️ {'; '.join(row['warnings'][:2])}")
            if row["stop"] and row["target"]:
                lines.append(
                    f"   🎯 target Rp {row['target']:,.0f} · "
                    f"🛑 stop Rp {row['stop']:,.0f} · R:R {row['reward_risk']:.1f}"
                )
            lines.append("")

    # Sisi sebaliknya: saham dengan struktur tren terbaik.
    strong = rec[rec["tech_score"] >= 0.6].sort_values("tech_score", ascending=False)
    if not strong.empty:
        lines.append(
            f"📈 TREN TERKUAT (model belum konfirmasi) ({len(strong)})"
        )
        lines.append(
            "  " + ", ".join(
                f"{r['ticker']} (prob {r['probability_up_3pct']:.0%})"
                for _, r in strong.head(8).iterrows()
            )
        )
        lines.append("")

    if not watch.empty:
        lines.append(f"🟠 PERTIMBANGKAN ({len(watch)})")
        lines.append(
            "  " + ", ".join(
                f"{r['ticker']} ({r['score']:.0f})"
                for _, r in watch.head(8).iterrows()
            )
        )
        lines.append("")

    if previous:
        prev_buys = {t for t, v in previous.items() if v.get("status") == "BELI"}
        now_buys = set(buys["ticker"])
        entered = sorted(now_buys - prev_buys)
        exited = sorted(prev_buys - now_buys)

        if entered:
            lines.append(f"➕ Masuk daftar: {', '.join(entered)}")
        if exited:
            lines.append(f"➖ Keluar daftar: {', '.join(exited)}")
        if entered or exited:
            lines.append("")

    lines.append("—")
    lines.append(
        "Catatan penting: model ini bersifat mean-reversion (mencari saham "
        "terjatuh yang berpeluang rebound), sehingga sering bertentangan "
        "dengan struktur tren. Pada uji holdout, strategi ini BELUM unggul "
        "dibanding beli-tahan -- Angka di atas adalah alat pemantauan, "
        "bukan rekomendasi transaksi. Lakukan riset sendiri sebelum membeli."
    )

    message = "\n".join(lines)
    with open(MESSAGE_TXT, "w", encoding="utf-8") as fh:
        fh.write(message)

    return message


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    rec, bundle, threshold = build()
    previous = save_history(rec)

    columns = [
        "rank", "date", "ticker", "sector", "close", "score",
        "probability_up_3pct", "prob_rank_pct", "prob_z", "tech_score", "status",
        "rsi_14", "return_1d_pct", "rel_20d_pct", "volatility_20_pct",
        "stop", "target", "reward_risk", "reasons", "warnings",
    ]
    rec[columns].to_csv(REC_CSV, index=False)

    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_threshold": threshold,
        "scoring": {
            "formula": "50 + 15*z_prob + 12*(teknikal_norm - 0.45)",
            "buy_score": BUY_SCORE,
            "watch_score": WATCH_SCORE,
        },
        "n_stocks": int(len(rec)),
        "n_buy": int((rec["status"] == "BELI").sum()),
        "n_watch": int((rec["status"] == "PERTIMBANGKAN").sum()),
        "recommendations": rec.to_dict("records"),
    }
    with open(REC_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

    message = format_message(rec, bundle, threshold, previous)
    print(message)
    print(f"\nTersimpan: {REC_CSV}")
    print(f"Siap kirim: {MESSAGE_TXT}")


if __name__ == "__main__":
    main()
