"""Backtest strategi sinyal model pada periode holdout (test).

Perbaikan dari versi lama:
  * bundle model dibaca dengan benar (bukan objek mentah)
  * kolom string tidak lagi dipaksa jadi numerik (merusak kolom ticker)
  * Simulasi portofolio event-driven: maks. 10 posisi, biaya 0,25%/sisi,
    keluar tepat 5 hari setelah masuk, satu posisi per saham.

Keluaran: results/backtest_metrics.json
"""

import glob
import json
import os

import joblib
import numpy as np
import pandas as pd

import predict

DATA_DIR = "data"
RESULT_DIR = "results"
MODEL_PATH = os.path.join("models", "saham_model.joblib")
METRICS_PATH = os.path.join(RESULT_DIR, "backtest_metrics.json")

HOLD_DAYS = 5
COST_PCT = 0.0025          # biaya broker + fee per sisi
MAX_POSITIONS = 10
MIN_HISTORY = HOLD_DAYS + 2


def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan di {MODEL_PATH}. Jalankan: python train.py"
        )
    bundle = joblib.load(MODEL_PATH)
    if "model" not in bundle:
        raise ValueError("Format model lama. Jalankan ulang: python train.py")
    return bundle


def load_market(bundle):
    """Gabungkan semua saham: tanggal, ticker, Close, probabilitas."""
    features = bundle["features"]
    frames = []

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_features.csv"))):
        ticker = os.path.basename(path).replace("_features.csv", "")

        # Lewati file kosong/rusak (mis. tertulis setengah jadi saat
        # proses sebelumnya dihentikan).
        if os.path.getsize(path) < 100:
            print(f"  melewati {ticker}: file kosong.")
            continue

        try:
            df = pd.read_csv(path, index_col=0)
        except Exception as exc:
            print(f"  melewati {ticker}: {exc}")
            continue

        if df.empty:
            print(f"  melewati {ticker}: tidak ada baris.")
            continue

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.loc[~df.index.isna()].sort_index()

        missing = [c for c in features if c not in df.columns]
        if missing:
            continue

        for col in features + ["Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=features + ["Close"])
        if len(df) < MIN_HISTORY:
            continue

        df["ticker"] = ticker
        df["date"] = df.index
        frames.append(df[["date", "ticker", "Close", *features]])

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    data["probability_up_3pct"] = predict.predict_proba(bundle, data[features])
    return data.sort_values(["ticker", "date"]).reset_index(drop=True)


def simulate(data, threshold, hold_days=HOLD_DAYS,
             cost_pct=COST_PCT, max_positions=MAX_POSITIONS):
    """Simulasi portofolio event-driven.

    Kompleksitas dijaga linear terhadap jumlah tanggal dengan memakai
    kelompok per-saham yang sudah disiapkan (bukan filter ulang per hari).
    """
    capital = 1_000_000_000.0          # Rp 1 miliar
    start_capital = capital

    # Siapkan kelompok per saham sekali saja.
    groups = {
        ticker: g.sort_values("date").reset_index(drop=True)
        for ticker, g in data.groupby("ticker")
    }
    for ticker, g in groups.items():
        g["_pos"] = range(len(g))

    dates = sorted(data["date"].unique())
    positions = {}
    equity = []
    trades = []

    # Peta (ticker, tanggal) -> posisi baris untuk lookup harga.
    loc = {
        ticker: {d: i for i, d in enumerate(g["date"])}
        for ticker, g in groups.items()
    }

    signals = {}
    for date, g in data.groupby("date"):
        signals[date] = g.loc[g["probability_up_3pct"] >= threshold, "ticker"].tolist()

    for date in dates:
        # 1. Tutup posisi yang sudah mencapai horizon.
        for ticker in list(positions):
            pos = positions[ticker]
            if date >= pos["exit_date"]:
                g = groups[ticker]
                idx = loc[ticker].get(date)
                if idx is None:
                    continue
                exit_price = float(g.at[idx, "Close"])
                proceeds = pos["shares"] * exit_price * (1 - cost_pct)
                pnl = proceeds - pos["cost"]
                capital += proceeds

                trades.append({
                    "ticker": ticker,
                    "entry_date": str(pos["entry_date"])[:10],
                    "exit_date": str(date)[:10],
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "return_pct": round(pnl / pos["cost"] * 100, 2),
                    "pnl": round(pnl, 0),
                })
                del positions[ticker]

        # 2. Buka posisi baru dari sinyal hari ini.
        for ticker in sorted(signals.get(date, ())):
            if len(positions) >= max_positions:
                break
            if ticker in positions or ticker not in groups:
                continue

            g = groups[ticker]
            idx = loc[ticker].get(date)
            if idx is None or idx + hold_days >= len(g):
                continue

            entry_price = float(g.at[idx, "Close"])
            allocation = capital / max_positions
            shares = allocation / entry_price
            cost = shares * entry_price * (1 + cost_pct)

            positions[ticker] = {
                "entry_date": date,
                "entry_price": entry_price,
                "shares": shares,
                "cost": cost,
                "exit_date": g.at[idx + hold_days, "date"],
            }
            capital -= cost

        # 3. Nilai portofolio hari ini.
        value = capital
        for ticker, pos in positions.items():
            idx = loc[ticker].get(date)
            if idx is not None:
                value += pos["shares"] * float(groups[ticker].at[idx, "Close"])

        equity.append({"date": pd.Timestamp(date), "value": value})

    equity_df = pd.DataFrame(equity)
    trades_df = pd.DataFrame(trades)

    if equity_df.empty:
        raise ValueError("Tidak ada data untuk backtest.")

    equity_df["peak"] = equity_df["value"].cummax()
    equity_df["drawdown"] = equity_df["value"] / equity_df["peak"] - 1

    total_return = equity_df["value"].iloc[-1] / start_capital - 1
    years = max((equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days / 365.25, 1e-9)
    cagr = (equity_df["value"].iloc[-1] / start_capital) ** (1 / years) - 1

    if trades_df.empty:
        win_rate = avg_win = avg_loss = profit_factor = 0.0
        total_trades = 0
    else:
        wins = trades_df.loc[trades_df["return_pct"] > 0, "return_pct"]
        losses = trades_df.loc[trades_df["return_pct"] <= 0, "return_pct"]
        total_trades = int(len(trades_df))
        win_rate = float(len(wins) / len(trades_df))
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        gross_profit = float(wins.sum())
        gross_loss = abs(float(losses.sum()))
        profit_factor = gross_profit / gross_loss if gross_loss else float("inf")

    return {
        "equity": equity_df,
        "trades": trades_df,
        "metrics": {
            "start_capital": start_capital,
            "end_capital": float(equity_df["value"].iloc[-1]),
            "total_return_pct": round(float(total_return * 100), 2),
            "cagr_pct": round(float(cagr * 100), 2),
            "max_drawdown_pct": round(float(equity_df["drawdown"].min() * 100), 2),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else None,
            "threshold": threshold,
            "hold_days": hold_days,
            "cost_pct_per_side": cost_pct * 100,
            "max_positions": max_positions,
            "period": [
                str(equity_df["date"].iloc[0].date()),
                str(equity_df["date"].iloc[-1].date()),
            ],
        },
    }


def buy_and_hold(data):
    """Pembanding: beli & tahan semua saham secara merata per tanggal."""
    frames = []
    for ticker, sub in data.groupby("ticker"):
        sub = sub.sort_values("date")[["date", "Close"]].copy()
        sub["ticker"] = ticker
        sub["ret"] = sub["Close"].pct_change()
        frames.append(sub)

    if not frames:
        return 0.0

    all_rows = pd.concat(frames, ignore_index=True)
    daily = all_rows.groupby("date")["ret"].mean().dropna()
    return float(((1 + daily).prod() - 1) * 100)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    bundle = load_bundle()
    threshold = float(bundle.get("threshold", 0.5))

    data = load_market(bundle)
    if data.empty:
        raise ValueError("Tidak ada data untuk backtest.")

    result = simulate(data, threshold)
    metrics = result["metrics"]

    # Hanya periode holdout: buang 70% tanggal pertama (masa training).
    dates = np.sort(data["date"].unique())
    cut = dates[int(len(dates) * 0.70)]
    holdout = data[data["date"] >= cut].copy()
    holdout_result = simulate(holdout, threshold)
    metrics["holdout"] = holdout_result["metrics"]

    result["trades"].to_csv(
        os.path.join(RESULT_DIR, "backtest_trades.csv"), index=False
    )
    result["equity"].to_csv(
        os.path.join(RESULT_DIR, "backtest_equity.csv"), index=False
    )

    metrics["buy_and_hold_full_pct"] = round(buy_and_hold(data), 2)
    metrics["holdout_buy_and_hold_pct"] = round(buy_and_hold(holdout), 2)

    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print("=" * 56)
    print("HASIL BACKTEST (periode penuh)")
    print("=" * 56)
    print(f"Periode      : {metrics['period'][0]} - {metrics['period'][1]}")
    print(f"Total return : {metrics['total_return_pct']:.2f}%")
    print(f"CAGR         : {metrics['cagr_pct']:.2f}%")
    print(f"Max drawdown : {metrics['max_drawdown_pct']:.2f}%")
    print(f"Trade        : {metrics['total_trades']}")
    print(f"Win rate     : {metrics['win_rate_pct']:.2f}%")
    print(f"Profit factor: {metrics['profit_factor']}")
    print(f"Buy & hold   : {metrics['buy_and_hold_full_pct']:.2f}%")

    hold = metrics["holdout"]
    print("\nPeriode holdout (30% terakhir):")
    print(f"  Total return : {hold['total_return_pct']:.2f}%")
    print(f"  Win rate     : {hold['win_rate_pct']:.2f}%")
    print(f"  Trade        : {hold['total_trades']}")
    print(f"  Buy & hold   : {metrics['holdout_buy_and_hold_pct']:.2f}%")
    print(
        f"  Alpha        : "
        f"{hold['total_return_pct'] - metrics['holdout_buy_and_hold_pct']:.2f}%"
    )

    print(f"\nTersimpan: {METRICS_PATH}")


if __name__ == "__main__":
    main()
