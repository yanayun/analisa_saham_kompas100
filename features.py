"""Feature engineering untuk saham KOMPAS100.

Menghasilkan satu file <KODE>_features.csv per saham di folder data/.

Fitur dibagi menjadi:
  * return   : momentum berbagai horizon
  * moving average & trend alignment
  * osilator : RSI, MACD
  * volatilitas : Bollinger, ATR, deviasi standar return
  * volume & rentang harga
  * rel_20d  : kekuatan relatif terhadap IHSG (alpha kasar)

Target: harga penutupan naik >= 3% dalam 5 hari ke depan.
"""

import glob
import os

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands

DATA_DIR = "data"
BENCHMARK_FILE = os.path.join(DATA_DIR, "JKSE.csv")

# Minimal baris agar MA200 + future return tersedia.
MIN_BARS = 210

FEATURE_COLUMNS = [
    "return_1d", "return_3d", "return_5d",
    "return_10d", "return_20d", "return_60d",
    "close_ma5", "close_ma10", "close_ma20",
    "close_ma50", "close_ma200",
    "ma5_ma20", "ma20_ma50",
    "rsi_14", "rsi_6",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position",
    "atr_pct", "volatility_20", "volatility_60",
    "volume_ratio", "daily_range", "gap",
    "dist_high_60d", "rel_20d",
]

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

TARGET_COLUMN = "target"
FUTURE_RETURN_COLUMN = "future_return_5d"
FUTURE_RETURN_HORIZON = 5
TARGET_THRESHOLD = 0.03


def load_price_csv(filename):
    """Baca CSV harga dan pastikan kolom numerik."""
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for col in PRICE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Kolom {col} tidak ada di {filename}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def load_benchmark():
    """Return IHSG harian, dipakai untuk fitur kekuatan relatif."""
    if not os.path.exists(BENCHMARK_FILE):
        return None

    bench = pd.read_csv(BENCHMARK_FILE, index_col=0)
    bench.index = pd.to_datetime(bench.index, errors="coerce")
    bench = bench.loc[~bench.index.isna()].sort_index()
    close = pd.to_numeric(bench["Close"], errors="coerce")
    return close.dropna()


def create_features(df, benchmark=None):
    """Menghitung seluruh fitur teknikal untuk satu saham."""
    df = df.copy()
    close = df["Close"]

    # ---- return berbagai horizon -------------------------------------
    for period in (1, 3, 5, 10, 20, 60):
        df[f"return_{period}d"] = close.pct_change(period)

    # ---- jarak harga terhadap moving average -------------------------
    for period in (5, 10, 20, 50, 200):
        ma = close.rolling(period).mean()
        df[f"close_ma{period}"] = close / ma - 1

    # ---- alignment tren (slope kasar) --------------------------------
    df["ma5_ma20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    df["ma20_ma50"] = close.rolling(20).mean() / close.rolling(50).mean() - 1

    # ---- RSI ----------------------------------------------------------
    df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
    df["rsi_6"] = RSIIndicator(close=close, window=6).rsi()

    # ---- MACD ---------------------------------------------------------
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ---- Bollinger Bands ---------------------------------------------
    bb = BollingerBands(close=close, window=20, window_dev=2)
    upper, lower, mid = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    df["bb_width"] = (upper - lower) / mid
    df["bb_position"] = (close - lower) / (upper - lower)

    # ---- volatilitas --------------------------------------------------
    atr = AverageTrueRange(
        high=df["High"], low=df["Low"], close=close, window=14
    )
    df["atr_pct"] = atr.average_true_range() / close
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df["volatility_60"] = df["return_1d"].rolling(60).std()

    # ---- volume -------------------------------------------------------
    volume_ma = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / volume_ma

    # ---- rentang harga & gap -----------------------------------------
    df["daily_range"] = (df["High"] - df["Low"]) / close
    df["gap"] = df["Open"] / close.shift(1) - 1

    # ---- jarak dari titik tertinggi 60 hari --------------------------
    df["dist_high_60d"] = close / close.rolling(60).max() - 1

    # ---- kekuatan relatif terhadap IHSG -------------------------------
    df["rel_20d"] = np.nan
    if benchmark is not None:
        # Data IHSG sering tertinggal 1 hari dari saham. Isi tanggal yang
        # belum ada dengan nilai terakhir, supaya baris saham tidak terbuang.
        bench_close = benchmark.reindex(df.index).ffill()
        bench_ret = bench_close.pct_change(20)
        df["rel_20d"] = df["return_20d"] - bench_ret

    # ---- target -------------------------------------------------------
    df[FUTURE_RETURN_COLUMN] = close.shift(-FUTURE_RETURN_HORIZON) / close - 1
    df[TARGET_COLUMN] = (df[FUTURE_RETURN_COLUMN] >= TARGET_THRESHOLD).astype("float")

    return df


def process_stock(filename, benchmark=None):
    ticker = os.path.basename(filename).replace(".csv", "")

    if ticker in ("JKSE",) or ticker.endswith("_features"):
        return None

    df = load_price_csv(filename)

    if len(df) < MIN_BARS:
        print(f"Melewati {ticker}: hanya {len(df)} baris (< {MIN_BARS}).")
        return None

    df = create_features(df, benchmark=benchmark)

    # Fitur harus lengkap. Target BOLEH kosong: baris terakhir 5 hari
    # belum punya data masa depan, tetapi tetap dipakai untuk prediksi.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS).copy()

    # train.py yang membuang baris tanpa target, bukan di sini.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("float")

    if len(df) < 120:
        print(f"Melewati {ticker}: sisa {len(df)} baris setelah dropna.")
        return None

    output = os.path.join(DATA_DIR, f"{ticker}_features.csv")
    df.to_csv(output)
    print(f"{ticker}: {len(df):,} baris -> {output}")
    return output


def main():
    from tickers import BENCHMARK, KOMPAS100

    # Hanya konstituen KOMPAS100 (+ IHSG) yang diproses.
    allowed = set(KOMPAS100) | {BENCHMARK.lstrip("^")}

    files = sorted(
        f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if "_features" not in f and os.path.basename(f)[:-4] in allowed
    )

    if not files:
        raise SystemExit(f"Tidak ada CSV harga di {DATA_DIR}/. Jalankan download_data.py.")

    benchmark = load_benchmark()
    if benchmark is None:
        print("Peringatan: IHSSE.csv tidak ada, fitur rel_20d diisi NaN.")
    else:
        print(f"IHSG: {len(benchmark):,} baris ({benchmark.index.min().date()} - "
              f"{benchmark.index.max().date()})")

    done = skipped = 0
    for filename in files:
        if process_stock(filename, benchmark=benchmark):
            done += 1
        else:
            skipped += 1

    print(f"\nSelesai: {done} saham diproses, {skipped} dilewati.")


if __name__ == "__main__":
    main()
