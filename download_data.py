"""Unduh data harian saham KOMPAS100 + IHSG dari Yahoo Finance.

Cara pakai:
    python download_data.py              # unduh semua
    python download_data.py --force      # unduh ulang semua
    python download_data.py --ticker BBCA
"""

import argparse
import os
import sys
import time

import yfinance as yf

from tickers import BENCHMARK, KOMPAS100, SUFFIX

DATA_DIR = "data"
START_DATE = "2015-01-01"
MAX_RETRIES = 3


def flatten_columns(df):
    """yfinance versi baru mengembalikan kolom MultiIndex."""
    if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def csv_name(symbol):
    """Nama file konsisten dengan konvensi lama: BBCA.JK -> data/BBCA.csv."""
    name = symbol
    if name.endswith(SUFFIX):
        name = name[: -len(SUFFIX)]
    return name.lstrip("^").replace("-", "_").replace(".", "_")


def fetch(symbol, force=False):
    """Unduh satu simbol. Mengembalikan True/False/None."""
    path = os.path.join(DATA_DIR, f"{csv_name(symbol)}.csv")

    if not force and os.path.exists(path) and os.path.getsize(path) > 500:
        return False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                symbol,
                start=START_DATE,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # jaringan/rate-limit
            print(f"  percobaan {attempt} gagal: {exc}")
            time.sleep(2 * attempt)
            continue

        if df is None or df.empty:
            print(f"  tidak ada data untuk {symbol}")
            time.sleep(1)
            continue

        df = flatten_columns(df)
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        df.index.name = "Date"
        df.to_csv(path)
        print(f"  -> {path} ({len(df)} baris)")
        return True

    print(f"  GAGAL mengunduh {symbol} setelah {MAX_RETRIES} percobaan")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="unduh ulang semua")
    parser.add_argument("--ticker", help="unduh satu kode saja")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.ticker:
        symbols = [args.ticker if args.ticker.endswith(SUFFIX) else f"{args.ticker}{SUFFIX}"]
    else:
        symbols = [f"{t}{SUFFIX}" for t in KOMPAS100] + [BENCHMARK]

    print(f"Mengunduh {len(symbols)} simbol...\n")

    ok = failed = skipped = 0
    failures = []

    for i, symbol in enumerate(symbols, start=1):
        print(f"[{i}/{len(symbols)}] {symbol}")
        result = fetch(symbol, force=args.force)
        if result is True:
            ok += 1
        elif result is False:
            skipped += 1
        else:
            failed += 1
            failures.append(symbol)

    print("\n" + "=" * 50)
    print(f"Berhasil : {ok}")
    print(f"Dilewati : {skipped} (sudah ada)")
    print(f"Gagal    : {failed}")
    if failures:
        print("Simbol gagal: " + ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
