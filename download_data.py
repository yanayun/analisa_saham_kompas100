"""Unduh data harian saham KOMPAS100 + IHSG dari Yahoo Finance.

Mode:
    python download_data.py                  # unduh simbol yang belum ada
    python download_data.py --force          # unduh ulang semua
    python download_data.py --ticker BBCA    # satu kode saja
    python download_data.py --refresh-today  # perbarui bar HARI INI saja

--refresh-today penting saat jam istirahat bursa (12.00-13.30 WIB): bar harian
hari itu yang dikembalikan Yahoo berisi penutupan SESI PAGI (09.00-12.00),
sehingga laporan siang memakai harga penutupan yang benar-benar terjadi,
bukan penutupan hari sebelumnya. Semua simbol diunduh dalam SATU permintaan
agar cepat (paralel di dalam yfinance).
"""

import argparse
import os
import sys
import time

import pandas as pd
import yfinance as yf

from tickers import BENCHMARK, KOMPAS100, SUFFIX

DATA_DIR = "data"
START_DATE = "2015-01-01"
MAX_RETRIES = 3
OHLCV = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


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


def symbols_to_fetch(ticker=None):
    if ticker:
        return [ticker if ticker.endswith(SUFFIX) else f"{ticker}{SUFFIX}"]
    return [f"{t}{SUFFIX}" for t in KOMPAS100] + [BENCHMARK]


def _to_date_index(df):
    """Samakan indeks menjadi tanggal naif (tanpa zona waktu)."""
    idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Jakarta").normalize().tz_localize(None)
    else:
        idx = idx.normalize()
    df = df.copy()
    df.index = idx
    df = df.loc[~df.index.isna()]
    return df


def _pick_ohlcv(df):
    return df[OHLCV] if all(c in df.columns for c in OHLCV) else df


def _save_history(df, symbol):
    path = os.path.join(DATA_DIR, f"{csv_name(symbol)}.csv")
    _pick_ohlcv(df).index.name = "Date"
    _pick_ohlcv(df).to_csv(path)
    return path


# ---------------------------------------------------------------------
# Unduhan penuh (riwayat)
# ---------------------------------------------------------------------

def fetch(symbol, force=False):
    """Unduh seluruh riwayat satu simbol. True/False/None = ada baru/lewat/gagal."""
    path = os.path.join(DATA_DIR, f"{csv_name(symbol)}.csv")

    if not force and os.path.exists(path) and os.path.getsize(path) > 500:
        return False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                symbol, start=START_DATE, auto_adjust=False,
                progress=False, threads=False,
            )
        except Exception as exc:  # jaringan / rate-limit
            print(f"  percobaan {attempt} gagal: {exc}")
            time.sleep(2 * attempt)
            continue

        if df is None or df.empty:
            time.sleep(1)
            continue

        df = flatten_columns(df)
        df = _to_date_index(df)
        path = _save_history(df, symbol)
        print(f"  -> {path} ({len(df)} baris)")
        return True

    print(f"  GAGAL mengunduh {symbol} setelah {MAX_RETRIES} percobaan")
    return None


# ---------------------------------------------------------------------
# Perbarui bar hari ini (dipakai saat jam istirahat bursa)
# ---------------------------------------------------------------------

def fetch_last_bars(symbol):
    """Unduh beberapa bar harian terakhir satu simbol. None bila gagal."""
    for _ in range(MAX_RETRIES):
        try:
            df = yf.download(
                symbol, period="5d", interval="1d",
                auto_adjust=False, progress=False, threads=False,
            )
        except Exception:
            time.sleep(1)
            continue
        if df is None or df.empty:
            time.sleep(1)
            continue
        df = flatten_columns(df)
        return _to_date_index(df)
    return None


def fetch_last_bars_batch(symbols):
    """Unduh bar terakhir SEMUA simbol sekaligus.

    Mengembalikan {symbol: DataFrame|None}. Satu permintaan, paralel.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                list(symbols), period="5d", interval="1d",
                auto_adjust=False, progress=False, threads=True,
                group_by="ticker",
            )
        except Exception as exc:
            print(f"  percobaan {attempt} gagal: {exc}")
            time.sleep(2 * attempt)
            continue

        if df is None or df.empty:
            time.sleep(1)
            continue

        multi = hasattr(df.columns, "levels") and df.columns.nlevels > 1
        out = {}
        for symbol in symbols:
            try:
                sub = flatten_columns(df[symbol] if multi else df)
                sub = _to_date_index(sub)
                out[symbol] = sub if not sub.empty else None
            except Exception:
                out[symbol] = None
        return out

    return {}


def apply_latest_bar(symbol, sub):
    """Tambahkan bar terbaru ke file riwayat simbol.

    Mengembalikan "updated" | "unchanged" | "failed".
    """
    if sub is None or sub.empty:
        return "failed"

    path = os.path.join(DATA_DIR, f"{csv_name(symbol)}.csv")

    if not os.path.exists(path) or os.path.getsize(path) < 500:
        _save_history(sub, symbol)
        print(f"  -> {csv_name(symbol)}.csv dibuat ({len(sub)} baris)")
        return "updated"

    existing = pd.read_csv(path, index_col=0)
    existing.index = pd.to_datetime(existing.index, errors="coerce")
    existing = existing.loc[~existing.index.isna()].sort_index()

    latest = sub.iloc[[-1]].copy()
    last_date = latest.index[0]

    # Bar hari ini belum tersedia (mis. sebelum pukul 09.00 WIB).
    if last_date < existing.index[-1]:
        return "unchanged"

    # Tanggal sama: ganti hanya bila isinya benar-benar berubah. Ini penting
    # karena bar hari ini mula-mula hanya berisi sesi pagi, lalu jadi penuh
    # setelah bursa tutup -- keduanya bertanggal sama.
    if last_date == existing.index[-1]:
        old, new = existing.iloc[-1], latest.iloc[-1]
        same = True
        for col in ("Close", "Volume"):
            try:
                if abs(float(old[col]) - float(new[col])) > 1e-6:
                    same = False
                    break
            except (KeyError, TypeError, ValueError):
                same = False
                break
        if same:
            return "unchanged"

    combined = pd.concat([existing, _pick_ohlcv(latest)])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.index.name = "Date"
    combined.to_csv(path)

    print(f"  -> {csv_name(symbol)}.csv diperbarui ke {last_date.date()} "
          f"(total {len(combined)} baris)")
    return "updated"


# ---------------------------------------------------------------------
# Titik masuk
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="unduh ulang semua")
    parser.add_argument("--ticker", help="unduh satu kode saja")
    parser.add_argument("--refresh-today", action="store_true",
                        help="perbarui bar hari ini saja (jam istirahat bursa)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    symbols = symbols_to_fetch(args.ticker)

    if args.refresh_today:
        print(f"Memperbarui bar hari ini untuk {len(symbols)} simbol...\n")
        started = time.time()

        if len(symbols) == 1:
            fetched = {symbols[0]: fetch_last_bars(symbols[0])}
        else:
            fetched = fetch_last_bars_batch(symbols)

        updated = unchanged = failed = 0
        failures = []

        for symbol in symbols:
            result = apply_latest_bar(symbol, fetched.get(symbol))
            if result == "updated":
                updated += 1
            elif result == "unchanged":
                unchanged += 1
            else:
                failed += 1
                failures.append(symbol)

        print("\n" + "=" * 50)
        print(f"Diperbarui        : {updated}")
        print(f"Tidak ada bar baru: {unchanged}")
        print(f"Gagal             : {failed}")
        print(f"Waktu             : {time.time() - started:.1f}s")
        if failures:
            print("Simbol gagal: " + ", ".join(failures))
            sys.exit(1)
        return

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
