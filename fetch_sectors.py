"""Ambil sektor & industri tiap saham dari Yahoo Finance, simpan ke data/sectors.json.

Dijalankan sekali (atau berkala) karena hasilnya di-cache.
    python fetch_sectors.py            # semua, lewati yang sudah ada
    python fetch_sectors.py --force    # unduh ulang semua
"""

import argparse
import json
import os
import time

import yfinance as yf

from tickers import KOMPAS100, SUFFIX

CACHE_FILE = os.path.join("data", "sectors.json")
SLEEP_BETWEEN = 0.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    cache = {}
    if os.path.exists(CACHE_FILE) and not args.force:
        with open(CACHE_FILE, encoding="utf-8") as fh:
            cache = json.load(fh)

    targets = [t for t in KOMPAS100 if args.force or t not in cache]
    print(f"{len(targets)} saham perlu diambil sektornya.\n")

    ok = failed = 0
    for i, ticker in enumerate(targets, start=1):
        symbol = f"{ticker}{SUFFIX}"
        try:
            info = yf.Ticker(symbol).info
            sector = info.get("sector") or "Lainnya"
            industry = info.get("industry") or ""
            cache[ticker] = sector
            print(f"[{i}/{len(targets)}] {ticker}: {sector} | {industry}")
            ok += 1
        except Exception as exc:
            print(f"[{i}/{len(targets)}] {ticker}: GAGAL ({exc})")
            failed += 1
        time.sleep(SLEEP_BETWEEN)

    with open(CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"\nSelesai: {ok} berhasil, {failed} gagal.")
    print(f"Tersimpan: {CACHE_FILE}")


if __name__ == "__main__":
    main()
