"""Pipeline harian: unduh -> fitur -> (training) -> sinyal -> rekomendasi.

    python daily.py                  # unduh + fitur + rekomendasi
    python daily.py --retrain        # ikut melatih ulang model
    python daily.py --skip-download  # pakai data yang sudah ada
    python daily.py --refresh-today  # paksa perbarui bar hari ini
    python daily.py --quiet          # hanya cetak pesan siap kirim

Saat pasar buka atau istirahat (09.05-15.20 WIB), bar hari ini diperbarui.
Bar harian hari itu berisi penutupan SESI PAGI saat jam istirahat, sehingga
laporan siang memakai harga yang benar-benar terjadi.

Pesan akhir ditulis ke reports/latest_message.txt dan dicetak ke stdout,
sehingga bisa dipakai langsung oleh cron (no_agent=True).
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "myenv", "bin", "python")
HERE = os.path.dirname(os.path.abspath(__file__))
MESSAGE_TXT = os.path.join(HERE, "reports", "latest_message.txt")
LOG_FILE = os.path.join(HERE, "logs_daily.txt")


def wib_now():
    """Waktu sekarang di WIB (UTC+7). Server ini UTC+8, jadi jangan pakai now()."""
    return datetime.now(timezone.utc) + timedelta(hours=7)


def run(script, *args, quiet=False):
    cmd = [PYTHON, os.path.join(HERE, script), *args]
    started = time.time()

    if quiet:
        # Output disimpan ke log; stdout hanya untuk pesan final.
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    else:
        print(f"\n$ {' '.join(os.path.basename(c) for c in cmd)}\n" + "-" * 50)
        result = subprocess.run(cmd, cwd=HERE)

    elapsed = time.time() - started

    if not quiet:
        print("-" * 50 + f" selesai dalam {elapsed:.1f}s")

    if result.returncode != 0:
        print(f"\nGAGAL: {script} keluar dengan kode {result.returncode}")
        print(f"Lihat {LOG_FILE}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="latih ulang model")
    parser.add_argument("--skip-download", action="store_true", help="lewati unduhan")
    parser.add_argument("--refresh-today", action="store_true",
                        help="paksa perbarui bar hari ini")
    parser.add_argument("--quiet", action="store_true", help="hanya cetak pesan")
    args = parser.parse_args()

    os.chdir(HERE)

    if not args.skip_download:
        # --refresh-today mengambil bar terakhir SEMUA simbol sekaligus
        # (~14 detik) dan menangani dua hal: bar hari ini yang belum ada,
        # serta bar hari ini yang masih parsial (sesi pagi) lalu jadi penuh
        # setelah bursa tutup.
        run("download_data.py", "--refresh-today", quiet=args.quiet)

    run("features.py", quiet=args.quiet)

    if args.retrain or not os.path.exists(os.path.join(HERE, "models", "saham_model.joblib")):
        run("train.py", quiet=args.quiet)
    else:
        if not args.quiet:
            print("\nModel sudah ada, training dilewati (pakai --retrain untuk melatih ulang).")

    run("predict.py", quiet=args.quiet)
    run("recommend.py", quiet=args.quiet)

    with open(MESSAGE_TXT, encoding="utf-8") as fh:
        message = fh.read()

    # Untuk cron: stdout hanya berisi pesan yang siap kirim.
    print(message)
    if not args.quiet:
        print(f"\nPesan siap kirim ada di {MESSAGE_TXT}")


if __name__ == "__main__":
    main()
