"""Pipeline harian: unduh -> fitur -> (training) -> sinyal -> rekomendasi.

    python daily.py                  # unduh + fitur + rekomendasi
    python daily.py --retrain        # ikut melatih ulang model
    python daily.py --skip-download  # pakai data yang sudah ada
    python daily.py --quiet          # hanya cetak pesan siap kirim

Pesan akhir ditulis ke reports/latest_message.txt dan dicetak ke stdout,
sehingga bisa dipakai langsung oleh cron (no_agent=True).
"""

import argparse
import os
import subprocess
import sys
import time

PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "myenv", "bin", "python")
HERE = os.path.dirname(os.path.abspath(__file__))
MESSAGE_TXT = os.path.join(HERE, "reports", "latest_message.txt")
LOG_FILE = os.path.join(HERE, "logs_daily.txt")


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
    parser.add_argument("--quiet", action="store_true", help="hanya cetak pesan")
    args = parser.parse_args()

    os.chdir(HERE)

    if not args.skip_download:
        run("download_data.py", quiet=args.quiet)

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

    if args.quiet:
        # Untuk cron: stdout hanya berisi pesan yang siap kirim.
        print(message)
    else:
        print(message)
        print(f"\nPesan siap kirim ada di {MESSAGE_TXT}")


if __name__ == "__main__":
    main()
