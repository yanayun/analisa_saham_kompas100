"""Wrapper harian: jalankan daily.py hanya pada jendela WIB yang diinginkan.

    Morning : 09:00 - 11:59 WIB
    Afternoon: 13:00 - 15:00 WIB

Stdout berisi pesan laporan jika di dalam jendela, kosong jika tidak.
"""
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

WIB = timezone(timedelta(hours=7))
now_wib = datetime.now(WIB)
mins = now_wib.hour * 60 + now_wib.minute

# Market hour windows (menit sejak 00:00 WIB)
MORNING_START = 9 * 60      # 09:00
MORNING_END = 12 * 60       # 12:00
AFTER_START = 13 * 60       # 13:00
AFTER_END = 16 * 60         # 16:00

in_window = (MORNING_START <= mins < MORNING_END) or (AFTER_START <= mins < AFTER_END)
if not in_window:
    sys.exit(0)

# Weekend: pasar tutup
if now_wib.weekday() >= 5:
    sys.exit(0)

here = os.path.dirname(os.path.abspath(__file__))
python = os.path.join(here, "myenv", "bin", "python")
script = os.path.join(here, "daily.py")

result = subprocess.run(
    [python, script, "--quiet"],
    cwd=here,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

if result.returncode != 0:
    print(f"GAGAL: daily.py keluar dengan kode {result.returncode}")
    print(result.stdout)
    sys.exit(result.returncode)

out = result.stdout.strip()
if out:
    print(out)
