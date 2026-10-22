# AI Saham Indonesia — Kompas100

Alat pemantauan saham: model machine learning + konfirmasi teknikal untuk
mendapatkan daftar saham Indeks **KOMPAS100** yang berpotensi naik, dikirim
otomatis **setiap pagi (07.00 WIB) dan siang (13.30 WIB)**.

> ⚠️ **Bukan rekomendasi beli.** Lihat bagian [Kinerja jujur](#kinerja-jujur)
> di bawah — strategi ini *belum* unggul dibanding beli-tahan pada uji holdout.

---

## Mulai cepat

```bash
cd /root/ai-saham-indonesia

# Jalankan seluruh pipeline sekarang (unduh -> fitur -> sinyal -> rekomendasi)
./myenv/bin/python daily.py

# Hanya lihat pesan siap kirim
./myenv/bin/python daily.py --quiet
```

Model belum ada? Tambahkan `--retrain`:

```bash
./myenv/bin/python daily.py --retrain
```

## Alur kerja

| Langkah | Script | Keluaran |
|---|---|---|
| 1. Unduh harga | `download_data.py` | `data/<KODE>.csv` (100 saham + IHSG) |
| 1b. Perbarui bar hari ini | `download_data.py --refresh-today` | bar terakhir, ±14 detik |
| 2. Feature engineering | `features.py` | `data/<KODE>_features.csv` |
| 3. Latih model | `train.py` | `models/saham_model.joblib` |
| 4. Prediksi harian | `predict.py` | `results/signals.csv` |
| 5. Skor + laporan | `recommend.py` | `reports/latest_message.txt` |
| — Backtest | `backtest.py` | `results/backtest_metrics.json` |
| — Pilih model | `model_selection.py` | `results/model_selection.json` |

Semua langkah 1–5 dijalankan oleh `daily.py`. Langkah 1 selalu memakai
`--refresh-today`, sehingga bar hari ini selalu yang terbaru.

## Skoring

Probabilitas model dikalibrasi sehingga terkompresi di sekitar *base rate*.
Karena itu skor memakai **z-score lintas semesta** pada hari yang sama:

```
skor = 50 + 15 · z_probabilitas + 20 · (teknikal_normal − 0.45)
```

* **50** — rata-rata semesta
* **≥ 72** — `BERPOTENSI BELI`
* **62–71** — `PERTIMBANGKAN`

Lapisan teknikal menilai tren (MA5/MA20/MA50), kekuatan relatif terhadap IHSG,
RSI, volume, dan volatilitas.

## Model

* **XGBoost** `binary:logistic` — target: *naik ≥ 3% dalam 5 hari*
* **26 fitur**: return multi-horizon, jarak terhadap MA, alignment tren,
  RSI 6/14, MACD, Bollinger, ATR, volatilitas, volume, gap, *rel_20d* (vs IHSG)
* Split **kronologis** 70/15/15 dengan *purge* 5 hari di sekitar batas
* `scale_pos_weight` memperbaiki peringkat, lalu **Platt scaling** memasang
  ulang kalibrasi probabilitas tanpa mengubah peringkat
* Ambang probabilitas dipilih dari *validation* saja (precision tertinggi
  dengan minimal 10% sinyal), lalu diuji satu kali di *test*

## Pengiriman terjadwal

Cron Hermes menjalankan `~/.hermes/scripts/saham_harian.sh` **tiap jam**.
Script tersebut memutuskan sendiri kapan mengirim — dihitung dalam **WIB**
— sehingga tidak terpengaruh zona waktu server (yang mengikuti DST).

* Jendela kirim: **06.45–08.30 WIB** (pagi, sebelum bursa buka) dan
  **12.00–13.30 WIB** (siang, jam istirahat bursa)
* Cron jalan tiap jam; yang masuk jendela adalah tick **07.00/08.00 WIB**
  dan **12.00/13.00 WIB**
* Tidak kirim Sabtu–Minggu
* Penghitung `.last_pagi` / `.last_siang` mencegah kirim ganda
* stdout hanya berisi pesan laporan; kosong = tidak ada kirim

> **Data saat jam istirahat:** laporan siang memakai **penutupan sesi pagi**
> (09.00–12.00 WIB), bukan penutupan hari sebelumnya. Bar harian hari itu
> yang dikembalikan Yahoo pada jam istirahat memang berisi sesi pagi —
> terlihat dari `Open`/`High`/`Low`/`Volume` yang hanya mencakup tiga bar
> jam 09.00/10.00/11.00. `download_data.py --refresh-today` mengambil bar
> tersebut untuk semua simbol sekaligus (±14 detik). Setelah bursa tutup,
> bar yang sama tanggalnya diisi ulang dengan data penuh, sehingga laporan
> keesokan paginya tidak memakai bar parsial.
> Laporan menampilkan `(sesi pagi)` di baris ke-3 bila datanya masih parsial.

Kelola jadwalnya:

```bash
hermes cron list
hermes cron run <job_id>     # coba sekarang
hermes cron pause <job_id>
```

## Kinerja jujur

Diukur pada **periode holdout** (30% tanggal terakhir, tidak pernah dipakai
untuk training):

| Metrik | Nilai |
|---|---|
| Precision sinyal | **37,1%** vs *base rate* 28,9% (lift 1,28x) |
| ROC-AUC | 0,597 |
| Backtest holdout | **42,1%** vs beli-tahan **48,2%** (alpha **−6,1%**) |
| Win rate holdout | 46,7% |

**Artinya:** model ini punya nilai information yang lemah tapi nyata
(precision naik ~28% dari acak), **tetapi belum menghasilkan keunggulan
setelah biaya transaksi.** Angka-angka di laporan harian adalah alat
*pemantauan*, bukan sistem yang terbukti menguntungkan.

Satu temuan penting lagi: korelasi antara probabilitas model dan skor teknikal
adalah **−0,72**. Model ini bersifat **mean-reversion** — ia mencari saham yang
baru terjatuh dan berpeluang *rebound*. Karena itu daftar `BERPOTENSI BELI`
sering berisi saham dengan tren rusak, dan selalu disertai peringatan.
Bagian `TREN TERKUAT` di laporan memberi sisi sebaliknya.

## Struktur repo

```
tickers.py              Daftar konstituen KOMPAS100 (berlaku 3 Agu 2026)
download_data.py        Unduh harga dari Yahoo Finance
features.py             Feature engineering (26 fitur)
train.py                Training + kalibrasi + pemilihan ambang
predict.py              Inferensi harian
recommend.py            Skoring gabungan + susun pesan Telegram
backtest.py             Simulasi portofolio event-driven
model_selection.py      Perbandingan varian model (dokumen pilihan)
analyze_thresholds.py   Laporan ambang + evaluasi akhir
sectors.py              Pemetaan sektor (cache dari Yahoo)
fetch_sectors.py        Ambil sektor ke cache
daily.py                Orkestrator pipeline harian
data/                   Harga + fitur + cache sektor
models/                 Model terlatih
results/                Prediksi, metrik, riwayat
reports/                Pesan siap kirim + penghitung kirim
```

## Catatan teknis

* Interpreter: `myenv/bin/python` (Python 3.13)
* `data/archive/` menyimpan saham di luar Kompas100 supaya tidak ikut diproses
* Model lama (`models/xgb_saham.joblib`) sudah tidak dipakai; bundle baru
  berisi `model`, `calibrator`, `features`, `threshold`, `metrics`
