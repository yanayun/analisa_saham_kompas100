"""Pemetaan kode saham -> sektor.

Prioritas:
  1. cache data/sectors.json (dari fetch_sectors.py, sumber Yahoo Finance)
  2. SECTOR_FALLBACK di bawah ini
  3. "Lainnya"

Jangan diimpor oleh train.py -- murni untuk pelaporan.
"""

import json
import os

CACHE_FILE = os.path.join("data", "sectors.json")

SECTOR_FALLBACK = {
    # Perbankan & keuangan
    "BBCA": "Perbankan", "BBNI": "Perbankan", "BBRI": "Perbankan",
    "BBTN": "Perbankan", "BBYB": "Perbankan", "BMRI": "Perbankan",
    "BRIS": "Perbankan Syariah", "BKSL": "Perbankan", "BFIN": "Pembiayaan",
    "PNLF": "Pembiayaan", "ADMF": "Pembiayaan",
    # Energi & pertambangan
    "AADI": "Batu Bara", "ADMR": "Batu Bara", "ADRO": "Batu Bara",
    "BRPT": "Batu Bara", "BRMS": "Batu Bara", "BUMI": "Batu Bara",
    "PTBA": "Batu Bara", "ITMG": "Batu Bara", "INDY": "Batu Bara",
    "TOBA": "Batu Bara", "PSAB": "Batu Bara",
    "ANTM": "Logam & Mineral", "INCO": "Logam & Mineral",
    "MDKA": "Logam & Mineral", "AMMN": "Logam & Mineral",
    "TINS": "Logam & Mineral", "HRUM": "Logam & Mineral",
    "ENRG": "Minyak & Gas", "ESSA": "Minyak & Gas",
    "DEWA": "Minyak & Gas", "RMKE": "Minyak & Gas",
    "PGAS": "Gas Alam", "PGEO": "Energi Terbarukan",
    "UNTR": "Peralatan Berat",
    # Telekomunikasi & infrastruktur
    "TLKM": "Telekomunikasi", "EXCL": "Telekomunikasi",
    "ISAT": "Telekomunikasi", "FREN": "Telekomunikasi",
    "TOWR": "Infrastruktur Telco", "TBIG": "Infrastruktur Telco",
    "MTEL": "Infrastruktur Telco", "LINK": "Infrastruktur Telco",
    "JSMR": "Infrastruktur", "CMNP": "Infrastruktur",
    "BIPI": "Infrastruktur", "WIRG": "Infrastruktur",
    # Otomotif & komponennya
    "ASII": "Otomotif", "AUTO": "Otomotif", "IMAS": "Otomotif",
    # Properti & konstruksi
    "SMRA": "Properti", "BSDE": "Properti", "CTRA": "Properti",
    "LPKR": "Properti", "PWON": "Properti", "DMAS": "Properti",
    "MAPI": "Properti", "ARCI": "Properti", "DILD": "Properti",
    "PTPP": "Konstruksi", "WIKA": "Konstruksi", "WTON": "Konstruksi",
    "SSIA": "Konstruksi", "ADHI": "Konstruksi",
    # Konsumsi & ritel
    "UNVR": "Konsumsi", "ICBP": "Konsumsi", "INDF": "Konsumsi",
    "CPIN": "Konsumsi", "JPFA": "Konsumsi", "MYOR": "Konsumsi",
    "SIDO": "Konsumsi", "SILO": "Kesehatan", "KLBF": "Kesehatan",
    "MIKA": "Kesehatan", "HEAL": "Kesehatan", "PRDA": "Kesehatan",
    "ROTI": "Konsumsi", "LSIP": "Perkebunan", "SGER": "Perkebunan",
    "TAPG": "Perkebunan", "MINA": "Pariwisata",
    "AMRT": "Ritel", "ACES": "Ritel", "MAPA": "Ritel",
    "PANI": "Ritel", "MAPI ": "Properti",
    # Teknologi & media
    "GOTO": "Teknologi", "BUKA": "Teknologi", "COIN": "Teknologi",
    "INET": "Teknologi", "WIFI": "Teknologi", "DNET": "Teknologi",
    "EMTK": "Media", "MNCN": "Media", "SCMA": "Media",
    # Industri & kimia
    "AKRA": "Kimia", "INKP": "Pulp & Kertas", "TKIM": "Pulp & Kertas",
    "SIMP": "Kehutanan", "ARII": "Kimia", "DPNS": "Kimia",
    "ERTX": "Tekstil", "HIKM": "Tekstil",
    "BULL": "Transportasi", "PSSI": "Tambang Pasir",
    "BUVA": "Pariwisata", "SMIL": "Logam & Mineral",
    "EMAS": "Logam Mulia", "BNBR": "Industri",
    "CUAN": "Pembiayaan", "AMRT ": "Ritel",
}


def _load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# Terjemahan sektor Yahoo Finance ke Bahasa Indonesia.
SECTOR_ID = {
    "Energy": "Energi",
    "Basic Materials": "Material Dasar",
    "Industrials": "Industri",
    "Consumer Cyclical": "Konsumsi Siklis",
    "Consumer Defensive": "Konsumsi Pertahanan",
    "Financial Services": "Keuangan",
    "Healthcare": "Kesehatan",
    "Technology": "Teknologi",
    "Communication Services": "Komunikasi",
    "Utilities": "Utilitas",
    "Real Estate": "Properti",
}


def translate_sector(name):
    return SECTOR_ID.get(name, name)


def sector_of(ticker):
    """Nama sektor sebuah kode saham (dalam Bahasa Indonesia)."""
    cache = _load_cache()
    raw = cache.get(ticker) or SECTOR_FALLBACK.get(ticker, "Lainnya")
    return translate_sector(raw)


def all_sectors():
    from collections import Counter
    from tickers import KOMPAS100

    return Counter(sector_of(t) for t in KOMPAS100)


if __name__ == "__main__":
    from collections import Counter
    from tickers import KOMPAS100

    for sector, count in sorted(all_sectors().items()):
        print(f"{sector:<24} {count}")
    unknown = [t for t in KOMPAS100 if sector_of(t) == "Lainnya"]
    print(f"\nBelum terpetakan ({len(unknown)}): {', '.join(unknown)}")
