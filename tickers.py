"""Sumber tunggal daftar saham yang dipantau.

Indeks KOMPAS100 -- konstituen hasil evaluasi mayor 27 Juli 2026,
berlaku efektif 3 Agustus 2026 (sumber: Kompas/Kontan).
"""

KOMPAS100 = [
    "AADI", "ACES", "ADMR", "ADRO", "AKRA",
    "AMMN", "AMRT", "ANTM", "ARCI", "ARTO",
    "ASII", "BBCA", "BBNI", "BBRI", "BBTN",
    "BBYB", "BFIN", "BIPI", "BKSL", "BMRI",
    "BNBR", "BRIS", "BRMS", "BRPT", "BSDE",
    "BUKA", "BULL", "BUMI", "BUVA", "CBDK",
    "CMRY", "COIN", "CPIN", "CTRA", "CUAN",
    "DEWA", "DSNG", "ELSA", "EMAS", "EMTK",
    "ENRG", "ERAA", "ESSA", "EXCL", "GGRM",
    "GOTO", "HEAL", "HRTA", "HRUM", "ICBP",
    "IMPC", "INCO", "INDF", "INDY", "INET",
    "INKP", "ISAT", "ITMG", "JPFA", "JSMR",
    "KIJA", "KLBF", "KPIG", "LSIP", "MAPA",
    "MAPI", "MBMA", "MDKA", "MEDC", "MIKA",
    "MINA", "MYOR", "NCKL", "PANI", "PGAS",
    "PGEO", "PNLF", "PSAB", "PTBA", "PTRO",
    "PWON", "RAJA", "RATU", "RMKE", "SCMA",
    "SGER", "SMGR", "SMIL", "SMRA", "SSIA",
    "TAPG", "TINS", "TLKM", "TOBA", "TOWR",
    "TPIA", "UNTR", "UNVR", "WIFI", "WIRG",
]

# IHSG dipakai sebagai pembanding (beta/relatif strength).
BENCHMARK = "^JKSE"

# Saham yang baru listing dan sejarahnya pendek tetap diunduh,
# tetapi otomatis dikeluarkan dari training sampai datanya cukup.
MIN_HISTORY_ROWS = 260

SUFFIX = ".JK"


def all_tickers():
    """Semua kode untuk diunduh, termasuk pembanding."""
    return [f"{t}{SUFFIX}" for t in KOMPAS100] + [BENCHMARK]


if __name__ == "__main__":
    print(f"Jumlah konstituen KOMPAS100: {len(KOMPAS100)}")
    dupes = {t for t in KOMPAS100 if KOMPAS100.count(t) > 1}
    if dupes:
        raise SystemExit(f"Kode duplikat: {dupes}")
    print("Tidak ada kode duplikat.")
