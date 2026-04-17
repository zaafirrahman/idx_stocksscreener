"""
collect_idx_historical.py
==========================
Seed script — jalankan SEKALI secara manual untuk mengisi idx_ohlcv.parquet
dengan data historis 3 tahun (2022–2024) dari yfinance (.JK suffix).

Data ini TERPISAH dari parquet IDX utamamu (yang ada freq, foreign, dll).
Tujuannya hanya untuk training ML — OHLCV + derived momentum features.

Output: ml_data/idx_ohlcv.parquet
Schema: date | ticker | open | high | low | close | volume | [derived features]
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("ml_data")
OUTPUT_FILE = OUTPUT_DIR / "idx_ohlcv.parquet"

START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"

BATCH_SIZE  = 50     # fetch N tickers sekaligus (yfinance batch)
SLEEP_BATCH = 2.0    # detik antar batch

# IDX tickers — semua yang listed di IDX (~900 tickers)
# Kita auto-fetch dari Wikipedia / hardcode universe Pluang-mu
# Untuk simplicity, kita pakai IDX Composite components + common stocks
# Kamu bisa replace list ini dengan list ticker dari Pluang universe-mu

# NOTE: yfinance IDX format = "BBCA.JK", "TLKM.JK", dst.

# Untuk seed, kita ambil pendekatan: fetch IHSG composite + LQ45 + IDX80
# lalu expand ke semua yang bisa di-fetch

# Starter universe — kamu bisa expand dengan list dari scraped IDX-mu
IDX_TICKERS_RAW = """
AADI AALI ABBA ABDA ABMM ACES ACRO ACST ADCP ADES ADHI ADMF ADMG ADMR ADRO
AEGS AGAR AGII AGRO AGRS AHAP AIMS AISA AKKU AKPI AKRA AKSI ALDO ALII ALKA
ALMI ALTO AMAG AMAN AMAR AMFG AMIN AMMN AMMS AMOR AMRT ANDI ANJT ANTM APEX
APIC APII APLI APLN ARCI AREA ARGO ARII ARKA ARKO ARMY ARNA ARTA ARTI ARTO
ASBI ASDM ASGR ASHA ASII ASJT ASLC ASLI ASMI ASPI ASPR ASRI ASRM ASSA ATAP
ATIC ATLA AUTO AVIA AWAN AXIO AYAM AYLS BABP BABY BACA BAIK BAJA BALI BANK
BAPA BAPI BATA BATR BAUT BAYU BBCA BBHI BBKP BBLD BBMD BBNI BBRI BBRM BBSI
BBSS BBTN BBYB BCAP BCIC BCIP BDKR BDMN BEBS BEEF BEER BEKS BELI BELL BESS
BEST BFIN BGTG BHAT BHIT BIKA BIKE BIMA BINA BINO BIPI BIPP BIRD BISI BJBR
BJTM BKDP BKSL BKSW BLES BLOG BLTA BLTZ BLUE BMAS BMBL BMHS BMRI BMSR BMTR
BNBA BNBR BNGA BNII BNLI BOAT BOBA BOGA BOLA BOLT BOSS BPFI BPII BPTR BRAM
BREN BRIS BRMS BRNA BRPT BRRC BSBK BSDE BSIM BSML BSSR BSWD BTEK BTEL BTON
BTPN BTPS BUAH BUDI BUKA BUKK BULL BUMI BUVA BVIC BWPT BYAN CAKK CAMP CANI
CARE CARS CASA CASH CASS CBDK CBMF CBPE CBRE CBUT CCSI CDIA CEKA CENT CFIN
CGAS CHEK CHEM CHIP CINT CITA CITY CLAY CLEO CLPI CMNP CMNT CMPP CMRY CNKO
CNMA CNTB CNTX COAL COCO COIN COWL CPIN CPRI CPRO CRAB CRSN CSAP CSIS CSMI
CSRA CTBN CTRA CTTH CUAN CYBR DAAZ DADA DART DATA DAYA DCII DEAL DEFI DEPO
DEWA DEWI DFAM DGIK DGNS DGWG DIGI DILD DIVA DKFT DKHH DLTA DMAS DMMX DMND
DNAR DNET DOID DOOH DOSS DPNS DPUM DRMA DSFI DSNG DSSA DUCK DUTI DVLA DWGL
DYAN EAST ECII EDGE EKAD ELIT ELPI ELSA ELTY EMAS EMDE EMTK ENAK ENRG ENVY
ENZO EPAC EPMT ERAA ERAL ERTX ESIP ESSA ESTA ESTI ETWA EURO EXCL FAPA FAST
FASW FILM FIMP FIRE FISH FITT FLMC FMII FOLK FOOD FORE FORU FPNI FUJI FUTR
FWCT GAMA GDST GDYR GEMA GEMS GGRM GGRP GHON GIAA GJTL GLOB GLVA GMFI GMTD
GOLD GOLF GOLL GOOD GOTO GOTOM GPRA GPSO GRIA GRPH GRPM GSMF GTBO GTRA GTSI
GULA GUNA GWSA GZCO HADE HAIS HAJJ HALO HATM HBAT HDFA HDIT HEAL HELI HERO
HEXA HGII HILL HITS HKMU HMSP HOKI HOME HOMI HOPE HOTL HRME HRTA HRUM HUMI
HYGN IATA IBFN IBOS IBST ICBP ICON IDEA IDPR IFII IFSH IGAR IIKP IKAI IKAN
IKBI IKPM IMAS IMJS IMPC INAF INAI INCF INCI INCO INDF INDO INDR INDS INDX
INDY INET INKP INOV INPC INPP INPS INRU INTA INTD INTP IOTF IPAC IPCC IPCM
IPOL IPPE IPTV IRRA IRSX ISAP ISAT ISEA ISSP ITIC ITMA ITMG JARR JAST JATI
JAWA JAYA JECC JGLE JIHD JKON JMAS JPFA JRPT JSKY JSMR JSPT JTPE KAEF KAQI
KARW KAYU KBAG KBLI KBLM KBLV KBRI KDSI KDTN KEEN KEJU KETR KIAS KICI KIJA
KING KINO KIOS KJEN KKES KKGI KLAS KLBF KLIN KMDS KMTR KOBX KOCI KOIN KOKA
KONI KOPI KOTA KPIG KRAS KREN KRYA KSIX KUAS LABA LABS LAJU LAND LAPD LCGP
LCKM LEAD LFLO LIFE LINK LION LIVE LMAS LMAX LMPI LMSH LOPI LPCK LPGI LPIN
LPKR LPLI LPPF LPPS LRNA LSIP LTLS LUCK LUCY MABA MAGP MAHA MAIN MANG MAPA
MAPB MAPI MARI MARK MASB MAXI MAYA MBAP MBMA MBSS MBTO MCAS MCOL MCOR MDIA
MDIY MDKA MDKI MDLA MDLN MDRN MEDC MEDS MEGA MEJA MENN MERI MERK META MFMI
MGLV MGNA MGRO MHKI MICE MIDI MIKA MINA MINE MIRA MITI MKAP MKNT MKPI MKTR
MLBI MLIA MLPL MLPT MMIX MMLP MNCN MOLI MORA MPIX MPMX MPOW MPPA MPRO MPXL
MRAT MREI MSIE MSIN MSJA MSKY MSTI MTDL MTEL MTFN MTLA MTMH MTPS MTRA MTSM
MTWI MUTU MYOH MYOR MYTX NAIK NANO NASA NASI NATO NAYZ NCKL NELY NEST NETV
NFCX NICE NICK NICL NIKL NINE NIRO NISP NOBU NPGF NRCA NSSS NTBK NUSA NZIA
OASA OBAT OBMD OCAP OILS OKAS OLIV OMED OMRE OPMS PACK PADA PADI PALM PAMG
PANI PANR PANS PART PBID PBRX PBSA PCAR PDES PDPP PEGE PEHA PEVE PGAS PGEO
PGJO PGLI PGUN PICO PIPA PJAA PJHB PKPK PLAN PLAS PLIN PMJS PMMP PMUI PNBN
PNBS PNGO PNIN PNLF PNSE POLA POLI POLL POLU POLY POOL PORT POSA POWR PPGL
PPRE PPRI PPRO PRAY PRDA PRIM PSAB PSAT PSDN PSGO PSKT PSSI PTBA PTDU PTIS
PTMP PTMR PTPP PTPS PTPW PTRO PTSN PTSP PUDP PURA PURE PURI PWON PYFA PZZA
RAAM RAFI RAJA RALS RANC RATU RBMS RCCC RDTX REAL RELF RELI RGAS RICY RIGS
RIMO RISE RLCO RMKE RMKO ROCK RODA RONY ROTI RSCH RSGK RUIS RUNS SAFE SAGE
SAME SAMF SAPX SATU SBAT SBMA SCCO SCMA SCNP SCPI SDMU SDPC SDRA SEMA SFAN
SGER SGRO SHID SHIP SICO SIDO SILO SIMA SIMP SINI SIPD SKBM SKLT SKRN SKYB
SLIS SMAR SMBR SMCB SMDM SMDR SMGA SMGR SMIL SMKL SMKM SMLE SMMA SMMT SMRA
SMRU SMSM SNLK SOCI SOFA SOHO SOLA SONA SOSS SOTS SOUL SPMA SPRE SPTO SQMI
SRAJ SRIL SRSN SRTG SSIA SSMS SSTM STAA STAR STRK STTP SUGI SULI SUNI SUPA
SUPR SURE SURI SWAT SWID TALF TAMA TAMU TAPG TARA TAXI TAYS TBIG TBLA TBMS
TCID TCPI TDPM TEBE TECH TELE TFAS TFCO TGKA TGRA TGUK TIFA TINS TIRA TIRT
TKIM TLDN TLKM TMAS TMPO TNCA TOBA TOOL TOPS TOSK TOTL TOTO TOWR TOYS TPIA
TPMA TRAM TRGU TRIL TRIM TRIN TRIO TRIS TRJA TRON TRST TRUE TRUK TRUS TSPC
TUGU TYRE UANG UCID UDNG UFOE ULTJ UNIC UNIQ UNIT UNSP UNTD UNTR UNVR URBN
UVCR VAST VERN VICI VICO VINS VISI VIVA VKTR VOKS VRNA VTNY WAPO WBSA WEGE
WEHA WGSH WICO WIDI WIFI WIIM WIKA WINE WINR WINS WIRG WMPP WMUU WOMF WOOD
WOWS WSBP WSKT WTON YELO YOII YPAS YULE YUPI ZATA ZBRA ZINC ZONE ZYRX
""".split()

# Deduplicate
IDX_TICKERS_RAW = sorted(set(IDX_TICKERS_RAW))
IDX_TICKERS_YF  = [f"{t}.JK" for t in IDX_TICKERS_RAW]

print(f"Universe size: {len(IDX_TICKERS_YF)} tickers")


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV untuk batch tickers, return long-format DataFrame."""
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception as e:
        print(f"    [ERROR] Batch download failed: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    records = []

    # Single ticker: columns = [Open, High, Low, Close, Volume]
    if len(tickers) == 1:
        ticker = tickers[0]
        df = raw.copy()
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                      for c in df.columns]
        df["ticker"] = ticker.replace(".JK", "")
        df.index.name = "date"
        df = df.reset_index()
        return df[["date", "ticker", "open", "high", "low", "close", "volume"]]

    # Multi ticker: yfinance returns MultiIndex (field, ticker) or (ticker, field)
    # Detect order from level values
    lvl0 = raw.columns.get_level_values(0).unique().tolist()
    # If level 0 contains field names like 'Close', order is (field, ticker)
    field_first = any(x in lvl0 for x in ["Close", "Open", "High", "Low", "Volume"])

    for ticker in tickers:
        try:
            if field_first:
                # Access pattern: raw["Close"][ticker]
                if ticker not in raw.columns.get_level_values(1):
                    continue
                df = pd.DataFrame({
                    "open":   raw["Open"][ticker] if "Open" in raw.columns.get_level_values(0) else None,
                    "high":   raw["High"][ticker] if "High" in raw.columns.get_level_values(0) else None,
                    "low":    raw["Low"][ticker]  if "Low"  in raw.columns.get_level_values(0) else None,
                    "close":  raw["Close"][ticker],
                    "volume": raw["Volume"][ticker] if "Volume" in raw.columns.get_level_values(0) else None,
                }, index=raw.index)
            else:
                # Access pattern: raw[ticker]["Close"]
                if ticker not in raw.columns.get_level_values(0):
                    continue
                sub = raw[ticker].copy()
                sub.columns = [c.lower() for c in sub.columns]
                # Handle adj close column name variations
                if "adj close" in sub.columns:
                    sub = sub.drop(columns=["adj close"], errors="ignore")
                df = sub[["open", "high", "low", "close", "volume"]]

            df = df.dropna(subset=["close"])
            if df.empty:
                continue
            df = df.copy()
            df["ticker"] = ticker.replace(".JK", "")
            df.index.name = "date"
            df = df.reset_index()
            records.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])
        except Exception:
            continue

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah fitur momentum per ticker — ini yang akan jadi ticker-level features
    saat training ML nantinya.

    Computed per ticker (groupby), sorted by date.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    def per_ticker(g):
        g = g.copy()
        c = g["close"]
        v = g["volume"]

        g["ret_1d"]  = c.pct_change(1) * 100          # return kemarin
        g["ret_3d"]  = c.pct_change(3) * 100
        g["ret_5d"]  = c.pct_change(5) * 100
        g["ret_20d"] = c.pct_change(20) * 100

        g["vol_20d_avg"] = v.rolling(20).mean()
        g["vol_ratio"]   = v / g["vol_20d_avg"]        # volume spike indicator

        # Volatility
        g["volatility_20d"] = (c.pct_change() * 100).rolling(20).std()

        # Price vs moving averages
        g["ma5"]  = c.rolling(5).mean()
        g["ma20"] = c.rolling(20).mean()
        g["above_ma5"]  = (c > g["ma5"]).astype(int)
        g["above_ma20"] = (c > g["ma20"]).astype(int)

        # Target label: apakah next day close naik >3% dari hari ini?
        g["next_day_ret"] = c.pct_change(1).shift(-1) * 100
        g["label"]        = (g["next_day_ret"] > 3.0).astype(int)

        return g

    result = df.groupby("ticker", group_keys=False).apply(per_ticker)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"{'='*60}")
    print(f"  Quant Radar — IDX OHLCV Historical Collector")
    print(f"  Period : {START_DATE} → {END_DATE}")
    print(f"  Tickers: {len(IDX_TICKERS_YF)}")
    print(f"  Label  : next_day_ret > 3%")
    print(f"{'='*60}\n")

    OUTPUT_DIR.mkdir(exist_ok=True)

    if OUTPUT_FILE.exists():
        existing = pd.read_parquet(OUTPUT_FILE)
        print(f"[INFO] File sudah ada: {len(existing)} rows")
        ans = input("Overwrite? (y/n): ").strip().lower()
        if ans != "y":
            print("[ABORT] Cancelled.")
            return

    all_batches = []
    batches = [IDX_TICKERS_YF[i:i+BATCH_SIZE]
               for i in range(0, len(IDX_TICKERS_YF), BATCH_SIZE)]

    print(f"[1/3] Fetching IDX OHLCV in {len(batches)} batches of {BATCH_SIZE}...\n")

    failed_tickers = []
    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i:2d}/{len(batches)} — {batch[0]} … {batch[-1]}")
        df_batch = fetch_batch(batch, START_DATE, END_DATE)

        if df_batch.empty:
            print(f"    [WARN] Batch {i} returned no data")
            failed_tickers.extend(batch)
        else:
            success = df_batch["ticker"].nunique()
            print(f"    ✓ {success}/{len(batch)} tickers, {len(df_batch)} rows")
            all_batches.append(df_batch)

        if i < len(batches):
            time.sleep(SLEEP_BATCH)

    if not all_batches:
        print("[ERROR] No data fetched. Check internet / yfinance.")
        return

    print(f"\n[2/3] Combining & computing derived features...")
    df_all = pd.concat(all_batches, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])

    # Remove timezone info jika ada
    if hasattr(df_all["date"].dt, "tz") and df_all["date"].dt.tz is not None:
        df_all["date"] = df_all["date"].dt.tz_localize(None)

    print(f"  Raw rows     : {len(df_all):,}")
    print(f"  Unique ticker: {df_all['ticker'].nunique()}")
    print(f"  Date range   : {df_all['date'].min().date()} → {df_all['date'].max().date()}")

    df_featured = add_derived_features(df_all)

    # Drop rows tanpa label (last row per ticker, atau awal yg belum ada rolling)
    before = len(df_featured)
    df_featured = df_featured.dropna(subset=["label", "ret_5d", "volatility_20d"])
    print(f"  After dropna : {len(df_featured):,} rows (dropped {before - len(df_featured):,})")

    label_dist = df_featured["label"].value_counts(normalize=True) * 100
    print(f"\n  Label distribution:")
    print(f"    0 (tidak naik >3%): {label_dist.get(0, 0):.1f}%")
    print(f"    1 (naik >3%)       : {label_dist.get(1, 0):.1f}%")

    if failed_tickers:
        print(f"\n  Failed tickers ({len(failed_tickers)}): {failed_tickers[:10]}{'...' if len(failed_tickers)>10 else ''}")

    print(f"\n[3/3] Saving to {OUTPUT_FILE}...")
    df_featured.to_parquet(OUTPUT_FILE, engine="pyarrow",
                           compression="snappy", index=False)
    print(f"  ✓ Saved: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\n[DONE] {df_featured['ticker'].nunique()} tickers, {len(df_featured):,} rows stored.")
    print(f"[NEXT] Jalankan train.py untuk training XGBoost model.")


if __name__ == "__main__":
    main()
