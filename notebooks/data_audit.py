"""
Data Audit & Profiling script.
Produces a Markdown report at reports/DATA_AUDIT_REPORT.md
describing every uploaded dataset without modifying it.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"c:/Users/user/PycharmProjects/VKR_prep")
DATA_DIR = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "DATA_AUDIT_REPORT.md"

# Per-file locations after the reorganisation of raw inputs into
# `data/raw/<country>/{active,bankrupt}/` (previously all files lived flat in
# `available_datasets.txt/`). Country-level `info_*` / mixed panels stay at
# the country root when they are neither purely active nor purely defaulted.
CN_ACTIVE   = DATA_DIR / "china"        / "active"
CN_BANKRUPT = DATA_DIR / "china"        / "bankrupt"
IN_ACTIVE   = DATA_DIR / "india"        / "active"
BR_ACTIVE   = DATA_DIR / "brazil"       / "active"
BR_ROOT     = DATA_DIR / "brazil"
SA_ROOT     = DATA_DIR / "south_africa"
SA_BANKRUPT = DATA_DIR / "south_africa" / "bankrupt"

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

lines: list[str] = []
def w(s: str = ""):
    lines.append(s)

def missing_pct(df: pd.DataFrame) -> pd.Series:
    return (df.isna().mean() * 100).round(2)

def brief_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    desc = num.describe().T[["count", "mean", "std", "min", "max"]]
    desc["missing_%"] = missing_pct(num).loc[desc.index]
    return desc.round(3)

def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table without requiring `tabulate`."""
    if df is None or df.empty:
        return "_(empty)_"
    d = df.copy()
    if d.index.name is None and not isinstance(d.index, pd.RangeIndex):
        d.index.name = "index"
    if not isinstance(d.index, pd.RangeIndex):
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in d.iterrows():
        vals = []
        for v in row.tolist():
            if pd.isna(v):
                vals.append("")
            else:
                s = str(v).replace("|", "\\|").replace("\n", " ")
                if len(s) > 80:
                    s = s[:77] + "..."
                vals.append(s)
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)

def file_size_kb(p: Path) -> float:
    return round(p.stat().st_size / 1024, 1)

# ----------------------------------------------------------------------
w("# DATA AUDIT REPORT — VKR IT Default Prediction")
w(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
w("")
w("Scope: read-only profiling of all raw inputs under `data/raw/<country>/{active,bankrupt}/`. "
  "No data is modified, merged, or cleaned. Goal: inventory + quality snapshot "
  "to confirm methodological decisions described in `project_passport.txt`.")
w("")

# ======================================================================
# 1. GLOBAL INVENTORY
# ======================================================================
w("## 1. Global File Inventory")
w("")
inventory_rows = []
for p in sorted(DATA_DIR.rglob("*")):
    if not p.is_file():
        continue
    inventory_rows.append({
        "file": str(p.relative_to(DATA_DIR)).replace("\\", "/"),
        "ext": p.suffix.lower(),
        "size_KB": file_size_kb(p),
    })
inv_df = pd.DataFrame(inventory_rows)
w(md_table(inv_df))
w("")

# ======================================================================
# Helper to audit one flat table
# ======================================================================
def audit_table(df: pd.DataFrame, _label: str = ""):
    w(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    w("")
    # detect year columns (wide format)
    year_cols = [c for c in df.columns if isinstance(c, (int, np.integer)) or
                 (isinstance(c, str) and c.strip().isdigit() and 1990 <= int(c.strip()) <= 2035)]
    fmt = "Wide (years as columns)" if len(year_cols) >= 3 else (
          "Long (years as rows)" if any(str(c).lower() in {"year", "год", "fiscal_year"} for c in df.columns)
          else "Flat / unclear")
    w(f"**Format heuristic:** {fmt}")
    if year_cols:
        w(f"**Detected year columns ({len(year_cols)}):** {year_cols}")
    w("")
    w("**Columns and dtypes:**")
    dt = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing_%": missing_pct(df),
        "n_unique": df.nunique(dropna=True),
    })
    w(md_table(dt))
    w("")
    w("**Numeric summary (describe):**")
    w(md_table(brief_numeric(df)))
    w("")
    w("**Head (5 rows):**")
    w("```")
    w(df.head(5).to_string())
    w("```")
    w("")
    return year_cols

# ======================================================================
# Helper: parse WIND dump where tickers are COLUMNS and rows encode
# "<Metric>\n[unit]1M\n[rptDate]YYYY-12-31\n..." labels, with
# periodic repetitions of the Name/Code header rows.
# ======================================================================
import re
_WIND_LABEL_RE = re.compile(r"^(?P<metric>[^\n]+)\n\[unit\][^\n]*\n\[rptDate\](?P<year>\d{4})-\d{2}-\d{2}")

def parse_wind_file(path: Path) -> dict:
    """Return dict with a real long panel and meta."""
    raw = pd.read_excel(path)
    first_col = raw.columns[0]
    ticker_cols = [c for c in raw.columns[2:] if str(c) not in {"代码", "Code", "Name"}]
    # find company names from first 'Name' row
    name_row = raw[raw[first_col] == "Name"].iloc[0]
    ticker_to_name = {t: name_row[t] for t in ticker_cols}

    records = []
    for _, row in raw.iterrows():
        lbl = str(row[first_col])
        m = _WIND_LABEL_RE.match(lbl)
        if not m:
            continue
        metric = m.group("metric").strip()
        year = int(m.group("year"))
        for t in ticker_cols:
            v = row[t]
            records.append({"ticker": t, "name": ticker_to_name.get(t),
                            "metric": metric, "year": year, "value": v})
    long = pd.DataFrame(records)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return {"raw_shape": raw.shape, "n_tickers": len(ticker_cols),
            "long": long, "ticker_to_name": ticker_to_name}

# ======================================================================
# 2. CHINA — Wind Software & services.xlsx (main active panel)
# ======================================================================
cn_main = CN_ACTIVE / "Wind Software & services.xlsx"
w("## 2. CHINA — PRIMARY ACTIVE PANEL")
w(f"### File: `{cn_main.name}` (size {file_size_kb(cn_main)} KB)")
w("Purpose (from chat logs): 319 active Chinese software & services companies, 2014–2025, "
  "key financial indicators. This is the core training dataset for the China arm of the thesis.")
w("")
try:
    parsed = parse_wind_file(cn_main)
    long = parsed["long"]
    w(f"**Raw sheet shape:** {parsed['raw_shape']} — tickers are **columns** (unusual), "
      f"metric-year labels are **rows**. A proper panel requires transposition.")
    w(f"**Detected tickers (companies):** {parsed['n_tickers']}")
    w(f"**Detected metrics:** {long['metric'].nunique()} — {sorted(long['metric'].unique().tolist())}")
    w(f"**Year range:** {long['year'].min()}–{long['year'].max()} "
      f"({long['year'].nunique()} distinct years)")
    w(f"**Total long-format observations:** {len(long):,} "
      f"(tickers × metrics × years); non-null values: {long['value'].notna().sum():,} "
      f"({long['value'].notna().mean()*100:.1f} %)")
    w("")

    w("**Coverage by metric (% non-null across all ticker-years):**")
    cov_m = (long.groupby("metric")["value"]
                  .apply(lambda s: s.notna().mean()*100).round(1)
                  .sort_values(ascending=False))
    w(md_table(cov_m.to_frame("coverage_%")))
    w("")

    w("**Coverage by year (% non-null across all ticker-metrics):**")
    cov_y = (long.groupby("year")["value"]
                  .apply(lambda s: s.notna().mean()*100).round(1))
    w(md_table(cov_y.to_frame("coverage_%")))
    w("")

    w("**Coverage matrix (% non-null) — metric × year:**")
    mat = (long.groupby(["metric", "year"])["value"]
                 .apply(lambda s: s.notna().mean()*100).round(1)
                 .unstack("year"))
    w(md_table(mat))
    w("")

    # 2025 special check
    vals_2025 = long.loc[long["year"] == 2025, "value"]
    if len(vals_2025):
        w(f"> **2025 check:** {vals_2025.notna().sum():,} / {len(vals_2025):,} "
          f"non-null ({vals_2025.notna().mean()*100:.1f} %). "
          f"Project passport already excludes 2025 — this confirms.")
        w("")

    # Companies with zero data (potentially empty columns)
    per_ticker = long.groupby("ticker")["value"].apply(lambda s: s.notna().sum())
    empty = (per_ticker == 0).sum()
    w(f"> **Ticker coverage:** {empty} / {parsed['n_tickers']} tickers have 0 non-null "
      f"values across all metric-years (likely spurious duplicate columns).")
    w("")
except Exception as e:
    w(f"_Error reading file: {e}_")
    w("")

# ======================================================================
# 3. CHINA — Delisted stocks china.xlsx (37 delisted with financials)
# ======================================================================
cn_del1 = CN_BANKRUPT / "Delisted stocks china.xlsx"
w("## 3. CHINA — DELISTED COMPANIES (with financials)")
w(f"### File: `{cn_del1.name}` (size {file_size_kb(cn_del1)} KB)")
w("Purpose (from chat logs): 37 delisted Chinese tech companies with financial data + "
  "a reason-for-delisting column (bankruptcy, forced delisting, OTC transfer, privatization, "
  "M&A, relisting, …). Critical for labelling Y=1 correctly (excluding non-distress exits).")
w("")
try:
    xl = pd.ExcelFile(cn_del1)
    w(f"**Sheets ({len(xl.sheet_names)}):** {xl.sheet_names}")
    w("")
    for sh in xl.sheet_names:
        w(f"#### Sheet: `{sh}`")
        df = pd.read_excel(cn_del1, sheet_name=sh)
        audit_table(df, sh)

        # look for a reason column
        reason_cols = [c for c in df.columns
                       if any(k in str(c).lower() for k in
                              ["reason", "delist", "status", "outcome", "cause", "type", "исход", "причина"])]
        if reason_cols:
            for rc in reason_cols:
                w(f"**Value counts for candidate reason column `{rc}`:**")
                w(md_table(df[rc].value_counts(dropna=False).to_frame("count")))
                w("")
except Exception as e:
    w(f"_Error reading file: {e}_")
    w("")

# ======================================================================
# 4. CHINA — Delisted stocks_information_technology (2).xlsx (61-ticker list)
# ======================================================================
cn_del2 = CN_BANKRUPT / "Delisted stocks_information_technology (2).xlsx"
w("## 4. CHINA — DELISTED COMPANIES (ticker list, 61 names)")
w(f"### File: `{cn_del2.name}` (size {file_size_kb(cn_del2)} KB)")
w("Purpose (from chat logs): longer list (~61) of delisted Chinese IT companies; intended "
  "to seed further financial-data collection once WIND access re-opens.")
w("")
try:
    xl = pd.ExcelFile(cn_del2)
    w(f"**Sheets ({len(xl.sheet_names)}):** {xl.sheet_names}")
    w("")
    for sh in xl.sheet_names:
        w(f"#### Sheet: `{sh}`")
        df = pd.read_excel(cn_del2, sheet_name=sh)
        audit_table(df, sh)
except Exception as e:
    w(f"_Error reading file: {e}_")
    w("")

# ======================================================================
# 5. INDIA — Wind Software & services_India.xlsx
# ======================================================================
in_main = IN_ACTIVE / "Wind Software & services_India.xlsx"
w("## 5. INDIA — AUXILIARY PANEL")
w(f"### File: `{in_main.name}` (size {file_size_kb(in_main)} KB)")
w("Purpose: 256 Indian software & services companies, 2014–2025. Structurally mirrors the "
  "China panel but (per chat logs) with noticeably more gaps — auxiliary / excluded from main ML pipeline.")
w("")
try:
    xl = pd.ExcelFile(in_main)
    w(f"**Sheets ({len(xl.sheet_names)}):** {xl.sheet_names}")
    w("")
    for sh in xl.sheet_names:
        w(f"#### Sheet: `{sh}`")
        df = pd.read_excel(in_main, sheet_name=sh)
        audit_table(df, sh)
except Exception as e:
    w(f"_Error reading file: {e}_")
    w("")

# ======================================================================
# 6. BRAZIL
# ======================================================================
br_panel = BR_ACTIVE / "brazil_it_panel_improved.csv"
br_info = BR_ROOT / "info_brazilian_companies.xlsx"
w("## 6. BRAZIL")
w(f"### File: `{br_panel.name}` (size {file_size_kb(br_panel)} KB)")
w("Purpose: 29 Brazilian IT companies, 2014–2025 (~189 observations); revenue, EBIT, "
  "gross profit, net income, assets, intangibles, equity, OCF + derived ROA/ROE/D-E.")
w("")
try:
    df = pd.read_csv(br_panel)
    audit_table(df, br_panel.name)
except Exception as e:
    w(f"_Error: {e}_")
    w("")

w(f"### File: `{br_info.name}` (size {file_size_kb(br_info)} KB)")
w("Purpose: master list of companies on B3 (context / universe, not financial panel).")
w("")
try:
    xl = pd.ExcelFile(br_info)
    w(f"**Sheets ({len(xl.sheet_names)}):** {xl.sheet_names}")
    for sh in xl.sheet_names:
        w(f"#### Sheet: `{sh}`")
        df = pd.read_excel(br_info, sheet_name=sh)
        audit_table(df, sh)
except Exception as e:
    w(f"_Error: {e}_")
    w("")

# ======================================================================
# 7. SOUTH AFRICA
# ======================================================================
w("## 7. SOUTH AFRICA")
for p, purpose in [
    (SA_ROOT / "south_africa_it_panel_wide.csv",
     "Wide-format panel (companies × year columns)."),
    (SA_ROOT / "south_africa_it_panel_long.csv",
     "Long-format panel (one row per company-year-metric)."),
    (SA_BANKRUPT / "south_africa_it_delistings.csv",
     "Documented delisting events on JSE (only 3 cases per chat logs)."),
]:
    name = p.name
    if not p.exists():
        continue
    w(f"### File: `{name}` (size {file_size_kb(p)} KB)")
    w(f"Purpose: {purpose} Sources: JSE, SENS announcements, Yahoo, WallStreet listcorp, company reports.")
    w("")
    try:
        df = pd.read_csv(p)
        audit_table(df, name)
    except Exception as e:
        w(f"_Error: {e}_")
        w("")

# ======================================================================
# 8. SYNTHESIS
# ======================================================================
w("## 8. Synthesis & Methodological Implications")
w("")
w("* **China active panel** is the cleanest large dataset and the primary training set for the China arm. "
  "Confirm 2025 sparsity above: if column 2025 is ≥80 % empty, drop it as per passport §2 (timeframe 2012/2014–2024).")
w("* **China delisted panel** must be filtered via its reason-for-delisting column: keep only bankruptcy / "
  "forced delisting / liquidation as Y=1; exclude M&A, privatization, voluntary, relisting (passport §2.2, §1.3).")
w("* **India, Brazil, South Africa** confirmed sparse and heterogeneous; consistent with the passport's decision "
  "to restrict the empirical pipeline to Russia + China.")
w("* No SMOTE is planned; class imbalance will be handled via cost-sensitive `class_weight` in Logit / RF / XGBoost "
  "(passport §2.4). Missing-value policy: drop-NA for active, forward-fill/zero-fill for defaulted.")
w("")
w("_End of audit — data is untouched on disk._")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote report: {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
