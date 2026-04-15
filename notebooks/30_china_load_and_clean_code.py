# ===== CELL 1 =====import warnings; warnings.filterwarnings('ignore')
import re
import numpy as np
import pandas as pd
from pathlib import Path

CHINA_RAW = Path('../data/raw/china')
WIND = CHINA_RAW / 'active' / 'Wind Software & services.xlsx'
DELIST = CHINA_RAW / 'bankrupt' / 'Delisted stocks china.xlsx'

PROCESSED = Path('../data/processed'); PROCESSED.mkdir(parents=True, exist_ok=True)

print('Wind:   ', WIND, WIND.exists())
print('Delist: ', DELIST, DELIST.exists())
# ===== CELL 3 =====# Category 1 — True defaults (forced delisting / bankruptcy / non-compliance)
DEFAULT_NAMES = [
    'ChinaCache International Holdings', 'China TechFaith Wireless Comm',
    'China Sunergy Co Ltd', 'Z-Obee Holdings Ltd', 'GEONG International Ltd',
    'China Finance Online Co Ltd', 'LDK Solar Co Ltd', 'LED International Holdings',
    'Link Motion Inc', 'ChinaSing Investment Holdings', 'RCG Holdings Ltd',
]
# Category 0 — Strategic exits (M&A / privatization / voluntary)
STRATEGIC_NAMES = [
    'Actions Semiconductor Co Ltd', 'AutoNavi Holdings Ltd', 'LCT Holdings Ltd',
    'China Mobile Games and Entertainment', 'China Transinfo Technology Corp',
    'Sinotel Technologies', 'iDreamSky Technology Holdings',
    'Elec & Eltek International Co', 'eFuture Information Technology',
    'Sungy Mobile Ltd', 'Gridsum Holding Inc', 'Hollysys Automation Technologies',
    'Hanwha Q Cells Co Ltd', 'iSoftStone Holdings Ltd', 'JA Solar Holdings Co Ltd',
    'KongZhong Corp', 'Sky-Mobi Ltd', 'Montage Technology Group Ltd',
    'NetDimensions (Holdings) Ltd', 'OneConnect Financial Technology',
    'O2Micro International Ltd', 'Qihoo 360 Technology Co Ltd', 'SINA Corp',
    'Semiconductor Manufacturing International Corp',
]
SKIP_NAMES = ['ReneSola Ltd']  # not actually delisted

def _norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

DEFAULT_KEYS   = [_norm(x) for x in DEFAULT_NAMES]
STRATEGIC_KEYS = [_norm(x) for x in STRATEGIC_NAMES]
SKIP_KEYS      = [_norm(x) for x in SKIP_NAMES]

def _prefix_match(n, keys, min_common=8):
    """Match if longest common prefix is >= min_common (or full length of shorter)."""
    if not n: return False
    for k in keys:
        if not k: continue
        m = min(len(n), len(k))
        common = 0
        for i in range(m):
            if n[i] == k[i]: common += 1
            else: break
        if common >= min(min_common, m):
            return True
    return False

def classify_delisted(name, sheet_name=None):
    """Return 1=default, 0=strategic, None=skip. Tries both B2 name and sheet name."""
    candidates = [_norm(name)]
    if sheet_name is not None:
        s = re.sub(r'[-\s]*financ.*$', '', str(sheet_name), flags=re.I)
        candidates.append(_norm(s))
    for n in candidates:
        if not n: continue
        if _prefix_match(n, SKIP_KEYS):      return None
        if _prefix_match(n, DEFAULT_KEYS):   return 1
        if _prefix_match(n, STRATEGIC_KEYS): return 0
    return 0  # unknown → treat as strategic (conservative)

print(f'{len(DEFAULT_NAMES)} defaults, {len(STRATEGIC_NAMES)} strategic, {len(SKIP_NAMES)} skip')

# ===== CELL 5 =====WIND_METRIC_MAP = {
    'oper_rev':                 'total_revenue',
    'ebit2':                    'ebit',
    'ebitda2':                  'ebitda',
    'tot_assets':               'total_assets',
    'tot_liab':                 'total_liab',
    'tot_equity':               'total_equity',
    'tot_cur_assets':           'current_assets',
    'cash_cash_equ_beg_period': 'cash',
    'intang_assets':            'intangibles',
    'net_cash_flows_oper_act':  'cfo',
    'int_exp':                  'interest_expense',
    'rd_exp':                   'rd_expense',
    # current_liab + net_profit + total_debt берём из akshare-енричмента (см. cell 5a)
}

def parse_wind(path):
    raw = pd.read_excel(path, header=None)
    tickers = raw.iloc[0, 2:].tolist()
    names   = raw.iloc[1, 2:].tolist()
    rows = []
    for i in range(raw.shape[0]):
        mtxt = str(raw.iat[i, 0])
        short = raw.iat[i, 1]
        if mtxt in ('代码', 'Name') or pd.isna(short) or short not in WIND_METRIC_MAP:
            continue
        m = re.search(r'\[rptDate\](\d{4})', mtxt)
        if not m: continue
        year = int(m.group(1))
        metric = WIND_METRIC_MAP[short]
        for t, n, v in zip(tickers, names, raw.iloc[i, 2:].tolist()):
            rows.append((t, n, year, metric, v))
    long_df = (pd.DataFrame(rows, columns=['ticker','company_name','year','metric','value'])
                 .dropna(subset=['value'])
                 .drop_duplicates(subset=['ticker','year','metric'], keep='last'))
    wide = long_df.pivot_table(
        index=['ticker','company_name','year'],
        columns='metric', values='value', aggfunc='first'
    ).reset_index()
    wide.columns.name = None
    return wide

wind_panel = parse_wind(WIND)
wind_panel['source_class'] = 'active'
wind_panel['target'] = 0
print(f'Wind: {wind_panel.shape}, tickers={wind_panel.ticker.nunique()}, '
      f'years={sorted(wind_panel.year.unique())}')
wind_panel.head(3)

# ===== CELL 7 =====DELIST_LABEL_MAP = [
    (('total op rev', 'operating revenue', 'total revenue'),       'total_revenue'),
    (('operating income', 'ebit'),                                  'ebit'),
    (('ebitda',),                                                   'ebitda'),
    (('total assets',),                                             'total_assets'),
    (('total liabilities', 'total liab'),                           'total_liab'),
    (('total equity', "total shareholders' equity",
      "total shareholder's equity", 'total stockholders equity',
      'shareholders equity', "shareholders' equity"),               'total_equity'),
    (('total current assets', 'current assets'),                    'current_assets'),
    (('total current liab', 'current liab'),                        'current_liab'),
    (('cash and cash equivalents', 'cash & cash equivalents',
      'cash and equivalents', 'eop cash balance'),                  'cash'),
    (('intangible assets', 'intangibles'),                          'intangibles'),
    (('short-term borrow', 'short term borrow', 'short-term debt'), 'st_borrow'),
    (('cash flow from operating', 'net cash flow from operating',
      'operating cash flow', 'net cash flows from operating'),      'cfo'),
    (('interest expense',),                                         'interest_expense'),
    (('r&d', 'research and development'),                           'rd_expense'),
]

def _norm_label(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 &]', ' ', str(s).lower())).strip()

def _canonical(label):
    n = _norm_label(label)
    for keys, canon in DELIST_LABEL_MAP:
        if any(k in n for k in keys): return canon
    return None

def parse_delist_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    if df.shape[0] < 8 or df.shape[1] < 2: return None
    ticker, name = df.iat[0, 1], df.iat[1, 1]
    if pd.isna(ticker): return None

    date_row = None
    for i in range(4, 12):
        vals = df.iloc[i, 1:].tolist()
        dates, total = 0, 0
        for v in vals:
            if pd.isna(v): continue
            total += 1
            if hasattr(v, 'year'): dates += 1
            elif isinstance(v, (int, float)) and 30000 < v < 80000: dates += 1
            elif isinstance(v, str) and re.search(r'20\d{2}', v): dates += 1
        if total and dates == total: date_row = i; break
    if date_row is None: return None

    period_row = date_row + 1
    raw_dates = df.iloc[date_row, 1:].tolist()
    periods = df.iloc[period_row, 1:].astype(str).str.lower().str.strip().tolist()

    years, keep_cols = [], []
    for j, (d, per) in enumerate(zip(raw_dates, periods)):
        if per not in ('ann.', 'ann', 'annual'): continue
        if pd.isna(d): continue
        yr = None
        if hasattr(d, 'year'): yr = d.year
        elif isinstance(d, (int, float)):
            try: yr = (pd.to_datetime('1899-12-30') + pd.Timedelta(days=int(d))).year
            except Exception: pass
        elif isinstance(d, str):
            m = re.search(r'(20\d{2})', d)
            if m: yr = int(m.group(1))
        if yr:
            years.append(yr); keep_cols.append(j + 1)
    if not keep_cols: return None

    records = {y: {} for y in years}
    for i in range(period_row + 2, df.shape[0]):
        canon = _canonical(df.iat[i, 0])
        if canon is None: continue
        for yr, col in zip(years, keep_cols):
            v = df.iat[i, col]
            if pd.isna(v): continue
            records[yr].setdefault(canon, v)
    out = [{'ticker': ticker, 'company_name': name, 'year': yr, **m}
           for yr, m in records.items() if m]
    return pd.DataFrame(out) if out else None


def load_delisted(path):
    xl = pd.ExcelFile(path)
    frames, log = [], []
    seen_tickers = set()
    for s in xl.sheet_names:
        if re.fullmatch(r'Sheet\d+', s): continue
        parsed = parse_delist_sheet(path, s)
        if parsed is None or parsed.empty:
            log.append((s, 'skip — no annual data')); continue
        tkr = parsed.iat[0, parsed.columns.get_loc('ticker')]
        if tkr in seen_tickers:
            log.append((s, f'skip — duplicate ticker {tkr}')); continue
        seen_tickers.add(tkr)
        b2_name = parsed.iat[0, parsed.columns.get_loc('company_name')]
        cat = classify_delisted(b2_name, sheet_name=s)
        if cat is None:
            log.append((s, 'skip — excluded (ReneSola)')); continue
        parsed['target'] = cat
        parsed['source_class'] = 'default_delisted' if cat == 1 else 'strategic_delisted'
        frames.append(parsed)
        log.append((s, f'ok → target={cat}, name="{b2_name}"'))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), log

delisted_panel, log = load_delisted(DELIST)
for s, msg in log: print(f'  {s:34s} {msg}')
print(f'\nDelisted combined: {delisted_panel.shape}')
print(f'  target=1: {delisted_panel[delisted_panel.target==1].ticker.nunique()} компаний')
print(f'  target=0: {delisted_panel[delisted_panel.target==0].ticker.nunique()} компаний')

# ===== CELL 9 =====CORE_COLS = [
    'total_revenue', 'ebit', 'ebitda', 'total_assets', 'total_liab', 'total_equity',
    'current_assets', 'current_liab', 'cash', 'intangibles',
    'cfo', 'interest_expense', 'rd_expense', 'net_profit', 'total_debt',
]
ID_COLS      = ['ticker', 'company_name', 'year']
SERVICE_COLS = ['target', 'source_class']
ALL_COLS     = ID_COLS + CORE_COLS + SERVICE_COLS

# Выравниваем колонки (Wind не отдаёт net_profit/current_liab/total_debt — будут NaN до енричмента)
for frame in (wind_panel, delisted_panel):
    for c in CORE_COLS:
        if c not in frame.columns: frame[c] = np.nan

panel = pd.concat([wind_panel[ID_COLS + CORE_COLS + SERVICE_COLS],
                   delisted_panel[ID_COLS + CORE_COLS + SERVICE_COLS]],
                  ignore_index=True)
panel = panel[ALL_COLS].sort_values(['target','ticker','year'], ascending=[False,True,True]).reset_index(drop=True)

print(f'Combined panel: {panel.shape}')
print(f'Unique tickers: {panel.ticker.nunique()}')
print(f'\nПо source_class:')
print(panel.groupby('source_class')['ticker'].nunique().to_string())
print(f'\nNaN по колонкам (%) — ДО енричмента akshare:')
print((panel[CORE_COLS].isna().mean()*100).round(1).to_string())

# ===== CELL 12 =====# --- Load enrichment CSVs ---
ENRICH_ACTIVE   = CHINA_RAW / 'china_fetched_metrics.csv'
ENRICH_DELISTED = CHINA_RAW / 'delisted_fetched_metrics_raw.csv'

# Active tickers (akshare): balance-sheet и income-statement лежат в РАЗНЫХ строках одного (ticker,year).
# Сначала схлопываем по (ticker,year), беря первое непустое значение каждой метрики.
fa = pd.read_csv(ENRICH_ACTIVE)
fa = fa.rename(columns={
    'current_liabilities':       'current_liab_e',
    'total_debt':                'total_debt_e',
    'net_income':                'net_profit_e',
    'rd_expense_fetched':        'rd_expense_e',
    'interest_expense_fetched':  'interest_expense_e',
})
metric_cols = ['current_liab_e','total_debt_e','net_profit_e','rd_expense_e','interest_expense_e']
fa = (fa.groupby(['ticker','year'], as_index=False)[metric_cols]
        .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan))

# Wind панель уже в миллионах CNY; akshare возвращает raw CNY -> делим на 1e6.
SCALE = 1e6
for c in metric_cols:
    fa[c] = fa[c] / SCALE

# Delisted tickers (long → wide). Берём только net_income (cached Wind values уже в млн).
fd = pd.read_csv(ENRICH_DELISTED)
fd = (fd[fd.metric == 'net_income']
        .pivot_table(index=['ticker','year'], values='value', aggfunc='first')
        .reset_index()
        .rename(columns={'value':'net_profit_e'}))

enrich = pd.concat([fa, fd[['ticker','year','net_profit_e']]], ignore_index=True)
enrich = enrich.drop_duplicates(['ticker','year'], keep='first')

panel = panel.merge(enrich, on=['ticker','year'], how='left')

# Coalesce: enrichment-значение приоритетно; Wind-значение остаётся, если enrichment пуст.
def _coalesce(primary, fallback):
    return panel[primary].combine_first(panel[fallback])

panel['net_profit']       = _coalesce('net_profit_e', 'net_profit')
panel['current_liab']     = _coalesce('current_liab_e', 'current_liab')
panel['total_debt']       = _coalesce('total_debt_e', 'total_debt')
panel['rd_expense']       = _coalesce('rd_expense_e', 'rd_expense')
panel['interest_expense'] = _coalesce('interest_expense_e', 'interest_expense')

panel = panel.drop(columns=[c for c in panel.columns if c.endswith('_e')])

print(f'NaN по колонкам (%) — ПОСЛЕ енричмента (до ffill):')
print((panel[CORE_COLS].isna().mean()*100).round(1).to_string())

# ===== CELL 13 =====# 1) Drop 2025 (отчётность ещё не закрыта)
before = panel.shape
panel = panel[panel.year != 2025].copy()
print(f'Drop 2025: {before} -> {panel.shape}')

# 2) ffill+bfill по компании.
#    NB: NaN тут означает «значения нет в источнике» — лечим из соседних лет.
#    Реальный ноль (например, у компании буквально нет долга/R&D в данный год) приходит как 0,
#    а не как NaN, и ffill его не трогает (он не пропуск).
panel = panel.sort_values(['ticker','year']).reset_index(drop=True)
panel[CORE_COLS] = (panel.groupby('ticker', group_keys=False)[CORE_COLS]
                        .apply(lambda g: g.ffill().bfill()))
print('После ffill+bfill NaN %:')
print((panel[CORE_COLS].isna().mean()*100).round(1).to_string())

# ===== CELL 14 =====# 3) Target=0: drop rows missing CORE_BACKBONE (без них фичи не посчитать).
#    SPARSE_ZERO — метрики, отсутствие которых = «у компании этого реально нет» -> 0.
CORE_BACKBONE = ['total_revenue', 'total_assets', 'total_equity',
                 'current_assets', 'current_liab', 'net_profit']
SPARSE_ZERO   = [c for c in CORE_COLS if c not in CORE_BACKBONE]

active   = panel[panel.target == 0].copy()
defaults = panel[panel.target == 1].copy()

before_a = len(active)
active = active.dropna(subset=CORE_BACKBONE)
print(f'Target=0: drop rows missing backbone: {before_a:,} -> {len(active):,}')

active[SPARSE_ZERO] = active[SPARSE_ZERO].fillna(0)
defaults[CORE_COLS] = defaults[CORE_COLS].fillna(0)

cleaned = pd.concat([active, defaults], ignore_index=True)
cleaned = cleaned.sort_values(['target','ticker','year'], ascending=[False,True,True]).reset_index(drop=True)

print(f'\nAfter cleaning: {cleaned.shape}')
print(f'NaN в core: {cleaned[CORE_COLS].isna().sum().sum()}')

# ===== CELL 15 =====# --- Контроль классов ---
print('=== Unique companies ===')
print(cleaned.groupby('source_class')['ticker'].nunique().to_string())

print('\n=== Target на уровне строк-лет ===')
print(cleaned['target'].value_counts().rename({0:'Target=0',1:'Target=1'}).to_string())

print('\n=== Unique companies по Target ===')
t1 = cleaned[cleaned.target==1].ticker.nunique()
t0 = cleaned[cleaned.target==0].ticker.nunique()
print(f'  Target=1: {t1}')
print(f'  Target=0: {t0}')
print(f'  imbalance (rows) ≈ 1:{(cleaned.target==0).sum()//max((cleaned.target==1).sum(),1)}')

print('\n=== Годы ===', sorted(cleaned.year.unique()))

# ===== CELL 16 =====# TARGET_WINDOW_V2 (K=2)
# Event-based target with K-year horizon window:
# target=1 on the last K years of each DEFAULT company (source_class='default_delisted').
# Strategic delistings (source_class='strategic_delisted') and active companies remain target=0.
K_HORIZON = 2

# default_company = entity-level flag (всегда 1 для 11 дефолтов, 0 иначе)
cleaned['default_company'] = (cleaned['source_class'] == 'default_delisted').astype(int)

# Пересчёт target: 1 только на последних K годах данных дефолтной компании
def_mask = cleaned['default_company'] == 1
last_by_ticker = (
    cleaned.loc[def_mask]
    .groupby('ticker')['year'].max()
    .rename('_last_year')
)
cleaned = cleaned.merge(last_by_ticker, on='ticker', how='left')
cleaned['target'] = (
    def_mask
    & (cleaned['year'] >= cleaned['_last_year'] - (K_HORIZON - 1))
    & (cleaned['year'] <= cleaned['_last_year'])
).astype(int)
cleaned = cleaned.drop(columns='_last_year')

# --- Save ---
OUT = PROCESSED / 'cn_panel_enriched.csv'
cleaned.to_csv(OUT, index=False, encoding='utf-8-sig')

n_pos = int((cleaned.target == 1).sum())
n_neg = int((cleaned.target == 0).sum())
n_pos_companies = int(cleaned.loc[cleaned.target == 1, 'ticker'].nunique())
n_def_companies = int(cleaned['default_company'].sum() and cleaned.loc[cleaned.default_company == 1, 'ticker'].nunique() or 0)
ratio = n_neg / max(n_pos, 1)
print(f'OK Saved: {OUT}')
print(f'   size: {OUT.stat().st_size/1024/1024:.1f} MB')
print(f'   shape: {cleaned.shape}')
print(f'   target=1 rows: {n_pos} (окно K={K_HORIZON}) от {n_pos_companies} компаний')
print(f'   default_company=1: {n_def_companies} компаний (entity-level)')
print(f'   imbalance (rows) ~= 1:{int(ratio)}; recommended class_weight = {{0:1, 1:{ratio:.0f}}}')
