"""Idempotent helper: rebuild 30 + 40 China notebooks with the enriched-data pipeline.

Re-runnable: locates cells by stable content tokens; never inserts a duplicate cell.
Run with: python notebooks/_refactor_china.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

def _src(s): return [l + '\n' for l in s.rstrip('\n').split('\n')]

def _new_code(src):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': _src(src)}

def _new_md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': _src(src)}

def _set_code(nb, idx, src):
    nb['cells'][idx]['source'] = _src(src)
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None
    nb['cells'][idx]['cell_type'] = 'code'
    if 'metadata' not in nb['cells'][idx]: nb['cells'][idx]['metadata'] = {}

def _find_cell(nb, kind, marker):
    """Return index of first cell whose source contains `marker`. Raises if missing."""
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == kind and marker in ''.join(c['source']):
            return i
    raise KeyError(f'{kind}: {marker!r}')

def _has_cell(nb, kind, marker):
    try:
        _find_cell(nb, kind, marker)
        return True
    except KeyError:
        return False

# ============================================================================
# 30_china_load_and_clean.ipynb
# ============================================================================
nb = json.load(open(HERE / '30_china_load_and_clean.ipynb', encoding='utf-8'))

# (1) Patch WIND_METRIC_MAP cell — drop st_borrow proxy.
WIND_CELL = """\
WIND_METRIC_MAP = {
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
        m = re.search(r'\\[rptDate\\](\\d{4})', mtxt)
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
"""
i_wind = _find_cell(nb, 'code', 'WIND_METRIC_MAP')
_set_code(nb, i_wind, WIND_CELL)

# (2) Merge cell — drop placeholders, define real CORE_COLS incl. total_debt.
MERGE_CELL = """\
CORE_COLS = [
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
print(f'\\nПо source_class:')
print(panel.groupby('source_class')['ticker'].nunique().to_string())
print(f'\\nNaN по колонкам (%) — ДО енричмента akshare:')
print((panel[CORE_COLS].isna().mean()*100).round(1).to_string())
"""
i_merge = _find_cell(nb, 'code', 'CORE_COLS = [')
_set_code(nb, i_merge, MERGE_CELL)

# (3) Enrichment cells AFTER "## 5. Supervisor imputation rules" markdown.
#     Idempotent: insert if missing, OVERWRITE content if already present.
ENRICH_MARKER = '## 5a. Akshare-enrichment'
md_enrich = """\
## 5a. Akshare-enrichment: real net_profit / current_liab / total_debt / interest_expense / rd_expense

В Wind для IT-сектора ключевые метрики приходят пустыми или прокси:
* `net_profit` — отсутствует;
* `current_liab` — раньше использовали `st_borrow` (только short-term borrowing), это занижало пассивы;
* `total_debt` — отсутствует.

Заранее (`scripts` outside notebook) подняли реальные значения через **akshare** для всех 319 A-share активных тикеров (`china_fetched_metrics.csv`) и отдельно для делистнутых иностранных тикеров (`delisted_fetched_metrics_raw.csv`, спарсено из cached Wind formulas, `data_only=True`).

Здесь только **мерджим** их в основную панель — приоритет: значение из enrichment-CSV, если оно непустое; иначе оставляем то, что было от Wind. Akshare возвращает raw CNY → масштабируем в млн CNY (Wind-уровень).
"""
code_enrich = """\
# --- Load enrichment CSVs ---
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
"""
if _has_cell(nb, 'markdown', ENRICH_MARKER):
    i_md = _find_cell(nb, 'markdown', ENRICH_MARKER)
    nb['cells'][i_md]['source'] = _src(md_enrich)
    # Find adjacent code cell (the next code cell after the enrichment markdown).
    for j in range(i_md + 1, len(nb['cells'])):
        if nb['cells'][j]['cell_type'] == 'code':
            _set_code(nb, j, code_enrich)
            break
else:
    i_super = _find_cell(nb, 'markdown', '## 5. Supervisor imputation rules')
    nb['cells'].insert(i_super + 1, _new_md(md_enrich))
    nb['cells'].insert(i_super + 2, _new_code(code_enrich))

# (4) Patch "Drop 2025 + ffill" cell.
FFILL_CELL = """\
# 1) Drop 2025 (отчётность ещё не закрыта)
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
"""
i_ffill = _find_cell(nb, 'code', 'Drop 2025')
_set_code(nb, i_ffill, FFILL_CELL)

# (5) Patch "CORE_BACKBONE" cell.
BACKBONE_CELL = """\
# 3) Target=0: drop rows missing CORE_BACKBONE (без них фичи не посчитать).
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

print(f'\\nAfter cleaning: {cleaned.shape}')
print(f'NaN в core: {cleaned[CORE_COLS].isna().sum().sum()}')
"""
i_back = _find_cell(nb, 'code', 'CORE_BACKBONE')
_set_code(nb, i_back, BACKBONE_CELL)

# (6) Patch save cell — write to cn_panel_enriched.csv.
SAVE_CELL = """\
OUT = PROCESSED / 'cn_panel_enriched.csv'
cleaned.to_csv(OUT, index=False, encoding='utf-8-sig')

# Imbalance + class_weight рекомендация (как в russia_load_and_clean).
n_pos = int((cleaned.target == 1).sum())
n_neg = int((cleaned.target == 0).sum())
ratio = n_neg / max(n_pos, 1)
print(f'OK Saved: {OUT}')
print(f'   size: {OUT.stat().st_size/1024/1024:.1f} MB')
print(f'   shape: {cleaned.shape}')
print(f'   imbalance (rows) ~= 1:{int(ratio)}; recommended class_weight = {{0:1, 1:{ratio:.0f}}}')
"""
i_save = _find_cell(nb, 'code', 'cn_panel_')
_set_code(nb, i_save, SAVE_CELL)

# Patch intro/outro markdown to reference the new artefact name.
for i in range(len(nb['cells'])):
    if nb['cells'][i]['cell_type'] == 'markdown':
        s = ''.join(nb['cells'][i]['source'])
        if 'cn_panel_cleaned.csv' in s:
            nb['cells'][i]['source'] = _src(s.replace('cn_panel_cleaned.csv', 'cn_panel_enriched.csv'))

json.dump(nb, open(HERE / '30_china_load_and_clean.ipynb', 'w', encoding='utf-8'),
          indent=1, ensure_ascii=False)
print('OK 30_china_load_and_clean.ipynb rebuilt')

# ============================================================================
# 40_china_eda_and_models.ipynb
# ============================================================================
nb = json.load(open(HERE / '40_china_eda_and_models.ipynb', encoding='utf-8'))

# (1) Patch load+imports cell — read enriched, add StratifiedGroupKFold import.
LOAD_CELL = """\
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

from xgboost import XGBClassifier
import shap

RNG = 42
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 110

PROCESSED = Path('../data/processed')
REPORTS = Path('../reports/china')
REPORTS.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name):
    path = REPORTS / f'{name}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    print(f'  saved -> {path}')

_report_lines = ['# China — Stage-1 (TTC) results\\n']
def log(msg=''):
    print(msg)
    _report_lines.append(msg)

df = pd.read_csv(PROCESSED / 'cn_panel_enriched.csv', encoding='utf-8-sig')
print('Shape:', df.shape)
print('Tickers per target:', df.groupby('target')['ticker'].nunique().to_dict())
df.head(3)
"""
i_load = _find_cell(nb, 'code', 'cn_panel_')
_set_code(nb, i_load, LOAD_CELL)

# (2) Patch feature-engineering cell — use total_debt for leverage, keep total_liab for LT-proxy.
FEAT_CELL = """\
TARGET = 'target'
ID_COL = 'ticker'

A, CA, CASH, INT_A = 'total_assets', 'current_assets', 'cash', 'intangibles'
EQ, LIAB, ST_L = 'total_equity', 'total_liab', 'current_liab'
TOTD = 'total_debt'      # реальные процентные долги (akshare-енричмент)
REV, EBIT, NI = 'total_revenue', 'ebit', 'net_profit'
INTEREST, CFO, RD = 'interest_expense', 'cfo', 'rd_expense'

def safe_div(a, b):
    b = b.replace(0, np.nan)
    return a / b

panel = df.copy()
lt_liab = (panel[LIAB] - panel[ST_L]).clip(lower=0)

panel['current_ratio']         = safe_div(panel[CA], panel[ST_L])
panel['cash_to_assets']        = safe_div(panel[CASH], panel[A])
panel['cash_to_cl']            = safe_div(panel[CASH], panel[ST_L])
panel['wc_to_assets']          = safe_div(panel[CA] - panel[ST_L], panel[A])

panel['intangibles_to_assets'] = safe_div(panel[INT_A], panel[A])
panel['rd_to_revenue']         = safe_div(panel[RD], panel[REV])

# Leverage: реальные процентные обязательства, а не total_liab.
panel['debt_to_assets']        = safe_div(panel[TOTD], panel[A])
panel['debt_to_equity']        = safe_div(panel[TOTD], panel[EQ])
panel['lt_debt_to_assets']     = safe_div(lt_liab, panel[A])
panel['interest_coverage']     = safe_div(panel[EBIT], panel[INTEREST])

panel['roa']              = safe_div(panel[NI], panel[A])
panel['net_margin']       = safe_div(panel[NI], panel[REV])
panel['operating_margin'] = safe_div(panel[EBIT], panel[REV])
panel['cfo_to_assets']    = safe_div(panel[CFO], panel[A])

panel['log_assets']  = np.log1p(panel[A].clip(lower=0))
panel['log_revenue'] = np.log1p(panel[REV].clip(lower=0))

FEATURES = [
    'current_ratio', 'cash_to_assets', 'cash_to_cl', 'wc_to_assets',
    'intangibles_to_assets', 'rd_to_revenue',
    'debt_to_assets', 'debt_to_equity', 'lt_debt_to_assets', 'interest_coverage',
    'roa', 'net_margin', 'operating_margin', 'cfo_to_assets',
    'log_assets', 'log_revenue',
]
FEATURE_GROUPS = {
    'Liquidity':     ['current_ratio', 'cash_to_assets', 'cash_to_cl', 'wc_to_assets'],
    'Innovation':    ['intangibles_to_assets', 'rd_to_revenue'],
    'Leverage':      ['debt_to_assets', 'debt_to_equity', 'lt_debt_to_assets', 'interest_coverage'],
    'Profitability': ['roa', 'net_margin', 'operating_margin', 'cfo_to_assets'],
    'Size':          ['log_assets', 'log_revenue'],
}

panel[FEATURES] = panel[FEATURES].replace([np.inf, -np.inf], np.nan)
for c in FEATURES:
    lo, hi = panel[c].quantile([0.01, 0.99])
    panel[c] = panel[c].clip(lo, hi)
panel[FEATURES] = panel[FEATURES].fillna(panel[FEATURES].median(numeric_only=True))

panel[FEATURES + [TARGET]].describe().T.round(3).to_csv(REPORTS / 'cn_feature_stats.csv', encoding='utf-8-sig')
print(f'Фичи: {len(FEATURES)}  |  stats -> reports/china/cn_feature_stats.csv')
panel[FEATURES].describe().T.round(3)
"""
i_feat = _find_cell(nb, 'code', 'FEATURE_GROUPS')
_set_code(nb, i_feat, FEAT_CELL)

# (3) Patch row-level split cell → group-aware split.
SPLIT_CELL = """\
X = panel[FEATURES].values
y = panel[TARGET].values
groups = panel[ID_COL].values

# Company-level stratified split: 80/20 отдельно по default- и active-компаниям.
company_label = panel.groupby(ID_COL)[TARGET].max()
companies_d = np.array(company_label[company_label == 1].index.values, copy=True)
companies_a = np.array(company_label[company_label == 0].index.values, copy=True)

_rng = np.random.default_rng(RNG)
_rng.shuffle(companies_d)
_rng.shuffle(companies_a)

def _split_ids(arr, test_frac=0.2):
    n_test = max(1, int(round(len(arr) * test_frac)))
    return arr[n_test:], arr[:n_test]

train_d, test_d = _split_ids(companies_d, 0.20)
train_a, test_a = _split_ids(companies_a, 0.20)

train_ids = set(train_d) | set(train_a)
test_ids  = set(test_d)  | set(test_a)

mask_train = panel[ID_COL].isin(train_ids).values
mask_test  = panel[ID_COL].isin(test_ids).values

X_train, y_train = X[mask_train], y[mask_train]
X_test,  y_test  = X[mask_test],  y[mask_test]
g_train, g_test  = groups[mask_train], groups[mask_test]

print(f'Train: {X_train.shape},  defaults = {y_train.sum()}  ({y_train.mean()*100:.2f}%)')
print(f'Test:  {X_test.shape},   defaults = {y_test.sum()}   ({y_test.mean()*100:.2f}%)')
print(f'Companies — train: {len(train_ids):,}, test: {len(test_ids):,}, '
      f'overlap: {len(train_ids & test_ids)} (должно быть 0)')
print(f'Default companies — train: {len(train_d)}, test: {len(test_d)}')
"""
# Find the OLD-style row split (contains a `train_test_split(` call).
def _find_call(nb, substr):
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c['source'])
        if substr in s:
            return i
    raise KeyError(substr)
try:
    i_split = _find_call(nb, 'train_test_split(')
except KeyError:
    i_split = _find_cell(nb, 'code', 'companies_d')
_set_code(nb, i_split, SPLIT_CELL)

# (4) Patch model-training cell — use class_weight={0:1, 1:POS_WEIGHT}.
TRAIN_CELL = """\
POS_WEIGHT = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
print(f'POS_WEIGHT (n_neg / n_pos на train) = {POS_WEIGHT:.1f}')

logit = Pipeline([
    ('sc', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight={0: 1, 1: POS_WEIGHT},
                               solver='liblinear', random_state=RNG)),
])
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    class_weight={0: 1, 1: POS_WEIGHT}, n_jobs=-1, random_state=RNG,
)
xgb = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    scale_pos_weight=POS_WEIGHT,
    eval_metric='aucpr', tree_method='hist', random_state=RNG, n_jobs=-1,
)

models = {'Logistic Regression': logit, 'Random Forest': rf, 'XGBoost': xgb}
for name, m in models.items():
    m.fit(X_train, y_train)
    print(f'OK trained {name}')
"""
i_train = _find_cell(nb, 'code', 'POS_WEIGHT')
_set_code(nb, i_train, TRAIN_CELL)

# (5) Patch metrics cell — single-split + StratifiedGroupKFold CV (3-fold given 11 default companies).
METRIC_CELL = """\
from sklearn.base import clone

def scores(model, X, y):
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)[:, 1]
    else:
        p = model.decision_function(X)
    return roc_auc_score(y, p), average_precision_score(y, p)

# --- (a) Single-split метрики (group-aware 80/20) ---
rows = []
for name, m in models.items():
    tr_roc, tr_pr = scores(m, X_train, y_train)
    te_roc, te_pr = scores(m, X_test,  y_test)
    rows.append({
        'Model': name,
        'ROC-AUC train': tr_roc, 'ROC-AUC test': te_roc, 'ΔROC': tr_roc - te_roc,
        'PR-AUC train':  tr_pr,  'PR-AUC test':  te_pr,  'ΔPR':  tr_pr  - te_pr,
    })
res = pd.DataFrame(rows).set_index('Model').round(4)
res.to_csv(REPORTS / 'cn_h1_metrics.csv', encoding='utf-8-sig')
print('Single split (group-aware):')
print(res)

# --- (b) Stratified Group K-Fold CV ---
# При ~11 default-компаниях используем 3-fold, чтобы в каждом фолде остался хотя бы 1 default.
n_pos_companies = panel.loc[panel[TARGET]==1, ID_COL].nunique()
n_splits = min(5, max(2, n_pos_companies // 3))
print(f'Default companies: {n_pos_companies}, n_splits = {n_splits}')

skgf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
cv_rows = []
for name, m_proto in models.items():
    roc_tr_l, pr_tr_l, roc_te_l, pr_te_l = [], [], [], []
    for tr_idx, te_idx in skgf.split(X, y, groups=groups):
        yt = y[tr_idx]
        pw = float((yt == 0).sum() / max((yt == 1).sum(), 1))
        mdl = clone(m_proto)
        if isinstance(mdl, Pipeline):
            mdl.set_params(clf__class_weight={0: 1, 1: pw})
        elif isinstance(mdl, RandomForestClassifier):
            mdl.set_params(class_weight={0: 1, 1: pw})
        elif isinstance(mdl, XGBClassifier):
            mdl.set_params(scale_pos_weight=pw)
        mdl.fit(X[tr_idx], yt)
        r_tr, p_tr = scores(mdl, X[tr_idx], yt)
        r_te, p_te = scores(mdl, X[te_idx], y[te_idx])
        roc_tr_l.append(r_tr); pr_tr_l.append(p_tr)
        roc_te_l.append(r_te); pr_te_l.append(p_te)
    cv_rows.append({
        'Model': name,
        'ROC-AUC test mean': np.mean(roc_te_l),
        'ROC-AUC test std':  np.std(roc_te_l),
        'PR-AUC test mean':  np.mean(pr_te_l),
        'PR-AUC test std':   np.std(pr_te_l),
        'ΔROC (train−test)': np.mean(roc_tr_l) - np.mean(roc_te_l),
        'ΔPR (train−test)':  np.mean(pr_tr_l)  - np.mean(pr_te_l),
    })
cv = pd.DataFrame(cv_rows).set_index('Model').round(4)
cv.to_csv(REPORTS / 'cn_h1_cv_metrics.csv', encoding='utf-8-sig')
print(f'\\n{n_splits}-fold Stratified Group K-Fold CV:')
print(cv)
res
"""
i_metric = _find_cell(nb, 'code', 'ROC-AUC train')
_set_code(nb, i_metric, METRIC_CELL)

json.dump(nb, open(HERE / '40_china_eda_and_models.ipynb', 'w', encoding='utf-8'),
          indent=1, ensure_ascii=False)
print('OK 40_china_eda_and_models.ipynb rebuilt')
