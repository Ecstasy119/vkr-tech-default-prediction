"""
Russia — single-year target (K=1) sensitivity analysis.

Replicates 20_russia_eda_and_models.ipynb pipeline but with K=1 window
(is_bankrupt = 1 only for the LAST reporting year of each bankrupt company).

Outputs go to reports/russia_k1/ so the main K=2 artefacts remain untouched.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from xgboost import XGBClassifier
import shap

RNG = 42
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data' / 'processed'
REPORTS = ROOT / 'reports' / 'russia_k1'
REPORTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 110


def save_fig(fig, name):
    path = REPORTS / f'{name}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {path}')


_lines = ['# Russia - K=1 sensitivity (single year before bankruptcy)\n']


def log(msg=''):
    print(msg)
    _lines.append(msg)


# --- 1. Load ---------------------------------------------------------------
df = pd.read_csv(PROCESSED / 'ru_panel_cleaned.csv', encoding='utf-8-sig')
print('Loaded:', df.shape)

ID_COL = 'Регистрационный номер'
TARGET = 'is_bankrupt'

# --- 2. Rebuild target with K=1 (last year of each bankrupt company) ------
bank_mask = df['bankrupt_company'] == 1
last_year = (df.loc[bank_mask].groupby(ID_COL)['year'].max().rename('_last_year'))
df = df.merge(last_year, on=ID_COL, how='left')
df[TARGET] = ((df['bankrupt_company'] == 1) & (df['year'] == df['_last_year'])).astype(int)

n_pos = int(df[TARGET].sum())
n_neg = int((df[TARGET] == 0).sum())
n_bank_comp = int(bank_mask.groupby(df[ID_COL]).any().sum())
print(f'K=1 target: {n_pos} positives / {n_neg} negatives (bankrupt companies: {n_bank_comp})')
log(f'* Rows (company-year): **{len(df):,}**')
log(f'* Bankrupt companies:  **{n_bank_comp}**')
log(f'* Positives (K=1):     **{n_pos}**  (share {n_pos/len(df)*100:.3f}%)')

# --- 3. Build features (exact copy from 20_russia) -------------------------
A = 'Активы  всего'
CA = 'Оборотные активы'
CASH = 'Денежные средства и денежные эквиваленты'
INT_A = 'Нематериальные активы'
EQ = 'Капитал и резервы'
LT_L = 'Долгосрочные обязательства'
ST_L = 'Краткосрочные обязательства'
REV = 'Выручка'
EBIT = 'EBIT'
NI = 'Чистая прибыль (убыток)'
INTEREST = 'Проценты к уплате'
CFO = 'Сальдо денежных потоков от текущих операций'


def safe_div(a, b):
    b = b.replace(0, np.nan)
    return a / b


p = df.copy()
total_debt = p[LT_L] + p[ST_L]

p['current_ratio'] = safe_div(p[CA], p[ST_L])
p['cash_to_assets'] = safe_div(p[CASH], p[A])
p['cash_to_cl'] = safe_div(p[CASH], p[ST_L])
p['wc_to_assets'] = safe_div(p[CA] - p[ST_L], p[A])

p['intangibles_to_assets'] = safe_div(p[INT_A], p[A])

p['debt_to_assets'] = safe_div(total_debt, p[A])
p['debt_to_equity'] = safe_div(total_debt, p[EQ])
p['lt_debt_to_assets'] = safe_div(p[LT_L], p[A])
p['interest_coverage'] = safe_div(p[EBIT], p[INTEREST])

p['roa'] = safe_div(p[NI], p[A])
p['net_margin'] = safe_div(p[NI], p[REV])
p['operating_margin'] = safe_div(p[EBIT], p[REV])
p['cfo_to_assets'] = safe_div(p[CFO], p[A])

p['log_assets'] = np.log1p(p[A].clip(lower=0))
p['log_revenue'] = np.log1p(p[REV].clip(lower=0))

FEATURES = [
    'current_ratio', 'cash_to_assets', 'cash_to_cl', 'wc_to_assets',
    'intangibles_to_assets',
    'debt_to_assets', 'debt_to_equity', 'lt_debt_to_assets', 'interest_coverage',
    'roa', 'net_margin', 'operating_margin', 'cfo_to_assets',
    'log_assets', 'log_revenue',
]
FEATURE_GROUPS = {
    'Liquidity': ['current_ratio', 'cash_to_assets', 'cash_to_cl', 'wc_to_assets'],
    'Innovation': ['intangibles_to_assets'],
    'Leverage': ['debt_to_assets', 'debt_to_equity', 'lt_debt_to_assets', 'interest_coverage'],
    'Profitability': ['roa', 'net_margin', 'operating_margin', 'cfo_to_assets'],
    'Size': ['log_assets', 'log_revenue'],
}

p[FEATURES] = p[FEATURES].replace([np.inf, -np.inf], np.nan)
for c in FEATURES:
    lo, hi = p[c].quantile([0.01, 0.99])
    p[c] = p[c].clip(lo, hi)
p[FEATURES] = p[FEATURES].fillna(p[FEATURES].median(numeric_only=True))

# --- 4. Group-aware 80/20 split --------------------------------------------
X = p[FEATURES].values
y = p[TARGET].values
groups = p[ID_COL].values

company_label = p.groupby(ID_COL)[TARGET].max()
comp_b = np.array(company_label[company_label == 1].index.values, copy=True)
comp_a = np.array(company_label[company_label == 0].index.values, copy=True)
_rng = np.random.default_rng(RNG)
_rng.shuffle(comp_b)
_rng.shuffle(comp_a)


def _split(arr, frac=0.2):
    n = int(round(len(arr) * frac))
    return arr[n:], arr[:n]


train_b, test_b = _split(comp_b)
train_a, test_a = _split(comp_a)
train_ids = set(train_b) | set(train_a)
test_ids = set(test_b) | set(test_a)

m_tr = p[ID_COL].isin(train_ids).values
m_te = p[ID_COL].isin(test_ids).values
X_tr, y_tr, g_tr = X[m_tr], y[m_tr], groups[m_tr]
X_te, y_te, g_te = X[m_te], y[m_te], groups[m_te]

POS_W = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
log(f'* POS_WEIGHT (train) = **{POS_W:.1f}**')
log(f'* Train defaults: **{int(y_tr.sum())}** / test defaults: **{int(y_te.sum())}**')

# --- 5. Fit three models ---------------------------------------------------
logit = Pipeline([
    ('sc', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight={0: 1, 1: POS_W},
                               solver='liblinear', random_state=RNG)),
])
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=3,
    class_weight={0: 1, 1: POS_W}, n_jobs=-1, random_state=RNG,
)
xgb = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    scale_pos_weight=POS_W,
    eval_metric='aucpr', tree_method='hist', random_state=RNG, n_jobs=-1,
)
models = {'Logistic Regression': logit, 'Random Forest': rf, 'XGBoost': xgb}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    print(f'fit {name}')


def scores(model, Xa, ya):
    if hasattr(model, 'predict_proba'):
        pr = model.predict_proba(Xa)[:, 1]
    else:
        pr = model.decision_function(Xa)
    return roc_auc_score(ya, pr), average_precision_score(ya, pr)


# --- 6. Single-split metrics ----------------------------------------------
rows = []
for name, m in models.items():
    tr_r, tr_p = scores(m, X_tr, y_tr)
    te_r, te_p = scores(m, X_te, y_te)
    rows.append({'Model': name,
                 'ROC-AUC train': tr_r, 'ROC-AUC test': te_r, 'ΔROC': tr_r - te_r,
                 'PR-AUC train': tr_p, 'PR-AUC test': te_p, 'ΔPR': tr_p - te_p})
res = pd.DataFrame(rows).set_index('Model').round(4)
res.to_csv(REPORTS / 'ru_k1_h1_metrics.csv', encoding='utf-8-sig')
print('\nSingle split:')
print(res)

# --- 7. 5-fold Stratified Group K-Fold CV ----------------------------------
skgf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
cv_rows = []
for name, proto in models.items():
    r_tr_l, p_tr_l, r_te_l, p_te_l = [], [], [], []
    for tr_idx, te_idx in skgf.split(X, y, groups=groups):
        yt = y[tr_idx]
        pw = float((yt == 0).sum() / max((yt == 1).sum(), 1))
        mdl = clone(proto)
        if isinstance(mdl, Pipeline):
            mdl.set_params(clf__class_weight={0: 1, 1: pw})
        elif isinstance(mdl, RandomForestClassifier):
            mdl.set_params(class_weight={0: 1, 1: pw})
        elif isinstance(mdl, XGBClassifier):
            mdl.set_params(scale_pos_weight=pw)
        mdl.fit(X[tr_idx], yt)
        a, b = scores(mdl, X[tr_idx], yt)
        c, d = scores(mdl, X[te_idx], y[te_idx])
        r_tr_l.append(a); p_tr_l.append(b); r_te_l.append(c); p_te_l.append(d)
    cv_rows.append({'Model': name,
                    'ROC-AUC test mean': np.mean(r_te_l),
                    'ROC-AUC test std': np.std(r_te_l),
                    'PR-AUC test mean': np.mean(p_te_l),
                    'PR-AUC test std': np.std(p_te_l),
                    'ΔROC (train−test)': np.mean(r_tr_l) - np.mean(r_te_l),
                    'ΔPR (train−test)': np.mean(p_tr_l) - np.mean(p_te_l)})
cv = pd.DataFrame(cv_rows).set_index('Model').round(4)
cv.to_csv(REPORTS / 'ru_k1_h1_cv_metrics.csv', encoding='utf-8-sig')
print('\nCV:')
print(cv)

# --- 8. Log H1 verdict -----------------------------------------------------
best_roc = res['ROC-AUC test'].idxmax()
best_pr = res['PR-AUC test'].idxmax()
log('\n## H1 (K=1) - small-sample overfitting signature')
log(f'* Best test ROC-AUC: **{best_roc}** ({res.loc[best_roc, "ROC-AUC test"]:.4f})')
log(f'* Best test PR-AUC:  **{best_pr}**  ({res.loc[best_pr, "PR-AUC test"]:.4f})')
log(f'* Logit test ROC={res.loc["Logistic Regression","ROC-AUC test"]:.4f}, '
    f'PR={res.loc["Logistic Regression","PR-AUC test"]:.4f}, '
    f'ΔROC={res.loc["Logistic Regression","ΔROC"]:+.4f}')
log(f'* Max ensemble ΔROC (train−test) = **{res.loc[["Random Forest","XGBoost"],"ΔROC"].max():+.4f}** '
    '(overfit signal)')
log('')
log('**CV (mean ± std):**')
for n in cv.index:
    log(f'* {n}: ROC {cv.loc[n,"ROC-AUC test mean"]:.3f}±{cv.loc[n,"ROC-AUC test std"]:.3f}, '
        f'PR {cv.loc[n,"PR-AUC test mean"]:.3f}±{cv.loc[n,"PR-AUC test std"]:.3f}')

# --- 9. SHAP on best ensemble by PR-AUC -----------------------------------
ens_pr = res.loc[['Random Forest', 'XGBoost'], 'PR-AUC test']
best_name = ens_pr.idxmax()
best_model = models[best_name]
print(f'\nBest ensemble by PR-AUC: {best_name}')

explainer = shap.TreeExplainer(best_model)
sv = explainer.shap_values(X_te)
if isinstance(sv, list):
    sv = sv[1]
elif hasattr(sv, 'ndim') and sv.ndim == 3:
    sv = sv[:, :, 1]

mean_abs = np.abs(sv).mean(axis=0)
fi = pd.Series(mean_abs, index=FEATURES).sort_values(ascending=False)
fi.round(4).to_csv(REPORTS / 'ru_k1_shap_feature_importance.csv',
                    header=['mean_abs_shap'], encoding='utf-8-sig')

group_sum = {g: fi[cols].sum() for g, cols in FEATURE_GROUPS.items()}
group_df = pd.Series(group_sum).sort_values(ascending=False).round(4)
group_df.to_csv(REPORTS / 'ru_k1_shap_group_importance.csv',
                header=['sum_abs_shap'], encoding='utf-8-sig')

liq_inn = group_df.get('Liquidity', 0) + group_df.get('Innovation', 0)
lev = group_df.get('Leverage', 0)

log('\n## H2 (K=1) - Liquidity+Innovation vs Leverage (SHAP)')
log(f'* Best ensemble (PR-AUC): **{best_name}**')
log(f'* Top features: {", ".join(fi.head(5).index.tolist())}')
log('* Group Sum|SHAP|:')
for g, v in group_df.items():
    log(f'  * {g}: **{v:.4f}**')
log(f'* Liquidity+Innovation = **{liq_inn:.4f}**')
log(f'* Leverage            = **{lev:.4f}**')
if lev > 0:
    log(f'* Ratio = **{liq_inn/lev:.2f}x**')
log('')
if liq_inn > lev:
    log('[OK] **H2 (K=1) confirmed**: Liquidity+Innovation dominates Leverage under single-year window.')
else:
    log('[X] H2 (K=1) NOT confirmed: Leverage dominates under single-year window.')

# --- 10. Plot group importance --------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(['Liquidity + Innovation', 'Leverage'],
              [liq_inn, lev], color=['#2E75B6', '#C00000'])
ax.set_ylabel('Sum mean |SHAP|')
ax.set_title('Russia K=1: H2 group importance')
for b, v in zip(bars, [liq_inn, lev]):
    ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}', ha='center', va='bottom')
plt.tight_layout()
save_fig(fig, 'ru_k1_h2_group_importance')

# --- 11. Comparison table K=1 vs K=2 --------------------------------------
try:
    k2_m = pd.read_csv(ROOT / 'reports/russia/ru_h1_metrics.csv', index_col=0)
    k2_g = pd.read_csv(ROOT / 'reports/russia/ru_shap_group_importance.csv', index_col=0)
    cmp_rows = []
    for m in res.index:
        cmp_rows.append({
            'Model': m,
            'ROC test K=1': res.loc[m, 'ROC-AUC test'],
            'ROC test K=2': k2_m.loc[m, 'ROC-AUC test'],
            'PR test K=1': res.loc[m, 'PR-AUC test'],
            'PR test K=2': k2_m.loc[m, 'PR-AUC test'],
            'ΔROC K=1': res.loc[m, 'ΔROC'],
            'ΔROC K=2': k2_m.loc[m, 'ΔROC'],
        })
    cmp = pd.DataFrame(cmp_rows).set_index('Model').round(4)
    cmp.to_csv(REPORTS / 'ru_k1_vs_k2_metrics.csv', encoding='utf-8-sig')
    print('\nK=1 vs K=2 metrics:')
    print(cmp)

    gcmp = group_df.rename('K=1').to_frame().join(k2_g.rename(columns={'sum_abs_shap': 'K=2'}))
    gcmp['delta'] = gcmp['K=1'] - gcmp['K=2']
    gcmp.round(4).to_csv(REPORTS / 'ru_k1_vs_k2_groups.csv', encoding='utf-8-sig')
    log('\n## K=1 vs K=2 comparison')
    log('Group Sum|SHAP| (K=1 vs K=2):')
    for g in gcmp.index:
        log(f'* {g}: K=1 {gcmp.loc[g,"K=1"]:.3f}  vs  K=2 {gcmp.loc[g,"K=2"]:.3f}  '
            f'(Δ={gcmp.loc[g,"delta"]:+.3f})')
except Exception as e:
    print('K=2 comparison skipped:', e)

# --- 12. Save summary ------------------------------------------------------
(REPORTS / 'ru_k1_summary.md').write_text('\n'.join(_lines), encoding='utf-8')
print('\nAll artefacts saved to:', REPORTS)
for f in sorted(REPORTS.iterdir()):
    print(f'  {f.name}  ({f.stat().st_size/1024:.1f} KB)')
