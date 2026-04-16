# ===== CELL 1 =====
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
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
REPORTS = Path('../reports/russia')
REPORTS.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name):
    path = REPORTS / f'{name}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    print(f'  saved → {path}')

# Сюда копим текстовые выводы для итогового markdown-отчёта
_report_lines = ['# Russia — Stage-1 (TTC) results\n']
def log(msg=''):
    print(msg)
    _report_lines.append(msg)

df = pd.read_csv(PROCESSED / 'ru_panel_cleaned.csv', encoding='utf-8-sig')
print('Shape:', df.shape)
df.head(3)

# ===== CELL 3 =====
ID_COL = 'Регистрационный номер'
TARGET = 'is_bankrupt'

A  = 'Активы  всего'
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

panel = df.copy()
total_debt = panel[LT_L] + panel[ST_L]

panel['current_ratio']         = safe_div(panel[CA], panel[ST_L])
panel['cash_to_assets']        = safe_div(panel[CASH], panel[A])
panel['cash_to_cl']            = safe_div(panel[CASH], panel[ST_L])
panel['wc_to_assets']          = safe_div(panel[CA] - panel[ST_L], panel[A])

panel['intangibles_to_assets'] = safe_div(panel[INT_A], panel[A])

panel['debt_to_assets']        = safe_div(total_debt, panel[A])
panel['debt_to_equity']        = safe_div(total_debt, panel[EQ])
panel['lt_debt_to_assets']     = safe_div(panel[LT_L], panel[A])
panel['interest_coverage']     = safe_div(panel[EBIT], panel[INTEREST])

panel['roa']              = safe_div(panel[NI], panel[A])
panel['net_margin']       = safe_div(panel[NI], panel[REV])
panel['operating_margin'] = safe_div(panel[EBIT], panel[REV])
panel['cfo_to_assets']    = safe_div(panel[CFO], panel[A])

panel['log_assets']  = np.log1p(panel[A].clip(lower=0))
panel['log_revenue'] = np.log1p(panel[REV].clip(lower=0))

FEATURES = [
    'current_ratio', 'cash_to_assets', 'cash_to_cl', 'wc_to_assets',
    'intangibles_to_assets',
    'debt_to_assets', 'debt_to_equity', 'lt_debt_to_assets', 'interest_coverage',
    'roa', 'net_margin', 'operating_margin', 'cfo_to_assets',
    'log_assets', 'log_revenue',
]

FEATURE_GROUPS = {
    'Liquidity':     ['current_ratio', 'cash_to_assets', 'cash_to_cl', 'wc_to_assets'],
    'Innovation':    ['intangibles_to_assets'],
    'Leverage':      ['debt_to_assets', 'debt_to_equity', 'lt_debt_to_assets', 'interest_coverage'],
    'Profitability': ['roa', 'net_margin', 'operating_margin', 'cfo_to_assets'],
    'Size':          ['log_assets', 'log_revenue'],
}

# Human-readable labels used in every plot instead of raw underscore names.
FEATURE_LABELS = {
    'current_ratio':         'Current ratio (CA / CL)',
    'cash_to_assets':        'Cash / Total assets',
    'cash_to_cl':            'Cash / Current liabilities',
    'wc_to_assets':          'Working capital / Total assets',
    'intangibles_to_assets': 'Intangibles / Total assets',
    'debt_to_assets':        'Total debt / Total assets',
    'debt_to_equity':        'Total debt / Equity',
    'lt_debt_to_assets':     'Long-term debt / Total assets',
    'interest_coverage':     'Interest coverage (EBIT / Interest)',
    'roa':                   'Return on assets (ROA)',
    'net_margin':            'Net margin (NI / Revenue)',
    'operating_margin':      'Operating margin (EBIT / Revenue)',
    'cfo_to_assets':         'Operating cash flow / Total assets',
    'log_assets':            'log(Total assets)',
    'log_revenue':           'log(Revenue)',
}
FEATURE_LABELS_LIST = [FEATURE_LABELS[f] for f in FEATURES]

panel[FEATURES] = panel[FEATURES].replace([np.inf, -np.inf], np.nan)
for c in FEATURES:
    lo, hi = panel[c].quantile([0.01, 0.99])
    panel[c] = panel[c].clip(lo, hi)
panel[FEATURES] = panel[FEATURES].fillna(panel[FEATURES].median(numeric_only=True))

panel[FEATURES + [TARGET]].describe().T.round(3).to_csv(REPORTS / 'ru_feature_stats.csv', encoding='utf-8-sig')
print(f'Фичи: {len(FEATURES)}  |  stats → reports/russia/ru_feature_stats.csv')
panel[FEATURES].describe().T.round(3)

# ===== CELL 5 =====
vc = panel[TARGET].value_counts()
n_active  = int(vc.get(0, 0))
n_default = int(vc.get(1, 0))
ratio = n_default / (n_active + n_default)
counts = pd.Series({'Active (0)': n_active, 'Default (1)': n_default})

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(counts.index, counts.values, color=['#2E75B6', '#C00000'])
ax.set_title(
    f'Target class balance — Russia\n'
    f'Active vs Default companies (imbalance ≈ 1:{n_active//max(n_default,1)})'
)
ax.set_ylabel('Number of company-year observations')
for b, v in zip(bars, counts.values):
    ax.text(b.get_x()+b.get_width()/2, v, f'{v:,}', ha='center', va='bottom')
plt.tight_layout()
save_fig(fig, '01_class_distribution'); plt.show()

log(f'\n## Class balance\n* Active (0): **{n_active:,}**')
log(f'* Default (1): **{n_default:,}**')
log(f'* Positive share: **{ratio*100:.3f}%**  (≈ 1:{n_active//max(n_default,1)})')
# ===== CELL 7 =====
panel['_lbl'] = panel[TARGET].map({0:'Active', 1:'Default'})
show = ['current_ratio', 'intangibles_to_assets', 'debt_to_assets', 'roa']

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col in zip(axes.ravel(), show):
    sns.violinplot(data=panel, x='_lbl', y=col, ax=ax,
                   palette={'Active':'#2E75B6','Default':'#C00000'}, cut=0)
    ax.set_title(FEATURE_LABELS.get(col, col))
    ax.set_xlabel('Company status'); ax.set_ylabel('Ratio value')
plt.suptitle('Key financial ratios — Active vs Default companies (winsorized 1–99%)', y=1.02)
plt.tight_layout()
save_fig(fig, '02_violin_by_class'); plt.show()

med = panel.groupby('_lbl')[show].median().round(3)
med.to_csv(REPORTS / 'ru_medians_by_class.csv', encoding='utf-8-sig')
med

# ===== CELL 9 =====
corr = panel[FEATURES].corr()
corr_display = corr.rename(index=FEATURE_LABELS, columns=FEATURE_LABELS)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_display, cmap='coolwarm', center=0, annot=True, fmt='.2f',
            square=True, cbar_kws={'shrink':0.7, 'label':'Pearson correlation'},
            annot_kws={'size':7}, ax=ax)
ax.set_title('Pairwise correlation between financial ratios — Russia')
ax.set_xlabel(''); ax.set_ylabel('')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
save_fig(fig, '03_correlation_heatmap'); plt.show()

corr.round(3).to_csv(REPORTS / 'ru_correlation_matrix.csv', encoding='utf-8-sig')

high = (corr.abs() > 0.85) & (corr.abs() < 1.0)
pairs = [(i,j,corr.loc[i,j]) for i in corr.index for j in corr.columns if i<j and high.loc[i,j]]
print('|corr| > 0.85:')
for p in pairs: print(f'  {p[0]} × {p[1]}: {p[2]:+.2f}')
if not pairs: print('  нет — мультиколлинеарность под контролем')

# ===== CELL 11 =====
X = panel[FEATURES].values
y = panel[TARGET].values
groups = panel[ID_COL].values

# company-level stratified split: 80/20 отдельно по bankrupt- и active-компаниям
company_label = panel.groupby(ID_COL)[TARGET].max()  # 1, если у компании есть target=1
companies_b = np.array(company_label[company_label == 1].index.values, copy=True)
companies_a = np.array(company_label[company_label == 0].index.values, copy=True)

_rng = np.random.default_rng(RNG)
_rng.shuffle(companies_b)
_rng.shuffle(companies_a)

def _split_ids(arr, test_frac=0.2):
    n_test = int(round(len(arr) * test_frac))
    return arr[n_test:], arr[:n_test]

train_b, test_b = _split_ids(companies_b, 0.20)
train_a, test_a = _split_ids(companies_a, 0.20)

train_ids = set(train_b) | set(train_a)
test_ids  = set(test_b)  | set(test_a)

mask_train = panel[ID_COL].isin(train_ids).values
mask_test  = panel[ID_COL].isin(test_ids).values

X_train, y_train = X[mask_train], y[mask_train]
X_test,  y_test  = X[mask_test],  y[mask_test]
g_train, g_test  = groups[mask_train], groups[mask_test]

print(f'Train: {X_train.shape},  defaults = {y_train.sum()}  ({y_train.mean()*100:.2f}%)')
print(f'Test:  {X_test.shape},   defaults = {y_test.sum()}   ({y_test.mean()*100:.2f}%)')
print(f'Companies — train: {len(train_ids):,}, test: {len(test_ids):,}, '
      f'overlap: {len(train_ids & test_ids)} (должно быть 0)')
print(f'Bankrupt companies — train: {len(train_b)}, test: {len(test_b)}')
# ===== CELL 13 =====
POS_WEIGHT = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
print(f'POS_WEIGHT (n_neg / n_pos на train) = {POS_WEIGHT:.1f}')

logit = Pipeline([
    ('sc', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight={0: 1, 1: POS_WEIGHT},
                               solver='liblinear', random_state=RNG)),
])
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=3,
    class_weight={0: 1, 1: POS_WEIGHT}, n_jobs=-1, random_state=RNG,
)
xgb = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    scale_pos_weight=POS_WEIGHT,
    eval_metric='aucpr', tree_method='hist', random_state=RNG, n_jobs=-1,
)

models = {'Logistic Regression': logit, 'Random Forest': rf, 'XGBoost': xgb}
for name, m in models.items():
    m.fit(X_train, y_train)
    print(f'✓ trained {name}')
# ===== CELL 15 =====
from sklearn.model_selection import StratifiedGroupKFold
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
        'PR-AUC train':  tr_pr,  'PR-AUC test':  te_pr,  'ΔPR':  tr_pr - te_pr,
    })
res = pd.DataFrame(rows).set_index('Model').round(4)
res.to_csv(REPORTS / 'ru_h1_metrics.csv', encoding='utf-8-sig')
print('Single split (group-aware):')
print(res)

# --- (b) 5-fold Stratified Group K-Fold CV ---
skgf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
cv_rows = []
for name, m_proto in models.items():
    roc_tr_l, pr_tr_l, roc_te_l, pr_te_l = [], [], [], []
    for tr_idx, te_idx in skgf.split(X, y, groups=groups):
        # Пересчёт веса в каждом фолде от train-дисбаланса
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
cv.to_csv(REPORTS / 'ru_h1_cv_metrics.csv', encoding='utf-8-sig')
print('\n5-fold Stratified Group K-Fold CV:')
print(cv)
res
# ===== CELL 16 =====
best_roc_name = res['ROC-AUC test'].idxmax()
best_pr_name  = res['PR-AUC test'].idxmax()
logit_delta   = res.loc['Logistic Regression', 'ΔROC']
ens_delta_max = res.loc[['Random Forest','XGBoost'], 'ΔROC'].max()
ens_roc_max   = res.loc[['Random Forest','XGBoost'], 'ROC-AUC test'].max()
logit_roc     = res.loc['Logistic Regression', 'ROC-AUC test']
ens_pr_max    = res.loc[['Random Forest','XGBoost'], 'PR-AUC test'].max()
logit_pr      = res.loc['Logistic Regression', 'PR-AUC test']

log('\n## H1 — переобучение на малом числе дефолтов')
log(f'* Split: **group-aware** (по `Регистрационный номер`); POS_WEIGHT = n_neg/n_pos на train = **{POS_WEIGHT:.1f}**.')
log(f'* Best test ROC-AUC: **{best_roc_name}** ({res.loc[best_roc_name,"ROC-AUC test"]:.4f}); '
    f'Logit ROC-AUC = {logit_roc:.4f}.')
log(f'* Best test PR-AUC: **{best_pr_name}** ({res.loc[best_pr_name,"PR-AUC test"]:.4f}); '
    f'Logit PR-AUC = {logit_pr:.4f}.')
log(f'* ΔROC (train−test): Logit **{logit_delta:+.4f}**, max ensemble **{ens_delta_max:+.4f}**.')
log('')
log('**5-fold Stratified Group K-Fold CV (mean ± std):**')
for name in cv.index:
    log(f'* {name}: ROC {cv.loc[name,"ROC-AUC test mean"]:.3f}±{cv.loc[name,"ROC-AUC test std"]:.3f}, '
        f'PR {cv.loc[name,"PR-AUC test mean"]:.3f}±{cv.loc[name,"PR-AUC test std"]:.3f}, '
        f'ΔROC {cv.loc[name,"ΔROC (train−test)"]:+.3f}')

cond_roc   = ens_roc_max > logit_roc
cond_delta = ens_delta_max > logit_delta
logit_wins_pr = logit_pr >= ens_pr_max

if cond_roc and cond_delta:
    v_roc = '✅ По ROC-AUC H1 выполняется: ансамбли точнее и переобучаются сильнее.'
elif not cond_roc and cond_delta:
    v_roc = '⚠️ По ROC-AUC H1 частично: ансамбли не точнее Logit, но переобучаются сильнее.'
elif cond_roc and not cond_delta:
    v_roc = '⚠️ По ROC-AUC H1 частично: ансамбли точнее, но зазор сопоставим.'
else:
    v_roc = '❌ По ROC-AUC H1 не выполняется.'

if logit_wins_pr:
    v_pr = '✅ По PR-AUC (адекватной метрике при 1:{:.0f} дисбалансе) **Logit превосходит ансамбли** — главный аргумент H1.'.format(POS_WEIGHT)
else:
    v_pr = '⚠️ По PR-AUC ансамбли не проигрывают Logit — H1 ослабляется.'

verdict = v_roc + '\n\n' + v_pr
log(f'\n{verdict}')
print('\n' + verdict)
# ===== CELL 18 =====
ensemble_pr = res.loc[['Random Forest', 'XGBoost'], 'PR-AUC test']
best_name = ensemble_pr.idxmax()
best_model = models[best_name]
print(f'Best ensemble by PR-AUC test: {best_name} '
      f'(PR-AUC={ensemble_pr.max():.4f}, ROC-AUC={res.loc[best_name, "ROC-AUC test"]:.4f})')

explainer = shap.TreeExplainer(best_model)
sv = explainer.shap_values(X_test)

# shap может вернуть list или 3D-массив — нормализуем под бинарную задачу
if isinstance(sv, list):
    sv = sv[1]
elif hasattr(sv, 'ndim') and sv.ndim == 3:
    sv = sv[:, :, 1]
print('SHAP shape:', sv.shape)

plt.figure()
shap.summary_plot(sv, X_test, feature_names=FEATURE_LABELS_LIST, show=False)
fig = plt.gcf()
fig.suptitle(
    f'Russia — feature impact on predicted default risk (SHAP, {best_name})',
    y=1.02, fontsize=12,
)
save_fig(fig, '04_shap_summary'); plt.show()
# ===== CELL 19 =====
mean_abs = np.abs(sv).mean(axis=0)
fi = pd.Series(mean_abs, index=FEATURES).sort_values(ascending=False)
fi.round(4).to_csv(REPORTS / 'ru_shap_feature_importance.csv', header=['mean_abs_shap'], encoding='utf-8-sig')
print('Top features by mean |SHAP|:')
print(fi.round(4).to_string())

group_sum = {g: fi[cols].sum() for g, cols in FEATURE_GROUPS.items()}
group_df = pd.Series(group_sum).sort_values(ascending=False).round(4)
group_df.to_csv(REPORTS / 'ru_shap_group_importance.csv', header=['sum_abs_shap'], encoding='utf-8-sig')
print('\nΣ|SHAP| по блокам:')
print(group_df.to_string())

liq_inn = group_df.get('Liquidity', 0) + group_df.get('Innovation', 0)
lev     = group_df.get('Leverage',   0)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['Liquidity + Innovation', 'Leverage'],
              [liq_inn, lev], color=['#2E75B6', '#C00000'])
ax.set_ylabel('Sum of mean |SHAP| (group impact on default risk)')
ax.set_title('H2 — Liquidity + Innovation vs Leverage (Russia, SHAP group importance)')
ax.set_xlabel('Feature group')
for b, v in zip(bars, [liq_inn, lev]):
    ax.text(b.get_x()+b.get_width()/2, v, f'{v:.3f}', ha='center', va='bottom')
plt.tight_layout()
save_fig(fig, '05_h2_group_importance'); plt.show()

log('\n## H2 — Liquidity+Innovation vs Leverage (SHAP)')
log(f'* Best ensemble (by PR-AUC test): **{best_name}** '
    f'(PR-AUC={ensemble_pr.max():.4f}, ROC-AUC={res.loc[best_name, "ROC-AUC test"]:.4f})')
log(f'* Σ|SHAP| Liquidity + Innovation = **{liq_inn:.4f}**')
log(f'* Σ|SHAP| Leverage               = **{lev:.4f}**')
if lev > 0:
    log(f'* Ratio = **{liq_inn/lev:.2f}×**')

if liq_inn > lev:
    verdict2 = '✅ **H2 ПОДТВЕРЖДАЕТСЯ**: в IT-секторе России ликвидность и НМА важнее рычага.'
else:
    verdict2 = '❌ H2 НЕ подтверждается: рычаг остаётся доминирующим предиктором.'
log(f'\n{verdict2}')
print('\n' + verdict2)
# ===== CELL 21 =====
summary_path = REPORTS / 'ru_summary.md'
summary_path.write_text('\n'.join(_report_lines), encoding='utf-8')
print(f'✅ Итоговый отчёт: {summary_path}')
print('\nВсе артефакты в reports/russia/:')
for p in sorted(REPORTS.iterdir()):
    print(f'  {p.name}  ({p.stat().st_size/1024:.1f} KB)')
