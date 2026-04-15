"""Helper that generates 40_china_eda_and_models.ipynb and 50_cross_country_pit.ipynb."""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# 40_china_eda_and_models.ipynb
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# China — EDA, TTC-модели, SHAP

**Что тут:** EDA → Stage-1 (Through-The-Cycle) → проверка H1 и H2 по Китаю.

* **H1.** Ансамбли (RF, XGBoost) выдают более высокий ROC-AUC, чем Logit, но у них больше **зазор train–test** (переобучение).
* **H2.** Сумма `|SHAP|` по блокам **Ликвидность + Инновации (НМА+R&D)** выше, чем по блоку **Леверидж**.

**Выборка:** 339 активных Wind-компаний + 11 реально-дефолтных (11 vs 3412 строк, дисбаланс ≈ 1:71). Выбросы по делистингу с экономически нерелевантными причинами (M&A, приватизация) уже удалены на этапе загрузки (`30_china_load_and_clean.ipynb`) — то есть здесь мы проверяем Hypotheses именно на *true economic distress*.

**Правила научрука:** без SMOTE, `class_weight='balanced'`, stratified split. Артефакты сохраняются в `reports/china/`.""")

code("""import warnings
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
REPORTS = Path('../reports/china')
REPORTS.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name):
    path = REPORTS / f'{name}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    print(f'  saved → {path}')

_report_lines = ['# China — Stage-1 (TTC) results\\n']
def log(msg=''):
    print(msg)
    _report_lines.append(msg)

df = pd.read_csv(PROCESSED / 'cn_panel_cleaned.csv', encoding='utf-8-sig')
print('Shape:', df.shape)
print('Tickers per target:', df.groupby('target')['ticker'].nunique().to_dict())
df.head(3)""")

md("""## 1. Feature engineering

Мэппинг колонок `cn_panel_cleaned.csv` в ratios H2. В китайской выборке есть **R&D expense** — это отдельный канал «Innovation» сверх нематериальных активов, что важно для H2.

* **Liquidity:** `current_ratio`, `cash_to_assets`, `cash_to_cl`, `wc_to_assets`
* **Innovation:** `intangibles_to_assets`, `rd_to_revenue`
* **Leverage:** `debt_to_assets`, `debt_to_equity`, `lt_debt_to_assets`, `interest_coverage`
* **Profitability:** `roa`, `net_margin`, `operating_margin`, `cfo_to_assets`
* **Size:** `log_assets`, `log_revenue`

Винзоризация 1/99% — как в российской тетради. Пропуски заполняются медианой (после ffill/bfill внутри компании в loader-ноутбуке).""")

code("""TARGET = 'target'
ID_COL = 'ticker'

A, CA, CASH, INT_A = 'total_assets', 'current_assets', 'cash', 'intangibles'
EQ, LIAB, ST_L = 'total_equity', 'total_liab', 'current_liab'
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

panel['debt_to_assets']        = safe_div(panel[LIAB], panel[A])
panel['debt_to_equity']        = safe_div(panel[LIAB], panel[EQ])
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
print(f'Фичи: {len(FEATURES)}  |  stats → reports/china/cn_feature_stats.csv')
panel[FEATURES].describe().T.round(3)""")

md("""## 2. EDA

### 2.1 Классовый дисбаланс""")
code("""vc = panel[TARGET].value_counts()
n_active  = int(vc.get(0, 0))
n_default = int(vc.get(1, 0))
ratio = n_default / (n_active + n_default)
counts = pd.Series({'Active (0)': n_active, 'Default (1)': n_default})

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(counts.index, counts.values, color=['#2E75B6', '#C00000'])
ax.set_title(f'Class distribution — China (imbalance ≈ 1:{n_active//max(n_default,1)})')
ax.set_ylabel('rows (company-year)')
for b, v in zip(bars, counts.values):
    ax.text(b.get_x()+b.get_width()/2, v, f'{v:,}', ha='center', va='bottom')
plt.tight_layout()
save_fig(fig, '01_class_distribution'); plt.show()

log(f'\\n## Class balance\\n* Active (0): **{n_active:,}**')
log(f'* Default (1): **{n_default:,}**')
log(f'* Positive share: **{ratio*100:.3f}%**  (≈ 1:{n_active//max(n_default,1)})')""")

md("""### 2.2 Violin plots — по классу""")
code("""panel['_lbl'] = panel[TARGET].map({0:'Active', 1:'Default'})
show = ['current_ratio', 'intangibles_to_assets', 'debt_to_assets', 'roa']

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col in zip(axes.ravel(), show):
    sns.violinplot(data=panel, x='_lbl', y=col, ax=ax,
                   palette={'Active':'#2E75B6','Default':'#C00000'}, cut=0)
    ax.set_title(col); ax.set_xlabel('')
plt.suptitle('Distributions by class (winsorized 1–99%)', y=1.02)
plt.tight_layout()
save_fig(fig, '02_violin_by_class'); plt.show()

med = panel.groupby('_lbl')[show].median().round(3)
med.to_csv(REPORTS / 'cn_medians_by_class.csv', encoding='utf-8-sig')
med""")

md("""### 2.3 Корреляционная матрица""")
code("""corr = panel[FEATURES].corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=True, fmt='.2f',
            square=True, cbar_kws={'shrink':0.7}, annot_kws={'size':7}, ax=ax)
ax.set_title('Feature correlation matrix — China')
plt.tight_layout()
save_fig(fig, '03_correlation_heatmap'); plt.show()

corr.round(3).to_csv(REPORTS / 'cn_correlation_matrix.csv', encoding='utf-8-sig')

high = (corr.abs() > 0.85) & (corr.abs() < 1.0)
pairs = [(i,j,corr.loc[i,j]) for i in corr.index for j in corr.columns if i<j and high.loc[i,j]]
print('|corr| > 0.85:')
for p in pairs: print(f'  {p[0]} × {p[1]}: {p[2]:+.2f}')
if not pairs: print('  нет — мультиколлинеарность под контролем')""")

md("""## 3. Train / Test split

Stratified 80/20 на уровне строк (company-year). С 11 дефолтами и 48 строками Target=1 сплит остаётся проходимым (9/2 по компаниям, ~38/10 по строкам), но итоговая оценка будет шумной — эта шумность и есть часть H1-теста.""")
code("""X = panel[FEATURES].values
y = panel[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RNG
)
print(f'Train: {X_train.shape},  defaults = {y_train.sum()}  ({y_train.mean()*100:.2f}%)')
print(f'Test:  {X_test.shape},  defaults = {y_test.sum()}  ({y_test.mean()*100:.2f}%)')""")

md("""## 4. Stage-1 модели (TTC)

`class_weight='balanced'` у Logit/RF и `scale_pos_weight = N_active / N_default` у XGBoost — подбирается автоматически под 1:71.""")
code("""POS_WEIGHT = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f'scale_pos_weight (XGB) = {POS_WEIGHT:.2f}')

logit = Pipeline([
    ('sc', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                               solver='liblinear', random_state=RNG)),
])
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    class_weight='balanced', n_jobs=-1, random_state=RNG,
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
    print(f'✓ trained {name}')""")

md("""## 5. H1 — Train vs Test, зазор (overfit check)""")
code("""def scores(model, X, y):
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)[:, 1]
    else:
        p = model.decision_function(X)
    return roc_auc_score(y, p), average_precision_score(y, p)

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
print('metrics → reports/china/cn_h1_metrics.csv')
res""")

code("""best_test_roc = res['ROC-AUC test'].idxmax()
logit_delta = res.loc['Logistic Regression', 'ΔROC']
ens_delta_max = res.loc[['Random Forest','XGBoost'], 'ΔROC'].max()
ens_test_max  = res.loc[['Random Forest','XGBoost'], 'ROC-AUC test'].max()
logit_test    = res.loc['Logistic Regression', 'ROC-AUC test']

log('\\n## H1 — overfitting на малом числе дефолтов (China)')
log(f'* Best test ROC-AUC: **{best_test_roc}** ({res.loc[best_test_roc,"ROC-AUC test"]:.4f})')
log(f'* Logit ΔROC (train−test): **{logit_delta:+.4f}**')
log(f'* Max ensemble ΔROC:        **{ens_delta_max:+.4f}**')

cond_a = ens_test_max > logit_test
cond_b = ens_delta_max > logit_delta

if cond_a and cond_b:
    verdict = '✅ **H1 ПОДТВЕРЖДАЕТСЯ**: ансамбли точнее на тесте, но переобучаются сильнее Logit.'
elif cond_a and not cond_b:
    verdict = '⚠️ H1 частично: ансамбли точнее, но Logit НЕ стабильнее (Δ сопоставимы).'
elif not cond_a and cond_b:
    verdict = '⚠️ H1 частично: Logit сопоставим по точности, но ансамбли переобучены.'
else:
    verdict = '❌ H1 НЕ подтверждается: ансамбли не дают прироста и не хуже по зазору.'
log(f'\\n{verdict}')
print('\\n' + verdict)""")

md("""## 6. H2 — SHAP для лучшего ансамбля

Группировка фичей: `Liquidity + Innovation (intangibles + R&D)` vs `Leverage`.""")
code("""ensemble_scores = res.loc[['Random Forest', 'XGBoost'], 'ROC-AUC test']
best_name = ensemble_scores.idxmax()
best_model = models[best_name]
print(f'Best ensemble: {best_name}  (test ROC-AUC = {ensemble_scores.max():.4f})')

explainer = shap.TreeExplainer(best_model)
sv = explainer.shap_values(X_test)
if isinstance(sv, list):
    sv = sv[1]
elif hasattr(sv, 'ndim') and sv.ndim == 3:
    sv = sv[:, :, 1]
print('SHAP shape:', sv.shape)

plt.figure()
shap.summary_plot(sv, X_test, feature_names=FEATURES, show=False)
fig = plt.gcf()
save_fig(fig, '04_shap_summary'); plt.show()""")

code("""mean_abs = np.abs(sv).mean(axis=0)
fi = pd.Series(mean_abs, index=FEATURES).sort_values(ascending=False)
fi.round(4).to_csv(REPORTS / 'cn_shap_feature_importance.csv', header=['mean_abs_shap'], encoding='utf-8-sig')
print('Top features by mean |SHAP|:')
print(fi.round(4).to_string())

group_sum = {g: fi[cols].sum() for g, cols in FEATURE_GROUPS.items()}
group_df = pd.Series(group_sum).sort_values(ascending=False).round(4)
group_df.to_csv(REPORTS / 'cn_shap_group_importance.csv', header=['sum_abs_shap'], encoding='utf-8-sig')
print('\\nΣ|SHAP| по блокам:')
print(group_df.to_string())

liq_inn = group_df.get('Liquidity', 0) + group_df.get('Innovation', 0)
lev     = group_df.get('Leverage',   0)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['Liquidity + Innovation', 'Leverage'],
              [liq_inn, lev], color=['#2E75B6', '#C00000'])
ax.set_ylabel('Σ mean |SHAP|')
ax.set_title('H2: feature-group importance — China')
for b, v in zip(bars, [liq_inn, lev]):
    ax.text(b.get_x()+b.get_width()/2, v, f'{v:.3f}', ha='center', va='bottom')
plt.tight_layout()
save_fig(fig, '05_h2_group_importance'); plt.show()

log('\\n## H2 — Liquidity+Innovation vs Leverage (SHAP, China)')
log(f'* Best ensemble: **{best_name}** (test ROC-AUC = {ensemble_scores.max():.4f})')
log(f'* Σ|SHAP| Liquidity + Innovation = **{liq_inn:.4f}**')
log(f'* Σ|SHAP| Leverage               = **{lev:.4f}**')
if lev > 0:
    log(f'* Ratio = **{liq_inn/lev:.2f}×**')

if liq_inn > lev:
    verdict2 = '✅ **H2 ПОДТВЕРЖДАЕТСЯ (Китай)**: ликвидность + инновационные активы важнее рычага.'
else:
    verdict2 = '❌ H2 НЕ подтверждается (Китай): рычаг остаётся доминирующим предиктором.'
log(f'\\n{verdict2}')
print('\\n' + verdict2)""")

md("""## 7. Экспорт TTC-скора для Stage-2 (PIT) и кросс-странового сравнения

Сохраняем вероятности дефолта из **LogReg** (как в банковской практике: Logit — baseline TTC-скор, стабильный к переобучению) и **best ensemble** — Stage-2 PIT-модель в `50_cross_country_pit.ipynb` возьмёт один из них как единственный company-specific признак.""")
code("""logit_proba = models['Logistic Regression'].predict_proba(X)[:, 1]
best_proba  = best_model.predict_proba(X)[:, 1]

scores_df = panel[['ticker', 'company_name', 'year', TARGET, 'source_class']].copy()
scores_df['ttc_logit']    = logit_proba
scores_df['ttc_ensemble'] = best_proba
scores_df['ttc_best_model'] = best_name
scores_df.to_csv(REPORTS / 'cn_ttc_scores.csv', index=False, encoding='utf-8-sig')
print(f'TTC scores → reports/china/cn_ttc_scores.csv  ({len(scores_df):,} rows)')
scores_df.head()""")

md("""## 8. Сохраняем итоговый отчёт""")
code("""summary_path = REPORTS / 'cn_summary.md'
summary_path.write_text('\\n'.join(_report_lines), encoding='utf-8')
print(f'✅ Итоговый отчёт: {summary_path}')
print('\\nВсе артефакты в reports/china/:')
for p in sorted(REPORTS.iterdir()):
    print(f'  {p.name}  ({p.stat().st_size/1024:.1f} KB)')""")

md("""## 9. Интерпретация результатов — Китай

| Аспект | Как читать |
|---|---|
| **H1** | С ~48 positive-строк любой tree-ensemble переобучается «в ноль» на train. Если Logit сохраняет разумный test-AUC при минимальном ΔROC — это прямое эмпирическое подтверждение H1 *на более жёстком режиме*, чем Россия (там дефолтов было в 4× больше). |
| **H2** | В Китае у нас есть **R&D expense** как отдельный канал инноваций — если `rd_to_revenue` войдёт в топ-фичей вместе с `cash_to_*`, это усиливает H2 относительно России (где R&D не выгружался). |
| **Что дальше** | `cn_ttc_scores.csv` становится input-ом для `50_cross_country_pit.ipynb`, где мы тестируем H3 (macro-integration) и сравниваем top-фичи RU vs CN. |

Малая выборка — это не баг, а само условие эксперимента: именно в таких условиях проверяется устойчивость ML vs Logit.""")

nb['cells'] = cells
nbf.write(nb, HERE / '40_china_eda_and_models.ipynb')
print('wrote 40_china_eda_and_models.ipynb')


# ---------------------------------------------------------------------------
# 50_cross_country_pit.ipynb
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Cross-Country PIT — Russia vs China (H3)

**Цель:** формально протестировать Hypothesis 3 из паспорта работы —
*включение макропеременных улучшает предсказательную силу (PIT > TTC), а структура риска зависит от локального рынка:*
* **Россия** — дорогой капитал → доминирует **liquidity deficit**;
* **Китай** — институциональная поддержка → доминирует **operational profitability**.

### Этапы
1. **Data alignment** — обе страны приводятся к ratios (scale-invariant), период унифицирован 2014–2024.
2. **TTC refit** — одинаковая LogReg на каждой стране → `ttc_score` как единственный company-specific признак.
3. **PIT Stage-2** — LogReg на `[ttc_score, GDP_Growth, Inflation_Rate]`, сравнение TTC vs PIT AUC.
4. **SHAP side-by-side** — топ-3 фичей RU vs топ-3 фичей CN.

Макро-данные сейчас — *placeholder-константы*; функция `load_macro(country)` изолирована, чтобы заменить mock на реальный IMF/Росстат/NBS-feed одной правкой.""")

code("""import warnings
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
REPORTS = Path('../reports/cross_country')
REPORTS.mkdir(parents=True, exist_ok=True)

YEARS = range(2014, 2025)  # inclusive 2014..2024
print('Common window:', list(YEARS))""")

md("""## 1. Загрузка и выравнивание двух панелей

Каждая панель приводится к единой схеме колонок: `ticker, year, target` + 16 финансовых ratios. Абсолютные суммы в RUB/CNY не сравниваются никогда — только относительные показатели.""")

code("""# ---- Russia ---------------------------------------------------------------
ru_raw = pd.read_csv(PROCESSED / 'ru_panel_cleaned.csv', encoding='utf-8-sig')

RU = {
    'ID': 'Регистрационный номер', 'TARGET': 'is_bankrupt', 'YEAR': 'year',
    'A': 'Активы  всего', 'CA': 'Оборотные активы',
    'CASH': 'Денежные средства и денежные эквиваленты',
    'INT_A': 'Нематериальные активы', 'EQ': 'Капитал и резервы',
    'LT_L': 'Долгосрочные обязательства', 'ST_L': 'Краткосрочные обязательства',
    'REV': 'Выручка', 'EBIT': 'EBIT', 'NI': 'Чистая прибыль (убыток)',
    'INTEREST': 'Проценты к уплате',
    'CFO': 'Сальдо денежных потоков от текущих операций',
}
# ---- China ----------------------------------------------------------------
cn_raw = pd.read_csv(PROCESSED / 'cn_panel_cleaned.csv', encoding='utf-8-sig')

def safe_div(a, b):
    b = b.replace(0, np.nan)
    return a / b

def build_ratios_ru(df):
    d = df.copy()
    total_debt = d[RU['LT_L']] + d[RU['ST_L']]
    out = pd.DataFrame({
        'ticker': d[RU['ID']].astype(str),
        'year':   d[RU['YEAR']].astype(int),
        'target': d[RU['TARGET']].astype(int),
    })
    out['current_ratio']         = safe_div(d[RU['CA']], d[RU['ST_L']])
    out['cash_to_assets']        = safe_div(d[RU['CASH']], d[RU['A']])
    out['cash_to_cl']            = safe_div(d[RU['CASH']], d[RU['ST_L']])
    out['wc_to_assets']          = safe_div(d[RU['CA']] - d[RU['ST_L']], d[RU['A']])
    out['intangibles_to_assets'] = safe_div(d[RU['INT_A']], d[RU['A']])
    out['debt_to_assets']        = safe_div(total_debt, d[RU['A']])
    out['debt_to_equity']        = safe_div(total_debt, d[RU['EQ']])
    out['lt_debt_to_assets']     = safe_div(d[RU['LT_L']], d[RU['A']])
    out['interest_coverage']     = safe_div(d[RU['EBIT']], d[RU['INTEREST']])
    out['roa']                   = safe_div(d[RU['NI']], d[RU['A']])
    out['net_margin']            = safe_div(d[RU['NI']], d[RU['REV']])
    out['operating_margin']      = safe_div(d[RU['EBIT']], d[RU['REV']])
    out['cfo_to_assets']         = safe_div(d[RU['CFO']], d[RU['A']])
    out['log_assets']            = np.log1p(d[RU['A']].clip(lower=0))
    out['log_revenue']           = np.log1p(d[RU['REV']].clip(lower=0))
    return out

def build_ratios_cn(df):
    d = df.copy()
    lt_liab = (d['total_liab'] - d['current_liab']).clip(lower=0)
    out = pd.DataFrame({
        'ticker': d['ticker'].astype(str),
        'year':   d['year'].astype(int),
        'target': d['target'].astype(int),
    })
    out['current_ratio']         = safe_div(d['current_assets'], d['current_liab'])
    out['cash_to_assets']        = safe_div(d['cash'], d['total_assets'])
    out['cash_to_cl']            = safe_div(d['cash'], d['current_liab'])
    out['wc_to_assets']          = safe_div(d['current_assets'] - d['current_liab'], d['total_assets'])
    out['intangibles_to_assets'] = safe_div(d['intangibles'], d['total_assets'])
    out['debt_to_assets']        = safe_div(d['total_liab'], d['total_assets'])
    out['debt_to_equity']        = safe_div(d['total_liab'], d['total_equity'])
    out['lt_debt_to_assets']     = safe_div(lt_liab, d['total_assets'])
    out['interest_coverage']     = safe_div(d['ebit'], d['interest_expense'])
    out['roa']                   = safe_div(d['net_profit'], d['total_assets'])
    out['net_margin']            = safe_div(d['net_profit'], d['total_revenue'])
    out['operating_margin']      = safe_div(d['ebit'], d['total_revenue'])
    out['cfo_to_assets']         = safe_div(d['cfo'], d['total_assets'])
    out['log_assets']            = np.log1p(d['total_assets'].clip(lower=0))
    out['log_revenue']           = np.log1p(d['total_revenue'].clip(lower=0))
    return out

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

def preprocess(panel):
    p = panel[panel['year'].isin(list(YEARS))].copy()
    p[FEATURES] = p[FEATURES].replace([np.inf, -np.inf], np.nan)
    for c in FEATURES:
        lo, hi = p[c].quantile([0.01, 0.99])
        p[c] = p[c].clip(lo, hi)
    p[FEATURES] = p[FEATURES].fillna(p[FEATURES].median(numeric_only=True))
    return p

ru = preprocess(build_ratios_ru(ru_raw))
cn = preprocess(build_ratios_cn(cn_raw))
print(f'Russia:  {ru.shape}  defaults={int(ru["target"].sum())}')
print(f'China:   {cn.shape}   defaults={int(cn["target"].sum())}')""")

md("""## 2. Stage-1 refit — LogReg TTC на каждой стране

Для честного сравнения по H3 используем **одну и ту же архитектуру** (LogReg + StandardScaler + class_weight='balanced'). TTC-score — это output `predict_proba[:,1]`.""")

code("""def fit_ttc(panel, seed=RNG):
    X = panel[FEATURES].values
    y = panel['target'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    pipe = Pipeline([
        ('sc', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                   solver='liblinear', random_state=seed)),
    ])
    pipe.fit(X_tr, y_tr)
    proba_all = pipe.predict_proba(X)[:, 1]
    idx_train = np.zeros(len(y), dtype=bool)
    # reconstruct split indices
    rng = np.random.RandomState(seed)  # not used further, just keep reproducibility
    auc_tr = roc_auc_score(y_tr, pipe.predict_proba(X_tr)[:, 1])
    auc_te = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    return pipe, proba_all, auc_tr, auc_te

ru_model, ru_ttc, ru_auc_tr, ru_auc_te = fit_ttc(ru)
cn_model, cn_ttc, cn_auc_tr, cn_auc_te = fit_ttc(cn)

ru = ru.assign(ttc_score=ru_ttc)
cn = cn.assign(ttc_score=cn_ttc)

pd.DataFrame({
    'Country': ['Russia', 'China'],
    'TTC ROC-AUC train': [ru_auc_tr, cn_auc_tr],
    'TTC ROC-AUC test':  [ru_auc_te, cn_auc_te],
}).round(4)""")

md("""## 3. Макропеременные (placeholder)

Функция `load_macro(country)` возвращает `pd.DataFrame(columns=['year','GDP_Growth','Inflation_Rate'])`. Сейчас вшиты плейсхолдерные значения — это именно то, о чём просил промт: подключить реальные цифры позднее можно заменой одного словаря без изменения модельного кода.

Источники, которые стоит подключить позже:
* **Russia** — Росстат / IMF WEO (`NGDP_RPCH`, `PCPIPCH`)
* **China** — NBS / IMF WEO""")

code("""# Placeholder — замените на реальные значения при наличии данных.
# Значения взяты как правдоподобные публичные оценки, но **не используйте их для выводов**
# пока не обновите из официального источника.
_MACRO_PLACEHOLDER = {
    'Russia': {
        2014: (0.7, 11.4), 2015: (-2.0, 12.9), 2016: (0.2, 5.4),  2017: (1.8, 2.5),
        2018: (2.8, 4.3),  2019: (2.2, 3.0),   2020: (-2.7, 3.4), 2021: (5.6, 6.7),
        2022: (-1.2, 13.8),2023: (3.6, 7.4),   2024: (3.9, 8.3),
    },
    'China': {
        2014: (7.4, 2.0),  2015: (7.0, 1.4), 2016: (6.9, 2.0), 2017: (6.9, 1.6),
        2018: (6.8, 2.1),  2019: (6.0, 2.9), 2020: (2.2, 2.5), 2021: (8.4, 0.9),
        2022: (3.0, 2.0),  2023: (5.2, 0.2), 2024: (5.0, 0.4),
    },
}

def load_macro(country: str) -> pd.DataFrame:
    \"\"\"Return year-level macro frame. Swap to real data by editing this body only.\"\"\"
    src = _MACRO_PLACEHOLDER[country]
    return pd.DataFrame(
        [(y, g, i) for y, (g, i) in src.items()],
        columns=['year', 'GDP_Growth', 'Inflation_Rate'],
    )

macro_ru = load_macro('Russia')
macro_cn = load_macro('China')

ru = ru.merge(macro_ru, on='year', how='left')
cn = cn.merge(macro_cn, on='year', how='left')
display(pd.concat([macro_ru.assign(Country='Russia'), macro_cn.assign(Country='China')]))""")

md("""## 4. Stage-2 PIT model — `[ttc_score, GDP_Growth, Inflation_Rate]`

Сравниваем AUC: baseline TTC (один только ttc_score как классификатор) vs PIT (добавляем макро). Разница и есть прямой ответ H3 по **первой** части гипотезы.""")

code("""def compare_ttc_vs_pit(panel, country):
    X_ttc = panel[['ttc_score']].values
    X_pit = panel[['ttc_score', 'GDP_Growth', 'Inflation_Rate']].values
    y = panel['target'].values
    X_ttc_tr, X_ttc_te, X_pit_tr, X_pit_te, y_tr, y_te = train_test_split(
        X_ttc, X_pit, y, test_size=0.20, stratify=y, random_state=RNG
    )
    m_ttc = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=RNG)
    m_pit = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=RNG)
    m_ttc.fit(X_ttc_tr, y_tr); m_pit.fit(X_pit_tr, y_tr)
    ttc_auc = roc_auc_score(y_te, m_ttc.predict_proba(X_ttc_te)[:, 1])
    pit_auc = roc_auc_score(y_te, m_pit.predict_proba(X_pit_te)[:, 1])
    ttc_pr  = average_precision_score(y_te, m_ttc.predict_proba(X_ttc_te)[:, 1])
    pit_pr  = average_precision_score(y_te, m_pit.predict_proba(X_pit_te)[:, 1])
    return {
        'Country': country,
        'TTC ROC-AUC (test)': ttc_auc, 'PIT ROC-AUC (test)': pit_auc,
        'ΔROC (PIT − TTC)':   pit_auc - ttc_auc,
        'TTC PR-AUC (test)':  ttc_pr,  'PIT PR-AUC (test)': pit_pr,
    }, m_pit

ru_row, ru_pit_model = compare_ttc_vs_pit(ru, 'Russia')
cn_row, cn_pit_model = compare_ttc_vs_pit(cn, 'China')

pit_res = pd.DataFrame([ru_row, cn_row]).set_index('Country').round(4)
pit_res.to_csv(REPORTS / 'h3_pit_vs_ttc.csv', encoding='utf-8-sig')
pit_res""")

code("""# H3 verdict — part 1 (macro lift)
lines = ['# H3 — Cross-Country PIT results\\n']
lines.append('## Part 1 — Does macro improve predictive power?')
for country, row in pit_res.iterrows():
    lines.append(f'* **{country}**: TTC={row["TTC ROC-AUC (test)"]:.4f} → PIT={row["PIT ROC-AUC (test)"]:.4f}  (Δ={row["ΔROC (PIT − TTC)"]:+.4f})')
both_up = (pit_res['ΔROC (PIT − TTC)'] > 0).all()
any_up  = (pit_res['ΔROC (PIT − TTC)'] > 0).any()
if both_up:
    msg = '✅ **H3 (часть 1) подтверждается**: PIT > TTC в обеих странах — макро-интеграция даёт прирост.'
elif any_up:
    msg = '⚠️ H3 (часть 1) частично: PIT > TTC только в одной стране.'
else:
    msg = '❌ H3 (часть 1) не подтверждается: макро не улучшает TTC в данном окне.'
lines.append('\\n' + msg)
print(msg)""")

md("""## 5. Side-by-side SHAP — top-фичи RU vs CN (вторая часть H3)

Для каждой страны обучаем XGBoost на полном наборе фичей и берём mean |SHAP|. Сравниваем **топ-3** и относим их к экономическим блокам (Liquidity / Profitability / Leverage / Innovation / Size).""")

code("""def fit_xgb(panel, seed=RNG):
    X = panel[FEATURES].values
    y = panel['target'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    pos_w = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1.0)
    m = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_w,
        eval_metric='aucpr', tree_method='hist', random_state=seed, n_jobs=-1,
    )
    m.fit(X_tr, y_tr)
    return m, X_te, y_te

def shap_importance(model, X_te):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te)
    if isinstance(sv, list): sv = sv[1]
    elif hasattr(sv, 'ndim') and sv.ndim == 3: sv = sv[:, :, 1]
    return pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False), sv

ru_xgb, ru_Xte, _ = fit_xgb(ru)
cn_xgb, cn_Xte, _ = fit_xgb(cn)
ru_fi, ru_sv = shap_importance(ru_xgb, ru_Xte)
cn_fi, cn_sv = shap_importance(cn_xgb, cn_Xte)

def feature_group(name):
    for g, cols in FEATURE_GROUPS.items():
        if name in cols: return g
    return 'Other'

top_k = 3
ru_top = ru_fi.head(top_k)
cn_top = cn_fi.head(top_k)

compare = pd.DataFrame({
    'Rank':   list(range(1, top_k + 1)),
    'Russia feature':          ru_top.index,
    'Russia group':            [feature_group(f) for f in ru_top.index],
    'Russia |SHAP|':           ru_top.values.round(4),
    'China feature':           cn_top.index,
    'China group':             [feature_group(f) for f in cn_top.index],
    'China |SHAP|':            cn_top.values.round(4),
})
compare.to_csv(REPORTS / 'h3_top3_features.csv', index=False, encoding='utf-8-sig')
compare""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plt.sca(axes[0])
shap.summary_plot(ru_sv, ru_Xte, feature_names=FEATURES, show=False, plot_size=None)
axes[0].set_title('Russia — SHAP summary')
plt.sca(axes[1])
shap.summary_plot(cn_sv, cn_Xte, feature_names=FEATURES, show=False, plot_size=None)
axes[1].set_title('China — SHAP summary')
plt.tight_layout()
fig.savefig(REPORTS / 'h3_shap_side_by_side.png', dpi=160, bbox_inches='tight')
plt.show()
print('  saved → reports/cross_country/h3_shap_side_by_side.png')""")

code("""# H3 — part 2: risk-profile dependency on local market
ru_groups = pd.Series({g: ru_fi[cols].sum() for g, cols in FEATURE_GROUPS.items()}).sort_values(ascending=False)
cn_groups = pd.Series({g: cn_fi[cols].sum() for g, cols in FEATURE_GROUPS.items()}).sort_values(ascending=False)

groups_df = pd.DataFrame({'Russia Σ|SHAP|': ru_groups, 'China Σ|SHAP|': cn_groups}).round(4)
groups_df.to_csv(REPORTS / 'h3_group_importance.csv', encoding='utf-8-sig')
print(groups_df)

ru_dominant = ru_groups.idxmax()
cn_dominant = cn_groups.idxmax()
lines.append('\\n## Part 2 — Risk profile dependency on local market')
lines.append(f'* **Russia dominant group:** {ru_dominant} (Σ|SHAP|={ru_groups.iloc[0]:.4f})')
lines.append(f'* **China dominant group:**  {cn_dominant} (Σ|SHAP|={cn_groups.iloc[0]:.4f})')
lines.append(f'* **Russia top-3 features:** {", ".join(ru_top.index)}')
lines.append(f'* **China top-3 features:**  {", ".join(cn_top.index)}')

ru_liq_first = ru_dominant == 'Liquidity'
cn_prof_first = cn_dominant == 'Profitability'
if ru_liq_first and cn_prof_first:
    msg2 = '✅ **H3 (часть 2) подтверждается**: в РФ доминирует liquidity, в Китае — profitability.'
elif ru_liq_first or cn_prof_first:
    msg2 = '⚠️ H3 (часть 2) частично: совпадает только одна из двух ожидаемых картин.'
else:
    msg2 = (f'❌ H3 (часть 2) не подтверждается прямолинейно: '
            f'доминируют {ru_dominant} (RU) и {cn_dominant} (CN). '
            f'Интерпретация должна идти через экономическое объяснение наблюдаемой картины.')
lines.append('\\n' + msg2)
print('\\n' + msg2)""")

md("""## 6. Итоговый отчёт""")

code("""summary_path = REPORTS / 'h3_summary.md'
summary_path.write_text('\\n'.join(lines), encoding='utf-8')
print(f'✅ Итоговый отчёт: {summary_path}')
print('\\nАртефакты в reports/cross_country/:')
for p in sorted(REPORTS.iterdir()):
    print(f'  {p.name}  ({p.stat().st_size/1024:.1f} KB)')""")

md("""## 7. Как подключить реальную макру

1. Открыть ячейку с `_MACRO_PLACEHOLDER`.
2. Заменить значения на выгрузку из IMF WEO (`imf.org/weo`) или локального источника (Росстат / NBS).
3. Перезапустить ноутбук целиком — остальной код не трогать.

Для **robustness check** можно добавить в `load_macro` третью переменную (напр. policy rate) — `compare_ttc_vs_pit` уже работает с произвольным числом макро-колонок, если расширить список признаков в `X_pit`.""")

nb['cells'] = cells
nbf.write(nb, HERE / '50_cross_country_pit.ipynb')
print('wrote 50_cross_country_pit.ipynb')
