"""
V2 refactor: unified event-based target with K-year horizon window.

Changes:
1. 10_russia_load_and_clean: CELL 5 — is_bankrupt = 1 on last K years (was: only last 1).
2. 30_china_load_and_clean: CELL 16 — add target-window recomputation before save,
   also add `default_company` entity flag.

Idempotent: re-running leaves notebooks in the same final state.

Run:
    python notebooks/_refactor_v2.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
K_HORIZON = 2  # years before delisting/bankruptcy counted as "distressed"
MARKER = f'# TARGET_WINDOW_V2 (K={K_HORIZON})'


def _load(nb_path):
    return json.loads(Path(nb_path).read_text(encoding='utf-8'))


def _save(nb, nb_path):
    Path(nb_path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')


def _set_code(cell, text):
    cell['source'] = text.splitlines(keepends=True)
    cell['outputs'] = []
    cell['execution_count'] = None


def _find_code(nb, needle):
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code' and needle in ''.join(c['source']):
            return i
    return -1


# -------------------- RU: patch CELL 5 target formula --------------------
def patch_ru(path):
    nb = _load(path)
    idx = _find_code(nb, "df['is_bankrupt'] = (")
    if idx < 0:
        raise RuntimeError('RU target cell not found')

    new_source = f"""{MARKER}
# Event-based target with K-year horizon window:
# is_bankrupt=1 on the last K years of each bankrupt company (the distressed window).
# Previous years of the same company remain class=0 (they were healthy then).
K_HORIZON = {K_HORIZON}

# Объединяем active + bankrupt
df = pd.concat([active, bankrupt], ignore_index=True)

# --- Метка компании-банкрота (entity-level, для group-split и визуализации) ---
df['bankrupt_company'] = df['is_bankrupt'].copy()

# --- Target: is_bankrupt = 1 в последние K лет жизни компании-банкрота ---
fin_cols = [c for c in df.columns if c not in [ID_COL, 'year', 'is_bankrupt', 'bankrupt_company']]

# Последний год с данными у каждого банкрота
bankrupt_rows = df[df['bankrupt_company'] == 1].copy()
has_data = bankrupt_rows[fin_cols].notna().any(axis=1)
last_year_per_company = (
    bankrupt_rows.loc[has_data]
    .groupby(ID_COL)['year'].max()
    .reset_index()
    .rename(columns={{'year': '_last_year'}})
)

df = df.merge(last_year_per_company, on=ID_COL, how='left')
df['is_bankrupt'] = (
    (df['bankrupt_company'] == 1)
    & (df['year'] >= df['_last_year'] - (K_HORIZON - 1))
    & (df['year'] <= df['_last_year'])
).astype(int)
df = df.drop(columns='_last_year')

# --- Сортировка: банкроты первыми, потом действующие ---
df = df.sort_values(
    ['bankrupt_company', ID_COL, 'year'],
    ascending=[False, True, True]
).reset_index(drop=True)

service = ['bankrupt_company', 'is_bankrupt']
cols = [c for c in df.columns if c not in service] + service
df = df[cols]

print(f'Итоговый датасет: {{df.shape}}')
print(f'Годы: {{sorted(df["year"].unique())}}')
print(f'Активных компаний:  {{df[df.bankrupt_company==0][ID_COL].nunique()}}')
print(f'Компаний-банкротов: {{df[df.bankrupt_company==1][ID_COL].nunique()}}')
print(f'Строк с target=1 (окно K={{K_HORIZON}}): {{(df.is_bankrupt==1).sum()}}')
print(f'\\nПоказатели ({{len(fin_cols)}}):')
print(fin_cols)
"""
    _set_code(nb['cells'][idx], new_source)
    _save(nb, path)
    print(f'  RU CELL {idx} patched (target window K={K_HORIZON})')


# -------------------- CN: patch CELL 16 (add target window + default_company) --------------------
def patch_cn(path):
    nb = _load(path)
    # Find save cell (contains 'cn_panel_enriched')
    idx = _find_code(nb, "cn_panel_enriched.csv")
    if idx < 0:
        raise RuntimeError('CN save cell not found')

    new_source = f"""{MARKER}
# Event-based target with K-year horizon window:
# target=1 on the last K years of each DEFAULT company (source_class='default_delisted').
# Strategic delistings (source_class='strategic_delisted') and active companies remain target=0.
K_HORIZON = {K_HORIZON}

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
print(f'OK Saved: {{OUT}}')
print(f'   size: {{OUT.stat().st_size/1024/1024:.1f}} MB')
print(f'   shape: {{cleaned.shape}}')
print(f'   target=1 rows: {{n_pos}} (окно K={{K_HORIZON}}) от {{n_pos_companies}} компаний')
print(f'   default_company=1: {{n_def_companies}} компаний (entity-level)')
print(f'   imbalance (rows) ~= 1:{{int(ratio)}}; recommended class_weight = {{{{0:1, 1:{{ratio:.0f}}}}}}')
"""
    _set_code(nb['cells'][idx], new_source)
    _save(nb, path)
    print(f'  CN CELL {idx} patched (target window K={K_HORIZON} + default_company)')


# -------------------- CN: patch CELL 3 Innovation group (drop rd_to_revenue) --------------------
def patch_cn_features(path):
    nb = _load(path)
    idx = _find_code(nb, "FEATURE_GROUPS = {")
    if idx < 0:
        raise RuntimeError('CN FEATURE_GROUPS cell not found')

    src = ''.join(nb['cells'][idx]['source'])
    marker = '# INNOVATION_GROUP_V2'
    if marker in src:
        print(f'  CN CELL {idx} already has Innovation-group fix (skip)')
        return

    # Replace Innovation group to match Russia (only intangibles_to_assets).
    # rd_to_revenue stays in FEATURES (still used by models) but excluded from H2 group.
    old = "'Innovation':    ['intangibles_to_assets', 'rd_to_revenue'],"
    new = (f"{marker} — симметрично с Россией (только intangibles_to_assets в H2-группе).\n"
           "    # rd_to_revenue остаётся в FEATURES/модели, но не учитывается в H2 сравнении.\n"
           "    'Innovation':    ['intangibles_to_assets'],")
    if old not in src:
        raise RuntimeError(f'Expected old Innovation group not found in CELL {idx}')
    new_src = src.replace(old, new)
    _set_code(nb['cells'][idx], new_src)
    _save(nb, path)
    print(f'  CN CELL {idx} patched (Innovation group matches RU)')


# -------------------- CN: patch CELL 18 best-ensemble by PR-AUC --------------------
def patch_cn_best_ensemble(path):
    nb = _load(path)
    idx = _find_code(nb, "ensemble_scores = res.loc[['Random Forest', 'XGBoost'], 'ROC-AUC test']")
    if idx < 0:
        print('  CN best-ensemble cell already patched or missing')
        return

    new_src = """# BEST_ENSEMBLE_V2 — выбираем по PR-AUC test (адекватная метрика при дисбалансе, как в 20_russia)
ensemble_pr = res.loc[['Random Forest', 'XGBoost'], 'PR-AUC test']
best_name = ensemble_pr.idxmax()
best_model = models[best_name]
print(f'Best ensemble by PR-AUC test: {best_name} '
      f'(PR-AUC={ensemble_pr.max():.4f}, ROC-AUC={res.loc[best_name, "ROC-AUC test"]:.4f})')

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
save_fig(fig, '04_shap_summary'); plt.show()
"""
    _set_code(nb['cells'][idx], new_src)
    _save(nb, path)
    print(f'  CN CELL {idx} patched (best-ensemble by PR-AUC)')


# -------------------- CN: patch CELL 19 — use ensemble_pr in H2 log --------------------
def patch_cn_h2_log(path):
    nb = _load(path)
    idx = _find_code(nb, "log('\\n## H2 — Liquidity+Innovation vs Leverage (SHAP, China)')")
    if idx < 0:
        print('  CN H2-log cell not found (skip)')
        return
    src = ''.join(nb['cells'][idx]['source'])
    old = "log(f'* Best ensemble: **{best_name}** (test ROC-AUC = {ensemble_scores.max():.4f})')"
    new = "log(f'* Best ensemble (by PR-AUC test): **{best_name}** (PR-AUC={ensemble_pr.max():.4f}, ROC-AUC={res.loc[best_name, \"ROC-AUC test\"]:.4f})')"
    if old not in src:
        print(f'  CN CELL {idx} H2-log line not found (maybe already patched)')
        return
    new_src = src.replace(old, new)
    _set_code(nb['cells'][idx], new_src)
    _save(nb, path)
    print(f'  CN CELL {idx} patched (H2 log uses ensemble_pr)')


if __name__ == '__main__':
    import sys
    step = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if step in ('all', 'clean'):
        patch_ru(ROOT / '10_russia_load_and_clean.ipynb')
        patch_cn(ROOT / '30_china_load_and_clean.ipynb')
    if step in ('all', 'models'):
        patch_cn_features(ROOT / '40_china_eda_and_models.ipynb')
        patch_cn_best_ensemble(ROOT / '40_china_eda_and_models.ipynb')
        patch_cn_h2_log(ROOT / '40_china_eda_and_models.ipynb')
    print('Done.')
