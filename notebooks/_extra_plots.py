"""
Extra defence-ready plots for Russia (K=2), China (K=2) and Russia (K=1).

Produces, per setting:
  * ROC curve overlay (3 models)
  * PR curve overlay with prevalence baseline
  * Top-10 SHAP feature importance (colored by group)
  * Confusion matrix at F1-optimal threshold (best ensemble by PR-AUC)
  * Calibration reliability diagram (best ensemble)
  * 5-fold CV ROC-AUC boxplot

Plus Russia-specific K=1 vs K=2 group-importance side-by-side,
and bankruptcies-by-year distribution for the single-year window.

Files go to reports/{russia,china,russia_k1}/extras/.
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
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve,
                             confusion_matrix, f1_score)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from xgboost import XGBClassifier
import shap

RNG = 42
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data' / 'processed'

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 110

MODEL_COLORS = {'Logistic Regression': '#2E75B6',
                'Random Forest': '#548235',
                'XGBoost': '#C00000'}
GROUP_COLORS = {'Liquidity': '#2E75B6',
                'Innovation': '#7030A0',
                'Leverage': '#C00000',
                'Profitability': '#548235',
                'Size': '#ED7D31'}


# ---------- feature builders ----------------------------------------------
def build_russia_features(df: pd.DataFrame):
    A = 'Активы  всего'; CA = 'Оборотные активы'
    CASH = 'Денежные средства и денежные эквиваленты'
    INT_A = 'Нематериальные активы'; EQ = 'Капитал и резервы'
    LT_L = 'Долгосрочные обязательства'; ST_L = 'Краткосрочные обязательства'
    REV = 'Выручка'; EBIT = 'EBIT'
    NI = 'Чистая прибыль (убыток)'; INTEREST = 'Проценты к уплате'
    CFO = 'Сальдо денежных потоков от текущих операций'

    def sd(a, b): return a / b.replace(0, np.nan)
    td = df[LT_L] + df[ST_L]
    df['current_ratio'] = sd(df[CA], df[ST_L])
    df['cash_to_assets'] = sd(df[CASH], df[A])
    df['cash_to_cl'] = sd(df[CASH], df[ST_L])
    df['wc_to_assets'] = sd(df[CA] - df[ST_L], df[A])
    df['intangibles_to_assets'] = sd(df[INT_A], df[A])
    df['debt_to_assets'] = sd(td, df[A])
    df['debt_to_equity'] = sd(td, df[EQ])
    df['lt_debt_to_assets'] = sd(df[LT_L], df[A])
    df['interest_coverage'] = sd(df[EBIT], df[INTEREST])
    df['roa'] = sd(df[NI], df[A])
    df['net_margin'] = sd(df[NI], df[REV])
    df['operating_margin'] = sd(df[EBIT], df[REV])
    df['cfo_to_assets'] = sd(df[CFO], df[A])
    df['log_assets'] = np.log1p(df[A].clip(lower=0))
    df['log_revenue'] = np.log1p(df[REV].clip(lower=0))
    feats = ['current_ratio', 'cash_to_assets', 'cash_to_cl', 'wc_to_assets',
             'intangibles_to_assets',
             'debt_to_assets', 'debt_to_equity', 'lt_debt_to_assets', 'interest_coverage',
             'roa', 'net_margin', 'operating_margin', 'cfo_to_assets',
             'log_assets', 'log_revenue']
    groups = {
        'Liquidity': ['current_ratio', 'cash_to_assets', 'cash_to_cl', 'wc_to_assets'],
        'Innovation': ['intangibles_to_assets'],
        'Leverage': ['debt_to_assets', 'debt_to_equity', 'lt_debt_to_assets', 'interest_coverage'],
        'Profitability': ['roa', 'net_margin', 'operating_margin', 'cfo_to_assets'],
        'Size': ['log_assets', 'log_revenue'],
    }
    return df, feats, groups


def build_china_features(df: pd.DataFrame):
    def sd(a, b): return a / b.replace(0, np.nan)
    ta = df['total_assets']; tl = df['total_liab']
    rev = df['total_revenue']
    df['current_ratio'] = sd(df['current_assets'], df['current_liab'])
    df['cash_to_assets'] = sd(df['cash'], ta)
    df['cash_to_cl'] = sd(df['cash'], df['current_liab'])
    df['wc_to_assets'] = sd(df['current_assets'] - df['current_liab'], ta)
    df['intangibles_to_assets'] = sd(df['intangibles'], ta)
    df['debt_to_assets'] = sd(df['total_debt'], ta)
    df['debt_to_equity'] = sd(df['total_debt'], df['total_equity'])
    lt = (tl - df['current_liab']).clip(lower=0)
    df['lt_debt_to_assets'] = sd(lt, ta)
    df['interest_coverage'] = sd(df['ebit'], df['interest_expense'])
    df['roa'] = sd(df['net_profit'], ta)
    df['net_margin'] = sd(df['net_profit'], rev)
    df['operating_margin'] = sd(df['ebit'], rev)
    df['cfo_to_assets'] = sd(df['cfo'], ta)
    df['log_assets'] = np.log1p(ta.clip(lower=0))
    df['log_revenue'] = np.log1p(rev.clip(lower=0))
    feats = ['current_ratio', 'cash_to_assets', 'cash_to_cl', 'wc_to_assets',
             'intangibles_to_assets',
             'debt_to_assets', 'debt_to_equity', 'lt_debt_to_assets', 'interest_coverage',
             'roa', 'net_margin', 'operating_margin', 'cfo_to_assets',
             'log_assets', 'log_revenue']
    groups = {
        'Liquidity': ['current_ratio', 'cash_to_assets', 'cash_to_cl', 'wc_to_assets'],
        'Innovation': ['intangibles_to_assets'],
        'Leverage': ['debt_to_assets', 'debt_to_equity', 'lt_debt_to_assets', 'interest_coverage'],
        'Profitability': ['roa', 'net_margin', 'operating_margin', 'cfo_to_assets'],
        'Size': ['log_assets', 'log_revenue'],
    }
    return df, feats, groups


# ---------- preprocessing / split / fit -----------------------------------
def prep_and_split(panel, feats, id_col, target):
    panel[feats] = panel[feats].replace([np.inf, -np.inf], np.nan)
    for c in feats:
        lo, hi = panel[c].quantile([0.01, 0.99])
        panel[c] = panel[c].clip(lo, hi)
    panel[feats] = panel[feats].fillna(panel[feats].median(numeric_only=True))

    X = panel[feats].values
    y = panel[target].values
    grp = panel[id_col].values

    lbl = panel.groupby(id_col)[target].max()
    comp_b = np.array(lbl[lbl == 1].index.values, copy=True)
    comp_a = np.array(lbl[lbl == 0].index.values, copy=True)
    r = np.random.default_rng(RNG); r.shuffle(comp_b); r.shuffle(comp_a)

    def _s(arr, f=0.2):
        n = int(round(len(arr) * f))
        return arr[n:], arr[:n]

    tr_b, te_b = _s(comp_b); tr_a, te_a = _s(comp_a)
    tr_ids = set(tr_b) | set(tr_a); te_ids = set(te_b) | set(te_a)
    m_tr = panel[id_col].isin(tr_ids).values
    m_te = panel[id_col].isin(te_ids).values
    return (X[m_tr], y[m_tr], grp[m_tr],
            X[m_te], y[m_te], grp[m_te],
            X, y, grp)


def make_models(pw):
    logit = Pipeline([('sc', StandardScaler()),
                      ('clf', LogisticRegression(max_iter=2000,
                                                 class_weight={0: 1, 1: pw},
                                                 solver='liblinear',
                                                 random_state=RNG))])
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                class_weight={0: 1, 1: pw}, n_jobs=-1,
                                random_state=RNG)
    xgb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9,
                        scale_pos_weight=pw, eval_metric='aucpr',
                        tree_method='hist', random_state=RNG, n_jobs=-1)
    return {'Logistic Regression': logit, 'Random Forest': rf, 'XGBoost': xgb}


def get_proba(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


# ---------- plotting ------------------------------------------------------
def plot_roc(models, Xte, yte, title, out):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for name, m in models.items():
        p = get_proba(m, Xte)
        fpr, tpr, _ = roc_curve(yte, p)
        auc = roc_auc_score(yte, p)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})',
                color=MODEL_COLORS[name], lw=2)
    ax.plot([0, 1], [0, 1], '--', color='grey', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(title); ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)


def plot_pr(models, Xte, yte, title, out):
    prev = float(np.mean(yte))
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for name, m in models.items():
        p = get_proba(m, Xte)
        pr, rc, _ = precision_recall_curve(yte, p)
        ap = average_precision_score(yte, p)
        ax.plot(rc, pr, label=f'{name} (AP={ap:.3f})',
                color=MODEL_COLORS[name], lw=2)
    ax.axhline(prev, ls='--', color='grey', lw=1,
               label=f'Baseline = prevalence ({prev*100:.2f}%)')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(title); ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)


def plot_top10_shap(best_model, Xte, feats, groups, title, out):
    exp = shap.TreeExplainer(best_model)
    sv = exp.shap_values(Xte)
    if isinstance(sv, list):
        sv = sv[1]
    elif hasattr(sv, 'ndim') and sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[-10:]
    sel_feats = [feats[i] for i in order]
    sel_imp = imp[order]
    feat2grp = {f: g for g, lst in groups.items() for f in lst}
    colors = [GROUP_COLORS[feat2grp[f]] for f in sel_feats]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(sel_feats, sel_imp, color=colors)
    ax.set_xlabel('mean |SHAP|'); ax.set_title(title)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
               if g in {feat2grp[f] for f in sel_feats}]
    ax.legend(handles=handles, loc='lower right', fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
    return sv, imp


def plot_confusion(best_model, Xte, yte, title, out):
    p = get_proba(best_model, Xte)
    best_thr, best_f1 = 0.5, -1
    for thr in np.linspace(0.01, 0.99, 99):
        pred = (p >= thr).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(yte, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_thr = thr
    pred = (p >= best_thr).astype(int)
    cm = confusion_matrix(yte, pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'], ax=ax)
    ax.set_title(f'{title}\nthreshold={best_thr:.2f}, F1={best_f1:.3f}')
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
    return best_thr, best_f1, cm


def plot_calibration(best_model, Xte, yte, title, out):
    p = get_proba(best_model, Xte)
    # stratified bins for imbalanced data
    n_bins = 10
    try:
        prob_true, prob_pred = calibration_curve(yte, p, n_bins=n_bins, strategy='quantile')
    except ValueError:
        prob_true, prob_pred = calibration_curve(yte, p, n_bins=5, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], '--', color='grey', label='Perfect')
    ax.plot(prob_pred, prob_true, 'o-', color='#C00000', label='Model')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)


def plot_cv_box(X, y, grp, models_proto, title, out):
    skgf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
    rows = []
    for name, proto in models_proto.items():
        for tr, te in skgf.split(X, y, groups=grp):
            yt = y[tr]
            pw = float((yt == 0).sum() / max((yt == 1).sum(), 1))
            m = clone(proto)
            if isinstance(m, Pipeline):
                m.set_params(clf__class_weight={0: 1, 1: pw})
            elif isinstance(m, RandomForestClassifier):
                m.set_params(class_weight={0: 1, 1: pw})
            elif isinstance(m, XGBClassifier):
                m.set_params(scale_pos_weight=pw)
            m.fit(X[tr], yt)
            pr = get_proba(m, X[te])
            rows.append({'Model': name,
                         'ROC-AUC': roc_auc_score(y[te], pr),
                         'PR-AUC': average_precision_score(y[te], pr)})
    cv_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ['ROC-AUC', 'PR-AUC']):
        sns.boxplot(data=cv_df, x='Model', y=metric, ax=ax,
                    palette=[MODEL_COLORS[n] for n in cv_df['Model'].unique()])
        sns.stripplot(data=cv_df, x='Model', y=metric, ax=ax,
                      color='black', size=4, alpha=0.6)
        ax.set_title(f'{metric} — 5-fold StratifiedGroupKFold')
        ax.set_xlabel('')
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
    return cv_df


# ---------- runner --------------------------------------------------------
def run_setting(label, panel, id_col, target, build_fn, out_dir, title_prefix):
    out_dir.mkdir(parents=True, exist_ok=True)
    panel, feats, groups = build_fn(panel)
    X_tr, y_tr, g_tr, X_te, y_te, g_te, X_all, y_all, g_all = \
        prep_and_split(panel, feats, id_col, target)
    pw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f'\n[{label}] n_tr={len(y_tr)} pos={int(y_tr.sum())} | '
          f'n_te={len(y_te)} pos={int(y_te.sum())} | POS_W={pw:.1f}')
    models = make_models(pw)
    for nm, m in models.items():
        m.fit(X_tr, y_tr)

    # plots
    plot_roc(models, X_te, y_te, f'{title_prefix} — ROC curves (test)',
             out_dir / f'{label}_roc_curves.png')
    plot_pr(models, X_te, y_te, f'{title_prefix} — PR curves (test)',
            out_dir / f'{label}_pr_curves.png')

    # best ensemble by PR-AUC test
    ens_pr = {n: average_precision_score(y_te, get_proba(m, X_te))
              for n, m in models.items() if n != 'Logistic Regression'}
    best_name = max(ens_pr, key=ens_pr.get)
    best_model = models[best_name]
    print(f'  best ensemble (PR-AUC): {best_name} = {ens_pr[best_name]:.3f}')

    plot_top10_shap(best_model, X_te, feats, groups,
                    f'{title_prefix} — Top-10 SHAP ({best_name})',
                    out_dir / f'{label}_top10_shap.png')
    thr, f1, cm = plot_confusion(best_model, X_te, y_te,
                                  f'{title_prefix} — Confusion matrix ({best_name})',
                                  out_dir / f'{label}_confusion_matrix.png')
    plot_calibration(best_model, X_te, y_te,
                     f'{title_prefix} — Calibration ({best_name})',
                     out_dir / f'{label}_calibration.png')
    cv_df = plot_cv_box(X_all, y_all, g_all, make_models(pw),
                        f'{title_prefix} — CV distribution',
                        out_dir / f'{label}_cv_boxplot.png')
    cv_df.to_csv(out_dir / f'{label}_cv_folds.csv', index=False,
                 encoding='utf-8-sig')

    return {'best': best_name, 'thr': thr, 'f1': f1, 'cm': cm,
            'prev': float(np.mean(y_te))}


# ========== Russia K=2 ====================================================
ru = pd.read_csv(PROCESSED / 'ru_panel_cleaned.csv', encoding='utf-8-sig')
info_ru_k2 = run_setting('ru_k2', ru.copy(),
                         'Регистрационный номер', 'is_bankrupt',
                         build_russia_features,
                         ROOT / 'reports' / 'russia' / 'extras',
                         'Russia K=2')

# ========== China K=2 =====================================================
cn = pd.read_csv(PROCESSED / 'cn_panel_enriched.csv', encoding='utf-8-sig')
info_cn = run_setting('cn_k2', cn.copy(),
                      'ticker', 'target',
                      build_china_features,
                      ROOT / 'reports' / 'china' / 'extras',
                      'China K=2')

# ========== Russia K=1 ====================================================
ru_k1 = pd.read_csv(PROCESSED / 'ru_panel_cleaned.csv', encoding='utf-8-sig')
bmask = ru_k1['bankrupt_company'] == 1
last = ru_k1.loc[bmask].groupby('Регистрационный номер')['year'].max().rename('_ly')
ru_k1 = ru_k1.merge(last, on='Регистрационный номер', how='left')
ru_k1['is_bankrupt'] = ((ru_k1['bankrupt_company'] == 1) &
                        (ru_k1['year'] == ru_k1['_ly'])).astype(int)
info_ru_k1 = run_setting('ru_k1', ru_k1,
                         'Регистрационный номер', 'is_bankrupt',
                         build_russia_features,
                         ROOT / 'reports' / 'russia_k1' / 'extras',
                         'Russia K=1')


# ---------- RU K=1 vs K=2 side-by-side group importance -------------------
cmp = pd.read_csv(ROOT / 'reports/russia_k1/ru_k1_vs_k2_groups.csv', index_col=0)
cmp_norm = cmp.copy()
cmp_norm['K=1 share'] = cmp['K=1'] / cmp['K=1'].sum()
cmp_norm['K=2 share'] = cmp['K=2'] / cmp['K=2'].sum()
fig, ax = plt.subplots(figsize=(8, 4.8))
idx = np.arange(len(cmp_norm))
w = 0.35
ax.bar(idx - w/2, cmp_norm['K=1 share'], w, label='K=1 (last year only)',
       color='#2E75B6')
ax.bar(idx + w/2, cmp_norm['K=2 share'], w, label='K=2 (last 2 years)',
       color='#C00000')
ax.set_xticks(idx); ax.set_xticklabels(cmp_norm.index, rotation=20)
ax.set_ylabel('Share of total Σ|SHAP|')
ax.set_title('Russia — SHAP group share: K=1 vs K=2 (normalized)')
ax.legend()
plt.tight_layout()
out = ROOT / 'reports/russia_k1/extras/ru_k1_vs_k2_group_shares.png'
fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
cmp_norm.round(4).to_csv(ROOT / 'reports/russia_k1/extras/ru_k1_vs_k2_group_shares.csv',
                         encoding='utf-8-sig')
print('saved:', out)


# ---------- RU defaults-by-year (K=1 vs K=2 window coverage) --------------
df_year = pd.read_csv(PROCESSED / 'ru_panel_cleaned.csv', encoding='utf-8-sig')
bmask = df_year['bankrupt_company'] == 1
last2 = df_year.loc[bmask].groupby('Регистрационный номер')['year'].max().rename('_ly')
df_year = df_year.merge(last2, on='Регистрационный номер', how='left')
df_year['k1'] = ((df_year['bankrupt_company'] == 1) & (df_year['year'] == df_year['_ly'])).astype(int)
df_year['k2'] = ((df_year['bankrupt_company'] == 1) &
                 (df_year['year'] >= df_year['_ly'] - 1) &
                 (df_year['year'] <= df_year['_ly'])).astype(int)
yr = df_year.groupby('year')[['k1', 'k2']].sum().astype(int)
fig, ax = plt.subplots(figsize=(9, 4.8))
yr.plot.bar(ax=ax, color=['#2E75B6', '#C00000'], width=0.85)
ax.set_ylabel('Positive labels (company-year)')
ax.set_title('Russia — distribution of defaults by year: K=1 vs K=2 window')
ax.legend(['K=1 (last year)', 'K=2 (last 2 years)'])
plt.xticks(rotation=0)
plt.tight_layout()
out = ROOT / 'reports/russia_k1/extras/ru_defaults_by_year.png'
fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
yr.to_csv(ROOT / 'reports/russia_k1/extras/ru_defaults_by_year.csv',
          encoding='utf-8-sig')
print('saved:', out)

print('\nALL EXTRAS DONE')
print(' russia/extras:   ', sorted(p.name for p in (ROOT/'reports/russia/extras').iterdir()))
print(' china/extras:    ', sorted(p.name for p in (ROOT/'reports/china/extras').iterdir()))
print(' russia_k1/extras:', sorted(p.name for p in (ROOT/'reports/russia_k1/extras').iterdir()))
