# Russia — Stage-1 (TTC) results


## Class balance
* Active (0): **88,508**
* Default (1): **206**
* Positive share: **0.232%**  (≈ 1:429)

## H1 — переобучение на малом числе дефолтов
* Split: **group-aware** (по `Регистрационный номер`); POS_WEIGHT = n_neg/n_pos на train = **428.4**.
* Best test ROC-AUC: **XGBoost** (0.8833); Logit ROC-AUC = 0.8370.
* Best test PR-AUC: **Logistic Regression** (0.1140); Logit PR-AUC = 0.1140.
* ΔROC (train−test): Logit **+0.0416**, max ensemble **+0.1383**.

**5-fold Stratified Group K-Fold CV (mean ± std):**
* Logistic Regression: ROC 0.851±0.017, PR 0.060±0.015, ΔROC +0.025
* Random Forest: ROC 0.871±0.024, PR 0.045±0.012, ΔROC +0.129
* XGBoost: ROC 0.854±0.011, PR 0.043±0.024, ΔROC +0.146

✅ По ROC-AUC H1 выполняется: ансамбли точнее и переобучаются сильнее.

✅ По PR-AUC (адекватной метрике при 1:428 дисбалансе) **Logit превосходит ансамбли** — главный аргумент H1.

## H2 — Liquidity+Innovation vs Leverage (SHAP)
* Best ensemble (by PR-AUC test): **Random Forest** (PR-AUC=0.0851, ROC-AUC=0.8614)
* Σ|SHAP| Liquidity + Innovation = **0.1875**
* Σ|SHAP| Leverage               = **0.1200**
* Ratio = **1.56×**

✅ **H2 ПОДТВЕРЖДАЕТСЯ**: в IT-секторе России ликвидность и НМА важнее рычага.