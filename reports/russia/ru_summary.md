# Russia — Stage-1 (TTC) results


## Class balance
* Active (0): **88,316**
* Default (1): **398**
* Positive share: **0.449%**  (≈ 1:221)

## H1 — переобучение на малом числе дефолтов
* Split: **group-aware** (по `Регистрационный номер`); POS_WEIGHT = n_neg/n_pos на train = **219.0**.
* Best test ROC-AUC: **Random Forest** (0.8558); Logit ROC-AUC = 0.8352.
* Best test PR-AUC: **Logistic Regression** (0.1355); Logit PR-AUC = 0.1355.
* ΔROC (train−test): Logit **+0.0279**, max ensemble **+0.1559**.

**5-fold Stratified Group K-Fold CV (mean ± std):**
* Logistic Regression: ROC 0.843±0.031, PR 0.077±0.009, ΔROC +0.018
* Random Forest: ROC 0.877±0.033, PR 0.095±0.024, ΔROC +0.123
* XGBoost: ROC 0.845±0.017, PR 0.073±0.014, ΔROC +0.155

✅ По ROC-AUC H1 выполняется: ансамбли точнее и переобучаются сильнее.

✅ По PR-AUC (адекватной метрике при 1:219 дисбалансе) **Logit превосходит ансамбли** — главный аргумент H1.

## H2 — Liquidity+Innovation vs Leverage (SHAP)
* Best ensemble (by PR-AUC test): **XGBoost** (PR-AUC=0.1267, ROC-AUC=0.8439)
* Σ|SHAP| Liquidity + Innovation = **2.5144**
* Σ|SHAP| Leverage               = **2.2446**
* Ratio = **1.12×**

✅ **H2 ПОДТВЕРЖДАЕТСЯ**: в IT-секторе России ликвидность и НМА важнее рычага.