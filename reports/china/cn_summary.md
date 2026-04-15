# China — Stage-1 (TTC) results


## Class balance
* Active (0): **3,439**
* Default (1): **19**
* Positive share: **0.549%**  (≈ 1:181)

## H1 — overfitting на малом числе дефолтов (China)
* Best test ROC-AUC: **Logistic Regression** (1.0000)
* Logit ΔROC (train−test): **-0.0028**
* Max ensemble ΔROC:        **+0.0037**

⚠️ H1 частично: Logit сопоставим по точности, но ансамбли переобучены.

## H2 — Liquidity+Innovation vs Leverage (SHAP, China)
* Best ensemble (by PR-AUC test): **Random Forest** (PR-AUC=0.7500, ROC-AUC=0.9985)
* Σ|SHAP| Liquidity + Innovation = **0.1701**
* Σ|SHAP| Leverage               = **0.1118**
* Ratio = **1.52×**

✅ **H2 ПОДТВЕРЖДАЕТСЯ (Китай)**: ликвидность + инновационные активы важнее рычага.