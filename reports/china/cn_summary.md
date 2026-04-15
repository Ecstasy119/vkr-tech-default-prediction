# China — Stage-1 (TTC) results


## Class balance
* Active (0): **3,412**
* Default (1): **48**
* Positive share: **1.387%**  (≈ 1:71)

## H1 — overfitting на малом числе дефолтов (China)
* Best test ROC-AUC: **Random Forest** (1.0000)
* Logit ΔROC (train−test): **+0.0010**
* Max ensemble ΔROC:        **+0.0006**

⚠️ H1 частично: ансамбли точнее, но Logit НЕ стабильнее (Δ сопоставимы).

## H2 — Liquidity+Innovation vs Leverage (SHAP, China)
* Best ensemble: **Random Forest** (test ROC-AUC = 1.0000)
* Σ|SHAP| Liquidity + Innovation = **0.2934**
* Σ|SHAP| Leverage               = **0.0394**
* Ratio = **7.45×**

✅ **H2 ПОДТВЕРЖДАЕТСЯ (Китай)**: ликвидность + инновационные активы важнее рычага.