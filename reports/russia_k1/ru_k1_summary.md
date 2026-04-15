# Russia - K=1 sensitivity (single year before bankruptcy)

* Rows (company-year): **88,714**
* Bankrupt companies:  **206**
* Positives (K=1):     **206**  (share 0.232%)
* POS_WEIGHT (train) = **428.4**
* Train defaults: **165** / test defaults: **41**

## H1 (K=1) - small-sample overfitting signature
* Best test ROC-AUC: **XGBoost** (0.8833)
* Best test PR-AUC:  **Logistic Regression**  (0.1140)
* Logit test ROC=0.8370, PR=0.1140, ΔROC=+0.0416
* Max ensemble ΔROC (train−test) = **+0.1383** (overfit signal)

**CV (mean ± std):**
* Logistic Regression: ROC 0.851±0.017, PR 0.060±0.015
* Random Forest: ROC 0.871±0.024, PR 0.045±0.012
* XGBoost: ROC 0.854±0.011, PR 0.043±0.024

## H2 (K=1) - Liquidity+Innovation vs Leverage (SHAP)
* Best ensemble (PR-AUC): **Random Forest**
* Top features: cash_to_cl, operating_margin, debt_to_assets, log_assets, current_ratio
* Group Sum|SHAP|:
  * Liquidity: **0.1793**
  * Profitability: **0.1281**
  * Leverage: **0.1200**
  * Size: **0.0653**
  * Innovation: **0.0082**
* Liquidity+Innovation = **0.1875**
* Leverage            = **0.1200**
* Ratio = **1.56x**

[OK] **H2 (K=1) confirmed**: Liquidity+Innovation dominates Leverage under single-year window.

## K=1 vs K=2 comparison
Group Sum|SHAP| (K=1 vs K=2):
* Liquidity: K=1 0.179  vs  K=2 2.345  (Δ=-2.165)
* Profitability: K=1 0.128  vs  K=2 1.749  (Δ=-1.620)
* Leverage: K=1 0.120  vs  K=2 2.245  (Δ=-2.125)
* Size: K=1 0.065  vs  K=2 1.511  (Δ=-1.446)
* Innovation: K=1 0.008  vs  K=2 0.170  (Δ=-0.162)