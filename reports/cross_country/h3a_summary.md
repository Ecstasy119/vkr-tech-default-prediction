# H3-A - Cross-Country Asymmetric Feature Importance

## TTC single-split (group-aware company-level 80/20)
* **Russia**: ROC-AUC test=0.8291 (95%-CI [0.753, 0.886]), PR-AUC test=0.0888 (95%-CI [0.021, 0.174])
* **China**: ROC-AUC test=1.0000 (95%-CI [1.000, 1.000]), PR-AUC test=1.0000 (95%-CI [1.000, 1.000])

## PIT macro-lift (placeholder macro - cautious)
* Russia dROC median=+0.0087, 95%-CI=[-0.0198, +0.0344]
* China  dROC median=-0.0020, 95%-CI=[-0.0079, +0.0000]
* Russia significant lift: False
* China  significant lift: False
> Placeholder macro - real H3-lift conclusions only after IMF WEO swap.

## H3-A group-importance structure (main test)
* **Russia top group:** Liquidity (30.2% of total SHAP mass)
* **China  top group:** Innovation (26.2% of total SHAP mass)
* Russia top-3 features: cash_to_cl, debt_to_assets, log_assets
* China  top-3 features: intangibles_to_assets, log_revenue, net_margin

[OK] H3-A CONFIRMED: dominant risk-factor groups differ - Liquidity in Russia vs Innovation in China. Structural asymmetry matches the capital-regime hypothesis.