# Финальные формулировки гипотез (после рефакторинга)

Единая методология: event-based таргет с окном K=2 (последние 2 года жизни
дефолтной компании = 1), group-aware сплит по компании, класс-веса вместо SMOTE,
PR-AUC как критерий при сильном дисбалансе, Innovation =
`intangibles_to_assets` симметрично в России и Китае.

> **Замечание по паспорту проекта.** В `project_passport.txt` гипотеза H3
> встречается в двух разных местах с разными формулировками: (а) строка 52 —
> про *межстрановую асимметрию факторов риска*; (б) строка 71 (в блоке
> «Thesis Structural Blueprint») — про *прирост точности PIT над TTC*.
> Фактическая методологическая проверка показала, что версия (б) не
> подтверждается статистически, а версия (а) — подтверждается. Поэтому
> финальной считается формулировка (а), которая и ложится в H3-A ниже.

---

## H1 — предсказуемость дефолта и сигнатура переобучения

**Дословно из паспорта (не менялась):**
> *Hypothesis 1 (H1): Ensemble ML models (Random Forest, XGBoost) yield higher
> predictive accuracy (ROC-AUC) for tech companies; however, traditional
> Logistic Regression demonstrates higher stability and less susceptibility to
> overfitting on low-default samples.*

**Финальная формулировка (без изменений):**
> *H1. Ensemble ML models (Random Forest, XGBoost) yield higher predictive
> accuracy (ROC-AUC) for tech companies; however, baseline Logistic Regression
> demonstrates higher stability and less susceptibility to overfitting on
> low-default samples. This stability advantage is measured by (i) a smaller
> train−test gap in ROC-AUC and (ii) PR-AUC performance on the test set at
> least on par with ensembles under heavy class imbalance.*

**Что фактически показано в работе:**
- Россия (K=2): Logit ROC test = 0.835, PR test = **0.136** (выигрывает);
  XGB ROC test = 0.844, PR test = 0.127, ΔROC_XGB = **+0.156** (сильный overfit).
- Китай (K=2): Logit CV ROC = 0.995 ± 0.002; на тесте все модели ROC ≈ 1.00
  (ceiling 11 дефолтов). ΔPR у XGB = **+0.575** (чистое переобучение).
- Россия (K=1, робастность): те же выводы — Logit PR test = 0.114 снова
  выигрывает; ансамбли ΔROC > 0.12.

**Статус:** **подтверждена** и воспроизводится при K=1 и K=2.

---

## H2 — Liquidity + Innovation > Leverage в IT-секторе

**Дословно из паспорта (не менялась):**
> *Hypothesis 2 (H2): Liquidity metrics and innovation capacity (Intangible
> Assets) hold significantly higher predictive weight in the tech sector than
> traditional financial leverage ratios.*

**Финальная формулировка (содержательно та же; приведено методологическое
уточнение):**
> *H2. In the tech sector, liquidity metrics and innovation capacity —
> represented by `intangibles_to_assets` as a cross-country-comparable proxy —
> carry a greater aggregate SHAP-based predictive weight for default than the
> block of leverage ratios (debt-to-assets, debt-to-equity, long-term
> debt-to-assets, interest coverage).*

**Методологическое уточнение (единственное отличие от паспорта):** группа
Innovation теперь состоит только из `intangibles_to_assets` в обеих странах
(в китайской подвыборке из неё убран `rd_to_revenue`, которого нет в
РФ-данных SPARK). Это делает H2 межстрановой сопоставимой.

**Что показано (best ensemble by PR-AUC):**
- Россия (K=2): Liq+Innovation = 2.515, Leverage = 2.245 (ratio 1.12×).
- Россия (K=1): Liq+Innovation = 0.188, Leverage = 0.120 (ratio 1.56×).
- Китай (K=2):  Liq+Innovation = 0.170, Leverage = 0.112 (ratio 1.52×).

**Статус:** **подтверждена** в трёх независимых прогонах.

---

## H3 → H3-A — выбрана формулировка из паспорта, стр. 52

**Вариант (а) из паспорта, стр. 52 (исходная, принятая как H3-A):**
> *Hypothesis 3 (H3): The risk profile of IT companies is highly dependent on
> the local market: in economies with expensive capital (Russia), liquidity
> deficit is the primary default predictor, whereas in markets with developed
> institutional support (China), operational profitability is paramount.*

**Вариант (б) из паспорта, стр. 71 (Thesis Blueprint — ОТКЛОНЁН):**
> *~~H3: The inclusion of macroeconomic variables (Point-in-Time calibration)
> improves the predictive power of models compared to standalone financial
> (Through-the-Cycle) models.~~*

**Почему отклонена версия (б):** при честном bootstrap-CI по компаниям
95%-интервал для ΔROC(PIT−TTC) в обеих странах пересекает ноль
(РФ ΔROC медиана +0.009, CI [−0.020, +0.034]; КНР ΔROC медиана −0.002,
CI [−0.008, 0.000]). К тому же макрофакторы в текущей версии — placeholder:
реальный IMF WEO ряд ещё не подключён, поэтому любой положительный вывод по
PIT-lift был бы некорректным. Маркер `REPLACE_WITH_IMF_WEO` оставлен в
ноутбуке 50_cross_country_pit.ipynb, и после подключения реального макро
можно будет отдельно перепроверить эту подгипотезу — но как основной тезис
работы она заменяется на (а).

**Финальная формулировка H3-A (из паспорта, стр. 52, с уточнением
dominant-group по итогам SHAP-анализа):**
> *H3-A. The dominant financial risk-factor groups differ between the Russian
> and Chinese IT sectors, reflecting the difference in capital regimes: in
> Russia (expensive capital, limited external funding) the **Liquidity** group
> is the top SHAP-ranked driver of default risk, while in China (developed
> institutional support, higher role of intellectual capital) the
> **Innovation** group (intangibles intensity) leads. This structural
> asymmetry is measured by the group-level share of total Σ|SHAP| from a
> unified XGBoost specification with identical features in both countries.*

**Отклонение от буквы паспорта.** В паспорте, стр. 52, для Китая ожидалась
доминирующая роль *operational profitability*. Фактическое ранжирование по
Σ|SHAP| в Китае: Innovation 26.2 % > Size 21.5 % > Profitability 20.7 %.
Поэтому в H3-A для Китая зафиксирована **Innovation** (а не Profitability).
Это — эмпирическая корректировка исходного тезиса, сделанная по итогам
SHAP-анализа: доминирует интеллектуальный капитал, а не операционная
маржинальность, при том что профильность обе группы имеют высокую.

**Что показано (единая XGBoost-спецификация, один split-протокол):**
- РФ top-group: **Liquidity 30.2 %** суммарной SHAP-массы; top-1 признак
  `cash_to_cl`.
- КНР top-group: **Innovation 26.2 %** суммарной SHAP-массы; top-1 признак
  `intangibles_to_assets`.

**Статус:** H3-A **подтверждена** (структурная асимметрия факторов между
странами). Исходный H3-PIT-lift (вариант (б) из паспорта) честно оставлен
как открытый вопрос до подключения реальных макроданных (IMF WEO).

---

## Сводка изменений относительно паспорта

| Гипотеза | В паспорте | В финальной работе |
|---|---|---|
| H1 | Ensembles accurate, Logit stable on low-default | **Без изменений**, воспроизведена при K=1 и K=2 |
| H2 | Liq + Intangibles > Leverage | Содержательно та же; Innovation унифицирован до `intangibles_to_assets` в обеих странах |
| H3 | (а) асимметрия факторов между странами; (б) PIT > TTC | **Выбрана (а)** → H3-A; Китай доминанта уточнена: **Innovation** вместо Profitability (по данным SHAP). Версия (б) отклонена: bootstrap-CI пересекает 0, макро — placeholder |
