# Assessment of Financial Stability for Tech Companies in Emerging Markets

ВКР: оценка финансовой устойчивости и прогноз дефолта IT-компаний на развивающихся рынках (Россия и Китай) с помощью моделей машинного обучения и Explainable AI (SHAP).

## Суть работы

Классические модели банкротства (Altman Z-score и т.п.) разработаны для индустриальных компаний с «твёрдыми» активами и плохо работают для asset-light IT-сектора, где ключевую роль играют ликвидность, нематериальные активы и кэш. В работе строятся локализованные ML-модели (Logistic Regression, Random Forest, XGBoost) по двухэтапной схеме **TTC → PIT** (Through-the-Cycle → Point-in-Time), а интерпретация выполняется через SHAP.

### Гипотезы
- **H1.** Ансамбли (RF, XGBoost) дают более высокий ROC-AUC, но Logistic Regression устойчивее к переобучению на малой доле дефолтов.
- **H2.** В IT-секторе ликвидность и нематериальные активы важнее классического финансового рычага (проверяется через Σ|SHAP|).
- **H3.** Добавление макро-факторов (PIT-калибровка) повышает качество прогноза по сравнению с чистой TTC-моделью.

### Данные
- **Россия:** SPARK, ~10 000 действующих компаний, ~200 дефолтов, 2012–2024.
- **Китай:** WIND, ~319 действующих, ~37 проблемных, 2012–2024.
- Бразилия, Индия, ЮАР исключены из-за недостаточного объёма и качества данных.
- Исходные данные **не** публикуются в репозитории (см. `.gitignore`).

## Структура репозитория

```
├── notebooks/                   # Jupyter-пайплайн
│   ├── 10_russia_load_and_clean.ipynb
│   ├── 20_russia_eda_and_models.ipynb
│   ├── 30_china_load_and_clean.ipynb
│   ├── 40_china_eda_and_models.ipynb
│   ├── 50_cross_country_pit.ipynb
│   ├── _build_notebooks.py      # сборщик/регенератор ноутбуков
│   └── data_audit.py            # аудит входных данных
├── src/                         # общий код (метрики, препроцессинг)
├── reports/
│   ├── russia/                  # метрики, SHAP, фигуры по РФ
│   ├── china/                   # метрики, SHAP, фигуры по КНР
│   ├── cross_country/           # кросс-страновое сравнение (H3)
│   ├── notebook_exports/        # PDF-экспорты ноутбуков
│   ├── DATA_AUDIT_NARRATIVE.md
│   └── VKR_full_report.md
├── data/                        # данные (игнорируются git)
│   ├── raw/                     # исходники SPARK / WIND
│   └── processed/               # очищенные панели (ru_panel_cleaned, cn_panel_cleaned)
├── project_passport.txt         # развёрнутый паспорт проекта
├── requirements.txt
└── main.py
```

## Установка и запуск

1. Создать и активировать виртуальное окружение:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1     # Windows PowerShell
   # source .venv/bin/activate      # Linux/macOS
   ```
2. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Положить исходные данные SPARK / WIND в `data/raw/<country>/{active,bankrupt}/` (они игнорируются git и не должны попадать в репозиторий).
4. Прогнать ноутбуки по порядку номеров: `10_` → `20_` → `30_` → `40_` → `50_`.

## Ключевые результаты

- Russia (TTC): Logistic Regression — PR-AUC 0.114, ΔROC(train−test) +0.042; XGBoost — ROC 0.883 при ΔROC +0.138. Подтверждает H1: ансамбли точнее, но переобучаются сильнее.
- Russia (SHAP): Σ|SHAP| Liquidity+Innovation = 0.188 против Leverage = 0.120 (×1.56) → H2 подтверждается.
- Полные метрики и кросс-страновое сравнение — в `reports/`.

## Лицензия / использование

Репозиторий предназначен для научной работы (ВКР). Первичные данные (SPARK, WIND) распространяются под лицензиями правообладателей и в репозиторий не включаются.
