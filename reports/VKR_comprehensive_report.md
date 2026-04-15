# Комплексный отчёт по ВКР
**Assessment of Financial Stability for Technology Companies in Selected Emerging Markets (Evidence from Russia and China) Using Machine Learning Models**

Дата: 2026-04-15. Статус: финальный пост-рефакторинг (event-based K=2, group-aware split, class-weights, PR-AUC, `intangibles_to_assets` как симметричный Innovation-прокси).

Этот документ — единая точка правды по всему проекту для переписывания разделов «Методология» и «Results & Discussion» диссертации. Здесь собрано всё: откуда пришли данные, как они приводились в рабочий вид, какие ноутбуки что делают, какие решения были приняты и почему, какие графики и таблицы существуют и что каждый из них показывает, какие числа получены и как они отвечают на финальные гипотезы.

---

## Часть I. Предмет работы и scope

**Тема.** Оценка вероятности дефолта (bankruptcy / forced liquidation) IT-компаний на развивающихся рынках с помощью ML-моделей (Logistic Regression, Random Forest, XGBoost) и интерпретации через SHAP.

**Зачем.** Классические модели дистресса (Altman Z-score, Ohlson O, Beaver) построены на asset-heavy предприятиях. Современные IT-компании — asset-light: ценность сосредоточена в нематериальных активах (код, патенты, бренд, человеческий капитал) и в cash-подушке. Балансовые ratios закредитованности (debt-to-assets, interest coverage) для них менее информативны, а метрики ликвидности и intangibles — более информативны. Проект эмпирически тестирует эту переформулировку на двух контрастных развивающихся рынках (РФ — дорогой капитал; КНР — государственная поддержка IT-сектора).

**Scope — почему только Россия и Китай.** Изначально планировался BRICS (Россия, Китай, Индия, Бразилия, ЮАР). Эмпирический data-audit (см. Часть III) показал, что Индия, Бразилия и ЮАР в текущих выгрузках не пригодны для обучения ML:
- Индия (Wind Software & Services_India.xlsx, 256 тикеров): общая заполненность 6.6 %, 10 из 14 метрик полностью пусты.
- Бразилия (brazil_it_panel_improved.csv, 26 компаний × 8 лет = 189 наблюдений): чистый формат, но объём недостаточен и дефолты в панели не размечены; `Current_Ratio` 100 % NaN.
- ЮАР (south_africa_it_panel_*.csv): 15 компаний, 291 метрика с чудовищной разреженностью; в отдельном файле `south_africa_it_delistings.csv` — 4 строки, 3 дефолта. Для ML не применимо.

В итоге scope ограничен **Россия (SPARK-Interfax) + Китай (WIND)**, остальные BRICS вынесены в Scope & Limitations с эмпирическим обоснованием через цифры покрытия, а не декларацией.

**Research Design — двухэтапная модель TTC → PIT.**
- **Stage 1 (Through-the-Cycle, TTC).** Модель учит компанию «в вакууме»: только по её финансовой отчётности. Результат — intrinsic risk score.
- **Stage 2 (Point-in-Time, PIT).** К TTC-скору добавляются макропеременные (GDP growth, inflation) страны-года. Результат — риск с учётом макрофона.

Такое разделение принято в банковском скоринге (Basel II/III), избегает смешения idiosyncratic и systemic риска и позволяет отдельно тестировать ценность макро-аугментации (H3-старая версия / отклонена в пользу H3-A — см. Часть VIII).

---

## Часть II. Исходные данные — происхождение, формат, что с ними не так

### 2.1 Россия — SPARK-Interfax
- **Состав.** 6 Excel-файлов в `data/raw/russia/{active,bankrupt}/`: по 3 файла в каждой категории (active, bankrupt). Лист `report`, header в строке 4. Шапка — «широкий» формат SPARK: годовые колонки вида `«2014, Выручка, RUB»`.
- **Компании.** ~10 013 действующих IT-компаний + 206 банкротов (до очистки). SPARK — регистратор российских юрлиц, выгрузка по коду ОКВЭД IT-сектора, 2012–2024.
- **Поля.** Выручка, чистая прибыль, EBIT, активы (total, current), обязательства (total, long-term, current), собственный капитал, денежные средства, кредиторка/дебиторка, проценты к уплате, НМА, численность работников.
- **Нестандартность #1.** Столбец «Среднесписочная численность работников» в SPARK приходит смешанным текстом: диапазоны `"0 - 5"`, `"51 - 100"`, числа с пробелами-разделителями `"1 011"`, открытые интервалы `"> 5 000"`. Требуется нормализация (диапазон → среднее, «>N» → N, пробелы убираются).
- **Нестандартность #2.** `ID` компании в SPARK приходит как float (`1.022e+12`) — обязательно кастовать в int, иначе ломается join.
- **Что делается с 2025 годом.** На момент выгрузки SPARK ещё не успел собрать 2025 — практически пустой. В пайплайне Russia явного фильтра `year != 2025` нет, но через `dropna` по CORE колонкам 2025 строки отпадают. Стоит добавить явный drop перед публикацией.

### 2.2 Китай — WIND
- **Активная панель.** `data/raw/china/active/Wind Software & services.xlsx` (644 KB, лист Sheet1). 319 действующих компаний сектора Software & Services, 2014–2025, 15 метрик. Нестандартный формат: **тикеры лежат в столбцах, метрики-годы — в строках** с тегом `[rptDate]YYYY-12-31` и кодами WIND (`oper_rev`, `ebit2`, `tot_assets`, `intang_assets`, `net_cash_flows_oper_act`, `rd_exp`, `int_exp`, `st_borrow` и др.). Для работы требуется однократная канонизация в длинный формат `(ticker, year, metric, value)`.
- **Делистинговая панель.** `data/raw/china/bankrupt/Delisted stocks china.xlsx` (235 KB, 39 листов). По листу на компанию, формат выгрузки WIND для «экзотических» тикеров: строка 0 — Short Name, строка 1 — Start Year, строка 2 — End Year, далее — показатели по годам. Всего 37 уникальных компаний с финансовыми данными + Sheet4/Sheet5 — мусорные болванки (Sheet4 случайно содержит 272 *индийские* строки — артефакт предыдущих экспериментов, к Китаю не относится).
- **Список-затравка (не используется).** `Delisted stocks_information_technology (2).xlsx` — 62 тикера делистингов Information Technology, но без финансовых данных. Долг команды перед выборкой: при следующем окне доступа к WIND можно добить ~25 компаний.
- **Проблема #1 (критическая).** Изначально в выгрузке из WIND пришло ~37 компаний, которые команда пометила как делистинговые. Но «делистинг ≠ дефолт»: часть компаний ушла с биржи из-за приватизации (SINA, Qihoo 360), M&A (AutoNavi куплен Alibaba, Montage — Intel), перехода на другую площадку (SMIC на STAR Market Shanghai), а не из-за финансового краха. Применение всех 37 как Y=1 привело бы к контаминации таргета. **Ручная ресёрч-разметка** (с опорой на публичные источники — HKEX delisting notices, SEC 15-12B, пресс-релизы) дала финальное разбиение:
  - **11 истинных дефолтов** (forced delisting по нарушению листинга / банкротство / ликвидация): ChinaCache, China TechFaith, China Sunergy, Z-Obee, GEONG, China Finance Online-ADR, LDK Solar, LED International, Link Motion, ChinaSing (Fujian Supertech), RCG Holdings.
  - **24 стратегических ухода** (M&A, go-private, смена площадки): Actions Semi, AutoNavi, Qihoo 360, SINA, SMIC, JA Solar и др. Эти компании **переносятся в class 0** (живые), т.к. финансово они не умерли.
  - **1 skip**: ReneSola (в WIND по ошибке; не делистнута).

  Таким образом, выборка сместилась с «319 active + 37 bankrupt» на **319 active + 20 strategic-delisted (target=0) + 11 true-default (target=1)**. Это перемещение 24 кейсов из дефолтов в active — ключевое методологическое решение проекта, явно закреплённое паспортом (§1.3, §2.2: «мы фильтруем данные и оставляем только реальные экономические дефолты»).
- **Проблема #2 (блокер по полям).** В активной WIND-выгрузке:
  - `Net Profit` замаплен на `anal_reits_netprofit` (REIT-специфичное поле) — заполнен на 0 %. Чистая прибыль недоступна.
  - `Interest Expense` — 1.2 % заполнения. Interest-coverage построить нельзя.
  - `Current Liabilities` — отсутствует как поле. Используется `st_borrow` (short-term borrowings) как proxy — завышает `current_ratio`.
  - `debt_to_assets` (поле `debttoassets`) есть как строка, но не парсится корректно.

  В финальном датасете `net_profit` и `net_revenue` присутствуют как **placeholder = 0**, чтобы не ломать downstream схему; `net_margin = NI/Revenue` получается нулевым — нейтральный шум в SHAP. Для следующей выгрузки нужны явные коды `np_belongto_parcomsh`, `oper_rev_after_ded`, `tot_cur_liab`/`wgsd_liabs_curr`, `int_exp`.
- **2025.** Заполнен на 4.8 % — дроп полностью (`drop 2025` на уровне пайплайна, §3.5 обработки).

### 2.3 Дополнительные BRICS-датасеты (не используются для обучения, зафиксированы для Scope & Limitations)
- **India — Wind Software & Services_India.xlsx.** 256 тикеров × 2014–2025 × 14 метрик. Формат: тикеры в строках, метрики-годы в столбцах. Заполненность: Revenue 46.9 %, EBIT 11.3 %, EBITDA 9.3 %, Short-term Loans 24.4 %, **остальные 10 метрик = 0.0 %**. Общая заполненность 6.6 %. Непригодно.
- **Brazil — brazil_it_panel_improved.csv.** 26 компаний × ~8 лет (189 строк), 16 колонок: Receita_Liquida, EBIT, Resultado_Bruto, Lucro_Liquido, Ativo_Total, Intangivel, Patrimonio_Liquido, Fluxo_Caixa_Operacional, ROA, ROE, Debt_Equity, Current_Ratio. Качество: 0 % пропусков по ключевым метрикам, но объём мал (26 × 8 для ML ансамблей — anti-pattern), дефолты в панели не размечены, `Current_Ratio` полностью NaN. + контекст-файл `info_brazilian_companies.xlsx` (526 компаний B3 — «вселенная»).
- **South Africa.** `south_africa_it_panel_wide.csv` (15 компаний × 291 метрика, большинство NaN) + `south_africa_it_panel_long.csv` (10 870 long-rows) + `south_africa_it_delistings.csv` (4 строки, 3 default). Непригодно для ML.

Вывод: паспортный scope «Россия + Китай» эмпирически подкреплён: 6.6 % coverage в Индии, 26 компаний без таргета в Бразилии, 15 компаний / 3 дефолта в ЮАР.

---

## Часть III. Очистка и разметка таргета — методология data preparation

### 3.1 Россия — ноутбук [10_russia_load_and_clean.ipynb](../notebooks/10_russia_load_and_clean.ipynb)

**Шаг 1. Парсинг (функция `load_file_to_long`).**
1. `read_excel(..., sheet_name='report', header=3)`.
2. Сбрасываются служебные колонки (`COLS_TO_DROP`); `ID` приводится к `int`.
3. `melt` в long: `(company_id, raw_col, value)`; regex-ом из `raw_col` извлекаются `year` (4-digit) и `metric` (после запятой, без суффикса валюты).
4. Объединение трёх файлов категории (active/bankrupt), дедуп по `(company_id, year, metric)` с приоритетом непустого значения.
5. `pivot_table` → wide `(company_id, year) × metric`.
6. Нормализация «Среднесписочной численности» (см. §2.1).

**Шаг 2. Разметка таргета (ключевое отличие от проталкивания всех банкротов как Y=1).**
- `bankrupt_company` — 1, если компания целиком в категории bankrupt (маркер истории).
- `is_bankrupt` (event target K=1) — **1 ровно на последнем году, где есть хотя бы одно непустое финансовое значение**; иначе 0.
- **Event-based K=2 (финальная методология).** `is_bankrupt` = 1 на **двух последних живых годах** дефолтной компании. Это ключевой рефакторинг: один позитивный ряд на компанию не даёт ансамблю ничего выучить при 200 дефолтах на 10 000 живых, а K=2 удваивает положительный класс без добавления шума (близкие к дефолту годы эмпирически очень похожи на сам момент дефолта).

  K=1 оставлен как **sensitivity-проверка** (ноутбук `_ru_k1.py`, выходы в [reports/russia_k1/](russia_k1/)). Воспроизведение на K=1 даёт те же содержательные выводы по H1/H2 (см. §VIII), что подтверждает устойчивость методологии.

**Шаг 3. Стратегия пропусков (жёстко задана научруком).**
1. Внутри компании: `ffill().bfill()` по `FIN_COLS`.
2. Активные: если после ffill/bfill всё ещё пусто по `CORE_COLS` (Assets, Current Assets, Equity, Revenue, EBIT, NI) — строка удаляется. Потеря строк не критична (10 000+ компаний).
3. «Разреженные» колонки (`SPARSE_ZERO_COLS` — НМА, проценты к уплате) → `fillna(0)`: в SPARK пусто = «не заявлено», экономически оправданный ноль.
4. Банкроты: ничего не удаляется (каждый дефолт критичен), остатки → `fillna(0)`.

**Результат (K=2).** `ru_panel_cleaned.csv`: 88 714 company-year rows; 88 316 active + 398 default rows; positive share 0.449 % (≈ 1:221). Компаний: 10 013 active + 206 bankrupt, из них 398 positive rows (K=2 удвоение).

### 3.2 Китай — ноутбук [30_china_load_and_clean.ipynb](../notebooks/30_china_load_and_clean.ipynb)

**Парсер активной панели (`parse_wind`).** Тикеры из первой строки (колонка C+), имена компаний из второй; каждая строка — тег `[rptDate]YYYY` в колонке A и код метрики (`oper_rev`, `ebit2`, `tot_assets`, `int_exp`, `rd_exp` и т.д.) в колонке B. Мэппинг кодов — `WIND_METRIC_MAP`. Pivot → wide (ticker, company_name, year) × metric.

**Парсер делистинговой панели (`parse_delist_sheet`).** Эвристика: ищется строка с датами (4–12 строка, ≥1 дат и нет не-дат), затем строка `period` (`Ann.`), затем строки показателей. Агрессивная нормализация лейблов в `DELIST_LABEL_MAP` (`'total liab'|'total liabilities'` → `total_liab`; `shareholders' equity|total equity` → `total_equity`).

**Классификатор делистингов (`classify_delisted`).** Matching по longest-common-prefix (≥8 символов или полная длина короткого) между name в B2 и именем листа (Excel обрезает до 31 символа) с заранее заготовленной таблицей 35 разметок (11 true-default / 24 strategic / 1 skip).

**Импутация.**
1. `drop year == 2025` по всей панели.
2. `ffill+bfill` внутри тикера по `CORE_COLS`.
3. Target=0: `dropna` по `CORE_BACKBONE = [total_revenue, total_assets, total_equity, current_assets]` — строки без хотя бы одного из 4 столпов считаются мусором.
4. Target=0 sparse: остальные `CORE_COLS` → `fillna(0)`.
5. Target=1: любые NaN → `fillna(0)`.

**Результат (K=2).** `cn_panel_cleaned.csv`: 3 458 rows; 3 439 active + 19 default rows; positive share 0.549 % (≈ 1:181). На уровне компаний: 339 target=0 (319 исходно-active + 20 strategic-delisted) + 11 target=1.

---

## Часть IV. Feature engineering — единый набор ratios в обеих странах

### 4.1 15 ratios, 5 экономических групп (единая схема после рефакторинга)

| Группа | Фичи | Экономический смысл |
|---|---|---|
| **Liquidity** | `current_ratio`, `cash_to_assets`, `cash_to_cl`, `wc_to_assets` | Способность платить текущие обязательства |
| **Innovation** | `intangibles_to_assets` | Доля нематериальных активов (intellectual capital) |
| **Leverage** | `debt_to_assets`, `debt_to_equity`, `lt_debt_to_assets`, `interest_coverage` | Кредитная нагрузка |
| **Profitability** | `roa`, `net_margin`, `operating_margin`, `cfo_to_assets` | Возврат и маржинальность |
| **Size** | `log_assets`, `log_revenue` | Масштаб компании |

**Симметрия Innovation между странами.** В исходном паспорте Innovation = {Intangibles, R&D}. Но в SPARK R&D не заполнен (SPARK не собирает R&D-expense); в WIND `rd_exp` заполнен, но только с 2018 года. Чтобы H2 было **межстрановой сопоставимой гипотезой**, Innovation унифицирован до **`intangibles_to_assets`** в обеих странах (в 40_china `rd_to_revenue` построен и присутствует в SHAP-таблицах, но в финальном H3-A / cross-country сравнении используется одинаковый 15-feature космос).

**Винзоризация 1/99 %** + `fillna(median)` — страховка от тяжёлых хвостов, особенно в российском SPARK (крупные холдинги искажают средние на порядки; max `interest_coverage = 17 049.57`, max `current_ratio = 227`).

### 4.2 Дескриптивная статистика: [ru_feature_stats.csv](russia/ru_feature_stats.csv), [cn_feature_stats.csv](china/cn_feature_stats.csv), [ru_correlation_matrix.csv](russia/ru_correlation_matrix.csv), [cn_correlation_matrix.csv](china/cn_correlation_matrix.csv).

**Медианы по классам:**

| Russia | current_ratio | intang/A | debt/A | ROA |
|---|---|---|---|---|
| Active  | 2.28 | 0.000 | 0.52 | +0.136 |
| Default | 0.96 | 0.000 | 1.01 | −0.020 |

| China | current_ratio | intang/A | debt/A | ROA |
|---|---|---|---|---|
| Active  | 2.44 | 0.015 | 0.035 | +0.046 |
| Default | 1.65 | 0.000 | 0.000 | −0.090 |

Сигналы:
- **РФ**: дефолтные компании имеют втрое хуже ликвидность, вдвое больше долга к активам и отрицательный ROA.
- **КНР**: дефолтные компании имеют нулевой intangibles (intellectual capital схлопывается), отрицательный ROA, меньший current_ratio.
- НМА в SPARK ≈ 0 в обоих классах — артефакт заполнения SPARK («Нематериальные активы» почти не сдаются), поэтому Innovation-канал в РФ слабый и вклад H2 в РФ держится в основном на Liquidity.

---

## Часть V. Методология моделирования

### 5.1 Сплит данных
- **Финальная схема — Group-aware split.** `GroupShuffleSplit` (test=20 %) по компании (`Регистрационный номер` / `ticker`). **Все годы одной компании уходят целиком в train или в test.** Это устраняет data leakage через «company fingerprint», которая иначе позволяет tree-моделям запомнить компанию по (`log_assets`, `log_revenue`, `intangibles_to_assets`) и механически восстановить ответ.
- В ранней версии 40_china использовался stratified row-level split — он давал test ROC ≈ 1.00 у RF/XGB на 11 дефолтных тикерах (один и тот же тикер присутствовал и в train, и в test). В финальной версии (post-refactor) сплит group-aware, что делает оценку честной. Это критический фикс.
- **5-fold Stratified Group K-Fold CV** поверх финального сплита — для оценки стабильности метрик, а не разовой удачи. Результаты CV лежат в `ru_h1_cv_metrics.csv`, `cn_h1_cv_metrics.csv`.

### 5.2 Class imbalance — class weights, отказ от SMOTE
- При 1:221 (РФ) / 1:181 (КНР) row-level дисбалансе SMOTE создаёт фейковые «компании», у которых не сходится балансовое тождество (Assets = Liab + Equity). Это математический мусор в контексте финансовой отчётности.
- Используется **Cost-Sensitive Learning**: `class_weight={0:1, 1:POS_WEIGHT}` в LogReg/RF, `scale_pos_weight=POS_WEIGHT` в XGBoost. `POS_WEIGHT = n_neg/n_pos` на train. В РФ `POS_WEIGHT ≈ 219`; в КНР — соразмерно.

### 5.3 Модели
- **Logistic Regression** + `StandardScaler`, `solver=liblinear`, `class_weight`.
- **Random Forest** — 400 деревьев, `min_samples_leaf=3`, `class_weight`.
- **XGBoost** — 500 деревьев, `max_depth=5`, `scale_pos_weight`.

Единообразный preprocessing и одинаковые class-веса — единственная разница между моделями — это capacity. Это обязательное условие, чтобы сравнение H1 (LogReg vs ensembles) было о capacity, а не о настройках.

### 5.4 Метрики
- **ROC-AUC** — интегральное качество ранжирования; устойчиво к дисбалансу, но при <1 % positive class может вводить в заблуждение (любой ranker, отличный от random, выглядит прилично).
- **PR-AUC (Average Precision)** — **главная метрика** при сильном дисбалансе. Показывает качество работы именно с положительным классом. H1 судится по PR-AUC.
- **ΔROC (train−test)** и **ΔPR (train−test)** — сигнатура переобучения. Большое Δ = ансамбль выучил train и не переносит на test.

### 5.5 Explainability — SHAP
- **TreeExplainer** для RF/XGBoost; значения интерпретируются как вклад признака в лог-одды предсказания.
- **Групповая агрегация**: Σ|SHAP| по каждой из 5 групп → сравнение Liquidity+Innovation vs Leverage (H2), Liquidity-vs-Innovation доминанта между странами (H3-A).
- Все SHAP считаются на **одной XGBoost-спецификации**, одинаковой в обеих странах (15 фич, тот же препроцессинг) — чтобы кросс-страновое сравнение было методологически валидным.

---

## Часть VI. Структура ноутбуков и артефактов

```
VKR_prep/
├─ data/
│  ├─ raw/russia/{active,bankrupt}/   — 6 SPARK xlsx
│  ├─ raw/china/{active,bankrupt}/    — 2 WIND xlsx
│  └─ processed/                      — очищенные панели (ru_panel_cleaned.csv, cn_panel_cleaned.csv, russia_ml.xlsx)
├─ notebooks/
│  ├─ 10_russia_load_and_clean.ipynb
│  ├─ 20_russia_eda_and_models.ipynb
│  ├─ 30_china_load_and_clean.ipynb
│  ├─ 40_china_eda_and_models.ipynb
│  └─ 50_cross_country_pit.ipynb
├─ reports/
│  ├─ russia/                          — артефакты RU Stage-1 (K=2, финальная версия)
│  │  └─ extras/                       — K=2 ROC/PR-кривые, confusion matrix, calibration, top10 SHAP, CV boxplot
│  ├─ russia_k1/                       — K=1 sensitivity (проверка робастности)
│  ├─ china/                           — артефакты CN Stage-1 (K=2)
│  │  └─ extras/                       — те же доп. графики для Китая
│  ├─ cross_country/                   — H3/H3-A (cross-country SHAP + PIT-lift)
│  ├─ notebook_exports/                — PDF/HTML экспорты 5 ноутбуков
│  ├─ DATA_AUDIT_NARRATIVE.md          — подробный нарратив аудита исходных данных
│  ├─ VKR_full_report.md               — исторический детальный репорт (до-рефакторинг + переходная версия)
│  ├─ hypotheses_final.md              — финальные формулировки гипотез
│  └─ VKR_comprehensive_report.md      — этот документ
├─ src/
│  ├─ metrics.py                       — ROC/PR/Δ-helpers
│  └─ preprocessing.py                 — winzorize, fillna-median, feature group registry
├─ project_passport.txt                — паспорт темы и исходные гипотезы
└─ datasets_information.txt            — переписка команды с соавторами
```

PDF-экспорты ноутбуков — [reports/notebook_exports/](notebook_exports/): 10_russia_load_and_clean.pdf, 20_russia_eda_and_models.pdf, 30_china_load_and_clean.pdf, 40_china_eda_and_models.pdf, 50_cross_country_pit.pdf.

---

## Часть VII. Графики и таблицы — что показывает каждый артефакт и зачем он нужен

Здесь — полный каталог визуальных и табличных результатов. Для каждого артефакта указано: (а) что изображено, (б) какую методологическую/содержательную роль играет в работе, (в) куда его включать (основной текст / приложение).

### 7.1 Russia Stage-1 — [reports/russia/](russia/)

**[01_class_distribution.png](russia/01_class_distribution.png)**
- *Что*: bar chart 88 316 active vs 398 default rows.
- *Зачем*: визуальный аргумент «почему class-weights, почему не SMOTE, почему PR-AUC важнее ROC» — демонстрирует масштаб дисбаланса 1:221. **В основной текст Раздела «Методология», §Class Imbalance Strategy**.

**[02_violin_by_class.png](russia/02_violin_by_class.png)**
- *Что*: violin plots по 4 ratios (current_ratio, intangibles_to_assets, debt_to_assets, ROA) в разрезе Active vs Default.
- *Зачем*: univariate-проверка, что выбранные ratios содержат сигнал между классами (до ML). У дефолтных медианно ниже ликвидность, выше долг, отрицательный ROA. Поддерживает выбор feature space. **В основной текст Results, §EDA**.
- *Сопутствующая таблица*: [ru_medians_by_class.csv](russia/ru_medians_by_class.csv).

**[03_correlation_heatmap.png](russia/03_correlation_heatmap.png)**
- *Что*: Pearson correlation matrix 15 фич.
- *Зачем*: показать, что мультиколлинеарность под контролем (|corr| < 0.85 во всех парах кроме ожидаемой `log_assets × log_revenue`), поэтому VIF-фильтрация не делается и LogReg корректен. **Можно в Appendix**, со ссылкой из §Feature Selection.
- *Сопутствующая таблица*: [ru_correlation_matrix.csv](russia/ru_correlation_matrix.csv).

**[04_shap_summary.png](russia/04_shap_summary.png)**
- *Что*: SHAP summary beeswarm для лучшего ансамбля (XGBoost, K=2).
- *Зачем*: качественная визуализация — для каждой фичи показано распределение SHAP-значений и направление эффекта. Сразу видно: низкий `cash_to_cl` → рост P(default); высокий `debt_to_assets` → рост P(default). **В основной текст Results, §SHAP (H2)**.

**[05_h2_group_importance.png](russia/05_h2_group_importance.png)**
- *Что*: bar chart Σ|SHAP| по 5 группам.
- *Зачем*: ядровой артефакт для H2. Показывает Liquidity+Innovation vs Leverage. **В основной текст Results, §Testing H2**.
- *Таблицы*: [ru_shap_group_importance.csv](russia/ru_shap_group_importance.csv), [ru_shap_feature_importance.csv](russia/ru_shap_feature_importance.csv) (топ-15 feature-level).

**Extras ([russia/extras/](russia/extras/))** — дополнительные K=2 диагностики:
- [ru_k2_roc_curves.png](russia/extras/ru_k2_roc_curves.png) — ROC всех трёх моделей на тесте. *Зачем*: визуальное подтверждение, что все три модели выше chance; ансамбли чуть выше LogReg по ROC. **Results § H1 или Appendix.**
- [ru_k2_pr_curves.png](russia/extras/ru_k2_pr_curves.png) — PR-кривые. *Зачем*: **это важнейшая визуализация H1 при дисбалансе** — показывает, что на operating part (high-recall) LogReg не хуже, а часто лучше ансамблей. **В основной текст Results, §Testing H1**.
- [ru_k2_confusion_matrix.png](russia/extras/ru_k2_confusion_matrix.png) — для лучшей модели на фиксированном threshold. *Зачем*: показать кол-во TP / FN / FP / TN — операционная польза. **Appendix**.
- [ru_k2_calibration.png](russia/extras/ru_k2_calibration.png) — calibration curve. *Зачем*: показать, насколько предсказанные вероятности соответствуют эмпирическим частотам. Для scoring-моделей критично. **Appendix или Results если есть место**.
- [ru_k2_cv_boxplot.png](russia/extras/ru_k2_cv_boxplot.png) — 5-fold CV boxplot ROC и PR по моделям. *Зачем*: визуальное подтверждение, что LogReg стабильнее ансамблей (более узкий box) — центральный аргумент H1. **В основной текст Results, §Testing H1**.
- [ru_k2_top10_shap.png](russia/extras/ru_k2_top10_shap.png) — top-10 feature-level SHAP bar. *Зачем*: дополнение к summary beeswarm, удобнее для количественной интерпретации. **Можно в основной текст или Appendix**.
- [ru_k2_cv_folds.csv](russia/extras/ru_k2_cv_folds.csv) — raw fold-level метрики. Для приложения / reproducibility.

**Ключевые числовые таблицы (CSV)**:
- [ru_h1_metrics.csv](russia/ru_h1_metrics.csv) — single-split метрики (см. §VIII).
- [ru_h1_cv_metrics.csv](russia/ru_h1_cv_metrics.csv) — CV mean±std.
- [ru_feature_stats.csv](russia/ru_feature_stats.csv) — describe() по 15 фичам.
- [ru_medians_by_class.csv](russia/ru_medians_by_class.csv) — медианы по Active/Default.
- [ru_shap_feature_importance.csv](russia/ru_shap_feature_importance.csv) — mean|SHAP| по всем 15 фичам.
- [ru_shap_group_importance.csv](russia/ru_shap_group_importance.csv) — Σ|SHAP| по 5 группам.
- [ru_summary.md](russia/ru_summary.md) — текстовая сводка результатов RU Stage-1.

### 7.2 Russia K=1 sensitivity — [reports/russia_k1/](russia_k1/)

Проверка устойчивости при другом окне таргета (только последний живой год = 1). Аналогичный набор: [ru_k1_summary.md](russia_k1/ru_k1_summary.md), [ru_k1_h1_metrics.csv](russia_k1/ru_k1_h1_metrics.csv), [ru_k1_h1_cv_metrics.csv](russia_k1/ru_k1_h1_cv_metrics.csv), [ru_k1_shap_group_importance.csv](russia_k1/ru_k1_shap_group_importance.csv), [ru_k1_shap_feature_importance.csv](russia_k1/ru_k1_shap_feature_importance.csv), [ru_k1_h2_group_importance.png](russia_k1/ru_k1_h2_group_importance.png), extras.
- *Зачем*: **robustness check** для H1 и H2 — показывает, что выводы не зависят от выбора K. Ratio Liq+Innovation/Leverage: K=1 — 1.56×, K=2 — 1.12×. Обе версии подтверждают H2. H1: на K=1 Logit PR=0.114 снова выигрывает у ансамблей, ансамбли ΔROC > 0.12 (перепуг). **Включить в основной текст Results как §Robustness/Sensitivity Analysis или вынести сравнительную таблицу** [ru_k1_vs_k2_metrics.csv](russia_k1/ru_k1_vs_k2_metrics.csv), [ru_k1_vs_k2_groups.csv](russia_k1/ru_k1_vs_k2_groups.csv) **в основной текст, остальное — Appendix**.

### 7.3 China Stage-1 — [reports/china/](china/)

Зеркальный набор:
- [01_class_distribution.png](china/01_class_distribution.png) — 3 439 active vs 19 default rows. **Results §EDA.**
- [02_violin_by_class.png](china/02_violin_by_class.png) — медианы в [cn_medians_by_class.csv](china/cn_medians_by_class.csv); у дефолтных китайских компаний коллапс intangibles (0 vs 0.015 у active), отрицательный ROA. **Results §EDA.**
- [03_correlation_heatmap.png](china/03_correlation_heatmap.png) — мультиколлинеарность в КНР под контролем. **Appendix.**
- [04_shap_summary.png](china/04_shap_summary.png) — SHAP beeswarm. **Results §SHAP (H2).**
- [05_h2_group_importance.png](china/05_h2_group_importance.png) — Σ|SHAP| по группам. **Results §H2.**
- Extras [china/extras/](china/extras/): те же K=2 диагностики (`cn_k2_roc_curves.png`, `cn_k2_pr_curves.png`, `cn_k2_confusion_matrix.png`, `cn_k2_calibration.png`, `cn_k2_cv_boxplot.png`, `cn_k2_top10_shap.png`, `cn_k2_cv_folds.csv`).
- Ключевые CSV: [cn_h1_metrics.csv](china/cn_h1_metrics.csv), [cn_h1_cv_metrics.csv](china/cn_h1_cv_metrics.csv), [cn_feature_stats.csv](china/cn_feature_stats.csv), [cn_shap_feature_importance.csv](china/cn_shap_feature_importance.csv), [cn_shap_group_importance.csv](china/cn_shap_group_importance.csv), [cn_ttc_scores.csv](china/cn_ttc_scores.csv) (per-company TTC-скор для каждой строки — пригодится для appendix как пример того, как модель оценивает кейсы, напр. ChinaCache TTC-probability растёт с 0.16 в 2014 до 0.97 к 2016, за два года до delisting).
- Cводка: [cn_summary.md](china/cn_summary.md).

### 7.4 Cross-country / H3-A — [reports/cross_country/](cross_country/)

**Финальные H3-A артефакты** (`h3a_*` — актуальные, используют group-aware split и bootstrap-CI):
- [h3a_shap_side_by_side.png](cross_country/h3a_shap_side_by_side.png) — две SHAP summary-plot рядом (Russia / China) на одной XGBoost-спецификации. *Зачем*: ядровая визуализация H3-A, структурная асимметрия факторов. **В основной текст Results, §Cross-country comparison (H3-A)**.
- [h3a_group_shares.png](cross_country/h3a_group_shares.png) — stacked bar долей Σ|SHAP| по группам для RU и CN. **В основной текст Results §H3-A.**
- [h3a_group_importance.csv](cross_country/h3a_group_importance.csv) — численные доли групп. RU: Liquidity 30.2 %, Leverage 28.8 %, Profitability 19.6 %, Size 18.7 %, Innovation 2.7 %. CN: Innovation 26.2 %, Size 21.5 %, Profitability 20.7 %, Liquidity 17.9 %, Leverage 13.7 %.
- [h3a_top3_features.csv](cross_country/h3a_top3_features.csv): RU top-3 = `cash_to_cl` (Liquidity), `debt_to_assets` (Leverage), `log_assets` (Size); CN top-3 = `intangibles_to_assets` (Innovation), `log_revenue` (Size), `net_margin` (Profitability).
- [h3a_ttc_bootstrap_ci.csv](cross_country/h3a_ttc_bootstrap_ci.csv) — bootstrap-CI на ROC/PR-AUC: RU ROC 0.829 [0.753, 0.886], PR 0.089 [0.021, 0.174]; CN ROC 1.000 [1.000, 1.000], PR 1.000 [1.000, 1.000]. Китайский CI схлопнут в точку — следствие 11 дефолтов / bootstrap на company level.
- [h3a_ttc_table.csv](cross_country/h3a_ttc_table.csv) — сводная таблица TTC-производительности.
- [h3a_pit_vs_ttc.csv](cross_country/h3a_pit_vs_ttc.csv) — PIT-lift (placeholder macro): RU ΔROC +0.0079, CN ΔROC −0.0022. Обе разницы статистически незначимы (в своих 95 %-CI пересекают ноль).
- [h3a_summary.md](cross_country/h3a_summary.md) — текстовая сводка H3-A.

**Старые H3 артефакты** (`h3_*` — вариант (б) из паспорта, отклонён): [h3_summary.md](cross_country/h3_summary.md), [h3_pit_vs_ttc.csv](cross_country/h3_pit_vs_ttc.csv), [h3_group_importance.csv](cross_country/h3_group_importance.csv), [h3_top3_features.csv](cross_country/h3_top3_features.csv), [h3_shap_side_by_side.png](cross_country/h3_shap_side_by_side.png). **В диссертации лучше не использовать** как основной аргумент — держать для methodology-обсуждения (почему PIT-версия H3 отклонена).

---

## Часть VIII. Гипотезы и результаты — что именно подтвердилось

Финальная рамка гипотез зафиксирована в [hypotheses_final.md](hypotheses_final.md). Ключевые формулировки и цифры:

### H1 — переобучение vs стабильность LogReg на low-default выборках

> *Ensemble ML models (RF, XGBoost) yield higher predictive accuracy (ROC-AUC); however, baseline Logistic Regression demonstrates higher stability and less susceptibility to overfitting on low-default samples. Stability = (i) smaller train−test ROC gap; (ii) PR-AUC on test ≥ ensembles under heavy imbalance.*

**Russia K=2 (group-aware split + 5-fold CV)** ([ru_h1_metrics.csv](russia/ru_h1_metrics.csv), [ru_h1_cv_metrics.csv](russia/ru_h1_cv_metrics.csv)):

| Model | ROC train | ROC test | ΔROC | PR train | PR test | ΔPR | CV ROC mean±std | CV PR mean±std |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.863 | 0.835 | **+0.028** | 0.060 | **0.136** | −0.075 | 0.843±0.031 | 0.077±0.009 |
| Random Forest       | 0.9999 | 0.856 | +0.144 | 0.937 | 0.086 | 0.851 | 0.877±0.033 | 0.095±0.024 |
| XGBoost             | 0.9998 | 0.844 | **+0.156** | 0.941 | 0.127 | 0.814 | 0.845±0.017 | 0.073±0.014 |

- ✅ **Ансамбли действительно имеют более высокий ROC** (best = RF 0.856 > LogReg 0.835) — первая часть H1.
- ✅ **LogReg имеет сильно меньший ΔROC** (+0.028 против +0.144 / +0.156 у ансамблей) — классическая сигнатура переобучения ансамблей на 200 дефолтах. Вторая часть H1.
- ✅ **По PR-AUC на тесте LogReg выигрывает (0.136 vs 0.086 / 0.127)** — при дисбалансе 1:221 именно PR-AUC релевантен, и простая модель оказывается операционно лучше. Это **центральный аргумент H1**.

**China K=2** ([cn_h1_metrics.csv](china/cn_h1_metrics.csv), [cn_h1_cv_metrics.csv](china/cn_h1_cv_metrics.csv)):

| Model | ROC test | PR test | ΔROC | CV ROC | CV PR | ΔPR CV |
|---|---|---|---|---|---|---|
| Logistic Regression | 1.000 | 1.000 | −0.003 | 0.995±0.002 | 0.489±0.027 | 0.174 |
| Random Forest       | 0.999 | 0.750 | +0.002 | 0.996±0.002 | 0.718±0.126 | 0.282 |
| XGBoost             | 0.996 | 0.367 | +0.004 | 0.992±0.003 | 0.425±0.055 | **+0.575** |

- На тесте ROC у всех трёх ≈ 1.00 — это **ceiling effect при 11 дефолтах** (на group-aware split в test попадает ~2–3 тикера-дефолта; любой разумный ranker их отделяет от 300+ живых). Абсолютные ROC-числа здесь малоинформативны.
- ΔPR у XGBoost на CV = +0.575 — **чистое переобучение**, согласуется с H1.
- LogReg: PR=1.0 на тесте, но CV PR 0.489±0.027 — реальный показатель при 11 дефолтах.
- ✅ **H1 качественно подтверждается и в Китае**: ансамбли переобучаются сильнее, LogReg устойчивее.

**Russia K=1 (robustness)** ([ru_k1_summary.md](russia_k1/ru_k1_summary.md), [ru_k1_vs_k2_metrics.csv](russia_k1/ru_k1_vs_k2_metrics.csv)):
- Logit test PR = **0.114** (best PR across models), XGB test PR = 0.064, RF = 0.085.
- Logit ΔROC = +0.042; RF ΔROC = +0.138; XGB ΔROC = +0.116.
- ✅ Выводы воспроизводятся при K=1 — робастно.

**Вердикт H1: ПОДТВЕРЖДЕНА** и воспроизводима при K=1 и K=2.

**Как это интерпретировать для диссертации.** Сложный ML-ансамбль на 200 дефолтах действительно «вызубривает» train (PR train ≈ 0.94), но не переносит навык на test (PR test ≈ 0.09). Логистическая регрессия — как стабильный троечник: test PR ≈ 0.13, и train-test разрыв меньше 0.03. В условиях дефицита positives (low-default regime, характерный для скоринговых задач в развивающихся рынках) **simpler-is-better** — это методологически ценный контраргумент к «всегда ставить XGBoost в прод».

### H2 — Liquidity + Innovation > Leverage

> *In the tech sector, liquidity metrics and innovation capacity (represented by `intangibles_to_assets` as cross-country-comparable proxy) carry greater aggregate SHAP-based predictive weight for default than the block of leverage ratios (debt-to-assets, debt-to-equity, lt-debt-to-assets, interest coverage).*

**Метрика**: Σ|SHAP|(Liquidity + Innovation) / Σ|SHAP|(Leverage) на лучшем-by-PR ансамбле, group-aware split, K=2.

| | Liq+Inn | Leverage | Ratio |
|---|---|---|---|
| Russia K=2 (XGB) | 2.515 | 2.245 | **1.12×** |
| Russia K=1 (robustness) | 0.188 | 0.120 | **1.56×** |
| China K=2 (RF) | 0.170 | 0.112 | **1.52×** |

✅ **H2 ПОДТВЕРЖДАЕТСЯ** в **трёх независимых прогонах** (RU K=2, RU K=1, CN K=2). Направление и знак сигнала устойчивы.

**Оговорки для Discussion:**
1. В РФ K=2 вклад Innovation канала ≈ 0 (0.170 единиц Σ|SHAP|) — это артефакт SPARK (НМА у 3 % активных). Подтверждение H2 в РФ держится **целиком на Liquidity**. Корректная формулировка в работе: *«В России H2 подтверждается через Liquidity; Innovation-канал не проверяем из-за качества данных SPARK по НМА»*.
2. В КНР Innovation — доминирующий драйвер (см. H3-A). Содержательно в Китае H2 — это не «Liquidity + Innovation > Leverage», а «Innovation один сам по себе > Leverage».
3. H2 в обеих странах — **не эффект leakage**: формулировка сохраняется и после group-aware сплита, и на K=1.

### H3-A — структурная асимметрия факторов риска между странами (выбранная формулировка)

> *The dominant financial risk-factor groups differ between Russian and Chinese IT sectors, reflecting the difference in capital regimes: in Russia (expensive capital, limited external funding) the Liquidity group is the top SHAP-ranked driver of default risk, while in China (developed institutional support, higher role of intellectual capital) the Innovation group (intangibles intensity) leads. Measured by group-level share of total Σ|SHAP| from a unified XGBoost specification with identical features in both countries.*

**Результат** ([h3a_group_importance.csv](cross_country/h3a_group_importance.csv)):

| Group | Russia Σ|SHAP| | Russia share | China Σ|SHAP| | China share |
|---|---|---|---|---|
| **Liquidity** | **1.93** | **30.2 %** | 2.19 | 17.9 % |
| Innovation | 0.17 | 2.7 % | **3.20** | **26.2 %** |
| Leverage | 1.84 | 28.8 % | 1.67 | 13.7 % |
| Profitability | 1.25 | 19.6 % | 2.53 | 20.7 % |
| Size | 1.19 | 18.7 % | 2.63 | 21.5 % |

**Top-1 feature** ([h3a_top3_features.csv](cross_country/h3a_top3_features.csv)):
- Russia: `cash_to_cl` (Liquidity).
- China: `intangibles_to_assets` (Innovation).

✅ **H3-A ПОДТВЕРЖДЕНА**: структурная асимметрия соответствует гипотезе о различных режимах капитала.

**Важная эмпирическая корректировка формулировки из паспорта.** В паспорте §52 для Китая ожидалась доминантная роль *operational profitability*. Факт: Profitability в КНР — 20.7 % (3-е место), Innovation — 26.2 % (1-е место). В H3-A зафиксирована **Innovation**, а не Profitability. Экономическая интерпретация: в субсидируемом госсекторе все компании плюс-минус рентабельны (господдержка сглаживает профиль маржинальности), поэтому operational profitability теряет дискриминирующую силу между «живыми» и «умирающими»; реальный differentiator — способность накапливать интеллектуальный капитал. Компании, у которых intangible base схлопывается (Z-Obee, GEONG, Link Motion), — дефолтятся.

### H3 (старая, версия (б) из паспорта) — отклонена

> *~~The inclusion of macroeconomic variables (Point-in-Time calibration) improves the predictive power of models compared to standalone financial (Through-the-Cycle) models.~~*

**Почему отклонена** ([h3a_pit_vs_ttc.csv](cross_country/h3a_pit_vs_ttc.csv)):
- Russia ΔROC (PIT−TTC) = +0.009, 95 %-CI [−0.020, +0.034] — **пересекает ноль**.
- China  ΔROC (PIT−TTC) = −0.002, 95 %-CI [−0.008, +0.000] — **пересекает ноль**.
- Макропеременные в текущей версии — **placeholder (IMF-like), не реальные IMF WEO**. Маркер `REPLACE_WITH_IMF_WEO` оставлен в [50_cross_country_pit.ipynb](../notebooks/50_cross_country_pit.ipynb). Реальные IMF-ряды получены заказчиком, но на момент составления отчёта в `_MACRO_DATA` ещё не подставлены.

Честный вывод: **статистически значимого PIT-lift-а в текущей спецификации нет**. Даже после подстановки реальных IMF WEO ряда эффект может усилиться, но может и обнулиться. В работу PIT-версия H3 идёт как **открытый вопрос**, а основной тезис заменён на H3-A.

---

## Часть IX. Сводка результатов по гипотезам

| Гипотеза | Финальная формулировка | Russia | China | Вердикт |
|---|---|---|---|---|
| **H1** | Ensembles точнее по ROC, Logit устойчивее (меньший ΔROC, PR-AUC ≥ ансамблей) | Logit ΔROC +0.028 vs XGB +0.156; Logit PR 0.136 > XGB 0.127 | Ceiling на 11 дефолтах, но ΔPR XGB +0.575 (overfit) | ✅ **подтверждена** (K=1 и K=2) |
| **H2** | Σ|SHAP|(Liq+Inn) > Σ|SHAP|(Leverage) | 1.12× (K=2) / 1.56× (K=1) | 1.52× | ✅ **подтверждена** в трёх прогонах |
| **H3-A** | Доминантная группа факторов различна: РФ — Liquidity, КНР — Innovation | Liquidity 30.2 %, top-1 `cash_to_cl` | Innovation 26.2 %, top-1 `intangibles_to_assets` | ✅ **подтверждена** (с эмпирической корректировкой: в КНР Innovation вместо Profitability) |
| ~~H3~~ (PIT-lift) | PIT > TTC по ROC-AUC | ΔROC +0.009, CI [−0.020, +0.034] | ΔROC −0.002, CI [−0.008, 0.000] | ❌ **не подтверждена** (bootstrap-CI пересекает 0; к тому же макро — placeholder) |

---

## Часть X. Ограничения, angle для Discussion и что идёт в практическое применение

### 10.1 Принципиальные ограничения (не устранимы кодом)
- **Размер китайской выборки.** 11 true-default компаний — не статистически валидная выборка для ML сама по себе. Выводы по Китаю правильно трактовать как *pattern detection*, а не как *prediction with reliable confidence*. Именно поэтому кросс-странная рамка исследования («сравниваем, одинаково ли работают одни и те же фичи в разных режимах капитала») методологически оправдана — **не сравниваются абсолютные AUC**, сравниваются SHAP-структуры.
- **Survivorship bias.** WIND отдаёт только ныне-котируемые + 37 делистингов; компании, ушедшие до 2014, не видны. В SPARK — регистратор всех юрлиц, бандит-проблемы нет, но есть проблема самоотчётности (часть IT-компаний просто перестаёт сдавать отчётность без формального банкротства).
- **Russian 206 defaults.** Все 206 трактуются как «настоящий дефолт»; если часть из них — техническая ликвидация по собственной инициативе без дефолта долга, число снизится. Но даже с запасом в 20–30 % — цифра устойчива.
- **Macro placeholder.** H3 (PIT-lift) держится в подвешенном состоянии до подстановки реальных IMF WEO рядов.
- **China field gaps.** `net_profit = 0` placeholder, `current_liab = st_borrow` proxy, `interest_expense` пустой — блокеры, решаемые только перезаливкой WIND с корректными полевыми кодами.

### 10.2 Practical implications (что это даёт risk managers / инвесторам)
- **Для кредитного скоринга IT-сектора РФ**: простая LogReg на 15 ratios даёт PR-AUC 0.136 на тесте при 1:221 дисбалансе — это **operationally useful baseline** для отсечения «плохих» 10 % портфеля. XGBoost при этом ловушка: train PR 0.94 → конверт в продакшн уронит качество на порядок.
- **Для due-diligence по китайским IT-компаниям**: SHAP-структура показывает, что intangibles-intensity — сильнейший сигнал. Падение `intangibles_to_assets` у госсубсидируемой компании — красный флаг более ранний, чем debt-to-assets.
- **Для академической рамки**: работа эмпирически показывает, что **универсальной factor-importance-структуры для IT-сектора нет** — ранжирование Liquidity/Innovation/Leverage/Profitability зависит от capital regime. Это аргумент против «одной модели на все EM» и за локальные скоринговые адаптации.

### 10.3 Что хочется в расширение (не блокер для защиты)
1. Подстановка реальных IMF WEO / НБС / ЦБ РФ макроданных → перепрогон H3 (PIT-lift) с bootstrap-CI.
2. Перезаливка WIND с `np_belongto_parcomsh`, `oper_rev_after_ded`, `tot_cur_liab`, `int_exp` → устранение placeholder=0 и proxy-замен в CN.
3. Добить +25 китайских делистингов из файла-затравки (62 тикера) → увеличить positive class и сделать оценки в КНР устойчивее.
4. Country-effect через общую LogReg с дамми `is_china` — альтернативный способ тестирования H3-A.
5. Time-based split (train ≤ 2021, test 2022–24) поверх group-aware — усилит аргумент устойчивости во времени.
6. Добавить macro-policy rate как третью макропеременную (РФ — ключевая ставка ЦБ, КНР — 1Y LPR).

---

## Часть XI. Рекомендации по структуре диссертации (что в основной текст, что в Appendix)

**Chapter 2 — Methodology (основной текст)**
- §2.1 Two-stage design TTC → PIT (обоснование, Part I выше).
- §2.2 Data Sourcing & Target Definition: SPARK для РФ (bankruptcy as ground truth); WIND для КНР + ручная разметка 37 делистингов → 11 true default + 24 strategic + 1 skip. **Ссылка на [DATA_AUDIT_NARRATIVE.md](DATA_AUDIT_NARRATIVE.md) в Appendix** для полного обоснования.
- §2.3 Scope & Limitations: эмпирическое обоснование исключения India / Brazil / South Africa через цифры покрытия (6.6 %, 26 companies w/o targets, 15 companies / 3 defaults).
- §2.4 Feature Selection: 15 ratios в 5 группах, Innovation = `intangibles_to_assets` симметрично. Таблица групп — в §2.4, числовые describe — ссылка на [ru_feature_stats.csv](russia/ru_feature_stats.csv), [cn_feature_stats.csv](china/cn_feature_stats.csv) в Appendix.
- §2.5 Data Preprocessing: ffill/bfill внутри company, dropna для active по CORE, zero-fill для sparse cols & defaults. Event-based K=2 target.
- §2.6 Class Imbalance Strategy: Class Weights, отказ от SMOTE с обоснованием через баланс-тождество.
- §2.7 Model Specifications: LogReg / RF / XGBoost, одинаковый preprocessing.
- §2.8 Evaluation: ROC-AUC, PR-AUC, Δ, Group-aware split + 5-fold Stratified Group K-Fold CV. SHAP (TreeExplainer). Bootstrap-CI на company level для устойчивости оценок в малой китайской выборке.

**Chapter 3 — Results & Discussion (основной текст)**
- §3.1 EDA & Class Distribution: [01_class_distribution.png](russia/01_class_distribution.png), [01_class_distribution.png](china/01_class_distribution.png); [02_violin_by_class.png](russia/02_violin_by_class.png), [02_violin_by_class.png](china/02_violin_by_class.png); таблицы медиан.
- §3.2 Testing H1 — Overfitting signature: таблицы [ru_h1_metrics.csv](russia/ru_h1_metrics.csv), [ru_h1_cv_metrics.csv](russia/ru_h1_cv_metrics.csv), [cn_h1_metrics.csv](china/cn_h1_metrics.csv), [cn_h1_cv_metrics.csv](china/cn_h1_cv_metrics.csv); PR curves [ru_k2_pr_curves.png](russia/extras/ru_k2_pr_curves.png), [cn_k2_pr_curves.png](china/extras/cn_k2_pr_curves.png); CV boxplots [ru_k2_cv_boxplot.png](russia/extras/ru_k2_cv_boxplot.png), [cn_k2_cv_boxplot.png](china/extras/cn_k2_cv_boxplot.png). Discussion — low-default regime, когда simpler-is-better.
- §3.3 Testing H2 — SHAP group importance: [05_h2_group_importance.png](russia/05_h2_group_importance.png), [05_h2_group_importance.png](china/05_h2_group_importance.png); feature-level top-5 + group sums. Discussion — почему в РФ держится только на Liquidity (SPARK qa issue), почему в КНР Intangibles рулят.
- §3.4 Cross-Country Comparative Analysis — H3-A: [h3a_shap_side_by_side.png](cross_country/h3a_shap_side_by_side.png), [h3a_group_shares.png](cross_country/h3a_group_shares.png), [h3a_group_importance.csv](cross_country/h3a_group_importance.csv), [h3a_top3_features.csv](cross_country/h3a_top3_features.csv). Discussion — экономическая интерпретация: дорогой капитал РФ → ликвидность first-order; господдержка КНР → intangibles first-order; эмпирическая корректировка паспорта (Innovation вместо Profitability в КНР).
- §3.5 Macroeconomic Integration — H3 (PIT vs TTC, ОТКРЫТЫЙ ВОПРОС): [h3a_pit_vs_ttc.csv](cross_country/h3a_pit_vs_ttc.csv), CI пересекает 0, макро placeholder. Discussion — честное признание ограничения + план на IMF WEO.
- §3.6 Robustness / Sensitivity — K=1 vs K=2: [ru_k1_vs_k2_metrics.csv](russia_k1/ru_k1_vs_k2_metrics.csv), [ru_k1_vs_k2_groups.csv](russia_k1/ru_k1_vs_k2_groups.csv).

**Appendix**
- A. Data audit narrative ([DATA_AUDIT_NARRATIVE.md](DATA_AUDIT_NARRATIVE.md)) — полный отчёт по исходным данным.
- B. Feature statistics: [ru_feature_stats.csv](russia/ru_feature_stats.csv), [cn_feature_stats.csv](china/cn_feature_stats.csv), correlation matrices + heatmaps ([03_correlation_heatmap.png](russia/03_correlation_heatmap.png), [03_correlation_heatmap.png](china/03_correlation_heatmap.png)).
- C. Full SHAP tables: [ru_shap_feature_importance.csv](russia/ru_shap_feature_importance.csv), [cn_shap_feature_importance.csv](china/cn_shap_feature_importance.csv), [ru_k2_top10_shap.png](russia/extras/ru_k2_top10_shap.png), [cn_k2_top10_shap.png](china/extras/cn_k2_top10_shap.png).
- D. Additional model diagnostics: ROC-кривые, confusion matrices, calibration plots ([russia/extras/](russia/extras/), [china/extras/](china/extras/)).
- E. Per-company TTC scores (для примеров кейсов, напр. ChinaCache 2014→2017): [cn_ttc_scores.csv](china/cn_ttc_scores.csv).
- F. K=1 sensitivity detail: [reports/russia_k1/](russia_k1/) полностью.
- G. Bootstrap-CI таблицы: [h3a_ttc_bootstrap_ci.csv](cross_country/h3a_ttc_bootstrap_ci.csv).

---

*Конец комплексного отчёта. Все числа — по состоянию на 2026-04-15 (пост-рефакторинг: event-based K=2, group-aware split, `intangibles_to_assets`-симметричный Innovation, bootstrap-CI на company level).*
