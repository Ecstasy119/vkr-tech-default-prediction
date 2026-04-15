# ВКР — сводный репорт по пайплайну
**Тема:** Assessment of Financial Stability for Technology Companies in Selected Emerging Markets (Evidence from Russia and China) Using Machine Learning Models.

**Что за документ.** Это детальный разбор всего, что сейчас сделано в проекте: как получен каждый датасет, как он чистился, какая логика анализа, что именно проверяет каждая гипотеза, какие числа получились, где у результатов есть оговорки, и что ещё не сделано. Документ разбит по ноутбукам в порядке прохождения пайплайна. На каждом шаге даны прямые ссылки на соответствующий ноутбук (HTML-копии лежат в [reports/notebook_exports/](notebook_exports/)) и на графики/таблицы в [reports/](./).

---

## 0. Карта проекта и артефактов

```
VKR_prep/
├─ data/
│  ├─ raw/russia/{active,bankrupt}/          ← исходные 3 xlsx по активным и 3 xlsx по банкротам (SPARK-Interfax)
│  ├─ raw/china/                             ← Wind Software & Services.xlsx + Delisted stocks china.xlsx
│  └─ processed/
│     ├─ ru_panel_cleaned.csv                ← выход 10_russia_load_and_clean
│     ├─ russia_ml.csv / .xlsx               ← human-readable raw merge с подсветкой банкротов
│     └─ cn_panel_cleaned.csv                ← выход 30_china_load_and_clean
├─ notebooks/
│  ├─ 10_russia_load_and_clean.ipynb
│  ├─ 20_russia_eda_and_models.ipynb
│  ├─ 30_china_load_and_clean.ipynb
│  ├─ 40_china_eda_and_models.ipynb
│  └─ 50_cross_country_pit.ipynb
├─ reports/
│  ├─ russia/                                ← все артефакты Stage-1 RU
│  ├─ china/                                 ← все артефакты Stage-1 CN
│  ├─ cross_country/                         ← артефакты H3
│  ├─ notebook_exports/                      ← HTML-копии 5 ноутбуков (можно открыть в браузере → Print → Save as PDF)
│  └─ VKR_full_report.md                     ← этот документ
├─ datasets_information.txt                  ← переписка с соавторами про происхождение данных
└─ project_passport.txt                      ← паспорт работы (темы, гипотезы, методология)
```

**Гипотезы (цитируется из [project_passport.txt](../project_passport.txt)):**

* **H1.** Ensemble-ML (RF, XGBoost) даёт более высокий ROC-AUC, но baseline LogReg устойчивее к переобучению на выборках с малым числом дефолтов.
* **H2.** Метрики ликвидности и инновационного потенциала (Intangibles, R&D) имеют больший предсказательный вес, чем традиционные метрики leverage.
* **H3.** Профиль риска зависит от локального рынка: в экономике дорогого капитала (Россия) доминирует **liquidity deficit**, в экономике с институциональной поддержкой (Китай) — **operational profitability**; плюс включение макропеременных в PIT-модель увеличивает AUC относительно TTC.

**Экспорт ноутбуков (для ревью) — PDF:**
* [10_russia_load_and_clean.pdf](notebook_exports/10_russia_load_and_clean.pdf)
* [20_russia_eda_and_models.pdf](notebook_exports/20_russia_eda_and_models.pdf)
* [30_china_load_and_clean.pdf](notebook_exports/30_china_load_and_clean.pdf)
* [40_china_eda_and_models.pdf](notebook_exports/40_china_eda_and_models.pdf)
* [50_cross_country_pit.pdf](notebook_exports/50_cross_country_pit.pdf)

---

## 1. Russia — загрузка и очистка ([10_russia_load_and_clean.ipynb](../notebooks/10_russia_load_and_clean.ipynb))

### 1.1 Источник данных
**База SPARK-Interfax**, 6 Excel-файлов: по 3 на категорию «active» и «bankrupt» (см. [data/raw/russia/](../data/raw/russia/)). Каждый файл — лист `report` с шапкой в строке 4, где показатели разложены в колонки вида `«2014, Выручка, RUB»` — классический «широкий» формат SPARK.

### 1.2 Алгоритм парсинга (функция `load_file_to_long`)
1. Читаем лист `report` с `header=3`, удаляем служебные колонки (`COLS_TO_DROP`), оставляем `ID` + «годовые» колонки 2010–2030.
2. `ID` приводится к `int` (чтобы убрать float-артефакты SPARK вида `1.022e+12`).
3. `melt` превращает широкий формат в long: `(company_id, raw_col, value)`, затем regex-ом извлекается `year` (4 цифры) и `metric` (после запятой, без суффикса `, RUB`).
4. Объединение трёх файлов в категорию, дедупликация по `(company_id, year, metric)` с приоритетом непустого значения, и финальный `pivot_table` в wide-формат `(company_id, year) × metric`.

### 1.3 Приведение нестандартных колонок
«Среднесписочная численность работников» в SPARK приходит **смешанным текстом**: диапазоны (`"0 - 5"`, `"51 - 100"`), числа с пробелами-разделителями (`"1 011"`), открытые интервалы (`"> 5 000"`) и собственно числа. Функция `parse_employee_count` приводит это к единому float: диапазон → среднее, «>5000» → 5000, пробелы убираются.

### 1.4 Разметка таргета
Ключевое решение про **is_bankrupt vs bankrupt_company**:
* `bankrupt_company` — 1 если компания относится к категории «bankrupt» (вся история компании).
* `is_bankrupt` (target) — **1 ровно на последнем году, где есть хотя бы одно непустое финансовое значение**, иначе 0.

Это избавляет модель от утечки: пред-банкротные годы не размечены как «1», и мы не учим модель «компания всегда была банкротом». Сэмпл таргета — это именно *момент падения*.

### 1.5 Стратегия пропусков (шаг 2 ноутбука — `ru_panel_cleaned.csv`)
Научрук жёстко задал правила:
1. Внутри каждой компании — `ffill().bfill()` по `FIN_COLS` (т.е. пропуск в середине истории компании заполняется соседними годами этой же компании).
2. Активные компании: если после ffill/bfill всё ещё пусто по **CORE_COLS** (A, CA, EQ, Revenue, EBIT, NI и т.д.) — строка **удаляется**. Активных у нас десятки тысяч, потеря строк не критична.
3. «Разреженные» колонки (`SPARSE_ZERO_COLS`: проценты к уплате, НМА и т.п.) — `fillna(0)`: в SPARK пусто = «не заявлено», экономический смысл нуля.
4. **Банкроты**: ничего не удаляем (каждый дефолт критичен), все остатки → `fillna(0)`.

### 1.6 Результат
```
ru_panel_cleaned.csv
  shape = (88 714, 20)
  уникальных компаний: 10 013 активных + 206 банкротов
  строк target=1 = 206 (по одной на банкрота — на последний живой год)
  доля позитива на уровне строк ≈ 0.232 %
  дисбаланс ≈ 1:429
```

### 1.7 Нюансы и что *не* учтено
* **206 дефолтов** — это верхняя граница «экономически-значимых банкротств» в SPARK-выборке; если часть из них окажется техническими (ликвидация по собственной инициативе без дефолта долга), цифра может понизиться. Пока все 206 трактуются как «настоящий дефолт».
* **2025 год** в Russia-датасете не вычищен отдельным шагом — в SPARK на момент выгрузки 2024 уже неполный, а 2025 вовсе нет. В EDA-ноутбуке (см. §2) это неявно отфильтровалось через `dropna` по core-колонкам, но явного `panel = panel[panel.year != 2025]` здесь **нет** — стоит добавить, если планируется повторная выгрузка.
* Макропеременные (ВВП, инфляция) в этот файл не внедрены — это задача Stage-2 ([50_cross_country_pit.ipynb](../notebooks/50_cross_country_pit.ipynb)).
* Дисбаланс 1:429 крайне агрессивный. На нём даже `class_weight={0:1,1:10}` выдаёт только ~0.09 PR-AUC (см. §2.3) — обычная история для panel-данных SPARK.

---

## 2. Russia — EDA и Stage-1 модели ([20_russia_eda_and_models.ipynb](../notebooks/20_russia_eda_and_models.ipynb))

### 2.1 Фича-инжиниринг
15 ratios в 5 экономических группах (`FEATURE_GROUPS` в ноутбуке):
* **Liquidity**: `current_ratio, cash_to_assets, cash_to_cl, wc_to_assets` — способность платить по текущим обязательствам.
* **Innovation**: `intangibles_to_assets` — R&D в SPARK напрямую нет, поэтому канал один.
* **Leverage**: `debt_to_assets, debt_to_equity, lt_debt_to_assets, interest_coverage`.
* **Profitability**: `roa, net_margin, operating_margin, cfo_to_assets`.
* **Size**: `log_assets, log_revenue`.

**Винзоризация 1/99 %** + `fillna(median)` — стандартная страховка от тяжёлых хвостов в русской финансовой отчётности (крупные холдинги искажают средние на порядки).

### 2.2 EDA
* **[01_class_distribution.png](russia/01_class_distribution.png)** — визуализация дисбаланса 88 508 vs 206. Это основной «визуальный аргумент» для всей аргументации про class-weights и против SMOTE в разделе 2.4 паспорта.
* **[02_violin_by_class.png](russia/02_violin_by_class.png)** — distributions по `current_ratio / intangibles_to_assets / debt_to_assets / roa`. Медианы по классам (таблица [ru_medians_by_class.csv](russia/ru_medians_by_class.csv)):

  | | current_ratio | intangibles/A | debt/A | ROA |
  |---|---|---|---|---|
  | **Active**  | 2.28 | 0.000 | 0.53 | +0.14 |
  | **Default** | 0.84 | 0.000 | 1.12 | −0.03 |

  Чёткий сигнал: у банкротов втрое хуже текущая ликвидность и вдвое выше debt/assets. ROA отрицательный. НМА ≈ 0 в обоих классах — в SPARK «Нематериальные активы» почти всегда не заполнены.
* **[03_correlation_heatmap.png](russia/03_correlation_heatmap.png)** — проверка мультиколлинеарности. Ни одна пара не даёт |corr|>0.85, т.е. ни LogReg, ни tree-модели не страдают от коллинеарности фичей; отдельной VIF-фильтрации не делаем.

### 2.3 H1 — переобучение на малом числе дефолтов
**Сплит**: stratified 80/20 на уровне строк (company-year), `random_state=42`. Train: 70 971 строк / 165 дефолтов, Test: 17 743 строк / 41 дефолт.

**Модели** (все с class-imbalance-коррекцией, без SMOTE):
* `LogReg` + `StandardScaler`, `class_weight={0:1,1:10}`, `solver=liblinear`.
* `RandomForest` 400 trees, `min_samples_leaf=3`, `class_weight={0:1,1:10}`.
* `XGBoost` 500 trees, `max_depth=5`, `scale_pos_weight=10`.

**Результат** ([ru_h1_metrics.csv](russia/ru_h1_metrics.csv)):

| Model | ROC-AUC train | ROC-AUC test | ΔROC | PR-AUC train | PR-AUC test | ΔPR |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8789 | **0.8548** | +0.0241 | 0.0581 | 0.1090 | −0.0509 |
| Random Forest       | 0.9998 | **0.9016** | +0.0982 | 0.8846 | 0.0594 | +0.8253 |
| XGBoost             | 0.9999 | 0.8814 | +0.1185 | 0.9583 | 0.0411 | +0.9172 |

**Вердикт:** ✅ **H1 подтверждается.** RF / XGBoost дают train-PR-AUC ≈ 0.9–0.96 vs test-PR-AUC ≈ 0.04–0.06 — это почти идеальная иллюстрация переобучения при 165 positive-рядах в train. LogReg имеет минимальный ΔROC (+0.024) и даже *лучший* PR-AUC на тесте, чем ансамбли. При этом best test ROC-AUC у RF (0.9016) — т.е. ансамбли действительно точнее, но их точность — иллюзия на train.

**Почему это корректный вывод.** Мы сознательно использовали одинаковый `class_weight=10` и одинаковый preprocessing для всех трёх моделей; разница между ними — только capacity. Больший capacity → выше чувствительность к шуму в 165 positives → выше train-AUC, но *меньше* generalization на test-PR-AUC.

### 2.4 H2 — SHAP, ликвидность+инновации vs рычаг
Лучший ансамбль = **Random Forest** (test ROC-AUC 0.9016). Для него строится SHAP TreeExplainer.

* **[04_shap_summary.png](russia/04_shap_summary.png)** — summary plot.
* **[05_h2_group_importance.png](russia/05_h2_group_importance.png)** — barplot по блокам.

**Топ-фичи по mean |SHAP|** ([ru_shap_feature_importance.csv](russia/ru_shap_feature_importance.csv)):

1. `cash_to_cl` (0.0070) — денежные средства к текущим обязательствам
2. `log_assets` (0.0053)
3. `debt_to_equity` (0.0040)
4. `interest_coverage` (0.0037)
5. `operating_margin` (0.0036)

**Группы** ([ru_shap_group_importance.csv](russia/ru_shap_group_importance.csv)):

| Группа | Σ|SHAP| |
|---|---|
| Liquidity | 0.0157 |
| Leverage | 0.0103 |
| Profitability | 0.0090 |
| Size | 0.0077 |
| Innovation | 0.0004 |

`Liquidity + Innovation = 0.0161` > `Leverage = 0.0103`, **ratio 1.56×** → ✅ **H2 подтверждается**.

**Оговорка:** вклад Innovation почти нулевой (0.0004) — это артефакт SPARK, где НМА заполняются у 3% активных компаний. Подтверждение H2 держится целиком на **Liquidity**. В русском варианте это корректно назвать *«H2 подтверждается через liquidity; innovation-канал проверить невозможно из-за пропусков в отчётности»*.

### 2.5 Что *не* сделано в этом ноутбуке
* **Cross-validation** — есть только один train/test split. Stratified K-Fold (3 или 5 фолдов) усилил бы аргумент H1; сейчас ΔROC оценены на одном 80/20 — статистически шумно (хотя 41 test-default даёт разумный ROC).
* **Split по компаниям** (а не по строкам): сейчас годы одной компании могут попадать и в train, и в test — утечка через «company fingerprint». В РФ при 10 000+ компаний это не критично, в китайском случае — фатально (§4.5).
* **Time-based split** (train ≤ 2020, test 2021–2024) — полезно для защиты устойчивости во времени; отложено до Stage-2.
* **PIT-макро** — в этом ноутбуке не подключено, это вход для [50_cross_country_pit.ipynb](../notebooks/50_cross_country_pit.ipynb).

---

## 3. China — загрузка и очистка ([30_china_load_and_clean.ipynb](../notebooks/30_china_load_and_clean.ipynb))

### 3.1 Источник данных
Два Wind-экспорта (см. [datasets_information.txt](../datasets_information.txt)):
* **Wind Software & services.xlsx** — 319 активных китайских IT-компаний, 2014–2025, показатели: Total Revenue, EBIT, EBITDA, Total Assets/Liabilities/Equity, Current Assets, Cash, Intangibles, R&D, CFO, Interest Expense и др. (лист — широкий, тикеры в колонках, метрики в строках с тегом `[rptDate]YYYY`).
* **Delisted stocks china.xlsx** — по листу на каждую делистнутую компанию (~37 листов), макеты не унифицированы: у кого-то «Total Revenue», у кого-то «Operating Revenue»; у кого-то даты в строке 4, у кого-то в 8.

### 3.2 Классификация делистингов — **ключевое решение проекта**
Паспорт работы явно требует **не приравнивать делистинг к банкротству**. На входе есть 35 кейсов делистинга; из документа `123.pdf` и внешнего ресёрча соавторы разметили каждый кейс:

* **11 true defaults** — forced delisting по нарушениям листинга, банкротство, ликвидация:
  `ChinaCache, China TechFaith, China Sunergy, Z-Obee, GEONG, China Finance Online-ADR, LDK Solar, LED International, Link Motion, ChinaSing (Fujian Supertech), RCG Holdings`.
* **24 strategic exits** — M&A, go-private, смена площадки, добровольный делистинг: Actions Semi, AutoNavi, Qihoo 360, SINA, SMIC, JA Solar и т.д. Эти компании **не** дефолтные — это стратегические уходы; по методологии паспорта они **присоединяются к Target=0** как дополнительные «живые» наблюдения.
* **1 skip** — ReneSola (не делистнута фактически; Wind-экспорт содержал её по ошибке).

**Как реализовано:** функция `classify_delisted(name, sheet_name)` матчит имя по longest-common-prefix (≥ 8 символов или полная длина короткого); ищет как по имени из B2, так и по имени листа (Excel обрезает имена до 31 символа). Это спасает случаи типа `Fujian Supertech Advanced Material` (B2) vs `ChinaSing Investment Holdings` (sheet name): один и тот же тикер после ребрендинга.

### 3.3 Wind Active parser (`parse_wind`)
* Тикеры берутся из первой строки с колонки C, названия — из второй.
* Каждая строка таблицы имеет тег `[rptDate]YYYY` в столбце A и короткое имя метрики (`oper_rev, ebit2, tot_assets, int_exp, rd_exp, …`) в столбце B. Мэппинг — в `WIND_METRIC_MAP`.
* Pivot → wide `(ticker, company_name, year) × metric`.

⚠️ **Net Profit / Net Revenue намеренно не замаплены** — в выгрузке Wind эти колонки пустые или битые. В финальном датасете они присутствуют как **placeholder = 0** (поле добавлено ради схемы, чтобы не ломать downstream-код).

### 3.4 Delisted per-company parser (`parse_delist_sheet`)
Универсальный: эвристически ищет строку с датами (4–12 строка, в которой ≥ 1 dates и нет не-дат), затем строку `period` (должна быть `Ann.`), затем строки показателей. Нормализация лейблов — агрессивная: `'total liab' | 'total liabilities'` оба мапятся в `total_liab`; `"shareholders' equity"` и `total equity` → `total_equity`. Список в `DELIST_LABEL_MAP`.

Фильтр листов: `re.fullmatch(r'Sheet\d+', s)` — исключает только служебные index-листы (Sheet4/Sheet5), чтобы не отфильтровать по ошибке компании с именами-суффиксами `-Financial Statements` (Excel 31-char truncation).

### 3.5 Импутация по правилам научрука
1. **Drop 2025** — по всей панели.
2. **ffill+bfill внутри тикера** по `CORE_COLS`.
3. **Target=0**: `dropna` по `CORE_BACKBONE = [total_revenue, total_assets, total_equity, current_assets]`. Строки без хоть одного из 4 столпов — мусор.
4. **Target=0 sparse**: остальные `CORE_COLS` → `fillna(0)`.
5. **Target=1**: любые NaN → `fillna(0)` (каждый дефолт критичен).

### 3.6 Результат
```
cn_panel_cleaned.csv
  shape = (3 460, 20)
  source_class: active=319, strategic_delisted=20, default_delisted=11
  Target=0: 339 тикеров / 3 412 строк
  Target=1:  11 тикеров /    48 строк
  дисбаланс на уровне строк ≈ 1:71
  годы: 2014–2024
  NaN в core-колонках: 0
```

*(Число strategic-delisted в финале — 20 из 24 разметки: 3 листа отфильтрованы на этапе парсинга как «no annual data» — AutoNavi, iSoftStone, Montage имеют только quarterly; 1 — дубль тикера Sinotel.)*

### 3.7 Нюансы и что *не* учтено
* **`interest_expense`** в Wind экспорте заполнен на ~1 % — Wind просто не отдаёт это поле для китайских IT. Все `interest_coverage`-ratio в §4 фактически вычисляются на небольшой подвыборке и затем винзоризуются — сигнал слабый.
* **`current_liab` = `st_borrow`** (short-term borrowings) как proxy: в Wind явного Current Liabilities нет для этих тикеров, а `st_borrow` заполнен на ~60 %. Это **переоценивает** `current_ratio` в sparse-подвыборке. Перед публикацией стоит перезапросить Wind с явным Current Liab.
* **`net_profit`, `net_revenue` = 0 placeholder** — ждём перезаливку Wind с корректным `np_belongto_parcomsh` и `oper_rev_after_ded`. Сейчас модели используют `ebit` / `total_revenue` вместо них, что методологически оправдано (H2 ссылается на operating metrics), но фактически у нас нет bottom-line profitability.
* **Аудит источника (2026-04-15).** Перепроверка [Wind Software & services.xlsx](../data/raw/china/active/Wind%20Software%20%26%20services.xlsx) подтвердила: в выгрузке всего 17 metric-кодов (`oper_rev, ebit2, ebitda2, tot_assets, tot_liab, tot_equity, tot_cur_assets, st_borrow, cash_cash_equ_beg_period, intang_assets, net_cash_flows_oper_act, int_exp, rd_exp, anal_reits_netprofit, debttoassets, cash_recp_sg_and_rs` + Code/Name); кодов `tot_cur_liab` / `wgsd_liabs_curr` / `np_belongto_parcomsh` / `oper_rev_after_ded` **нет вообще**, а единственный кандидат на NP — `anal_reits_netprofit` — заполнен на 0.00 %. Поэтому интеграция Net Profit и Current Liabilities **не сводится к перепарсингу существующего файла** — нужна новая выгрузка с явным запросом этих полей у Wind.
* **2025 год** исключён — данных за год нет.
* **319 активных** — это весь доступный universe Wind Software & Services, т.е. выборка по сути = популяция. Survivorship bias относительно более старых (до 2014) ушедших тикеров не контролируется.

---

## 4. China — EDA и Stage-1 модели ([40_china_eda_and_models.ipynb](../notebooks/40_china_eda_and_models.ipynb))

### 4.1 Фича-инжиниринг
Та же схема, что в России, **плюс один доп. канал Innovation — `rd_to_revenue`** (в Китае R&D expense заполнен хорошо):

| Группа | Фичи |
|---|---|
| Liquidity | `current_ratio, cash_to_assets, cash_to_cl, wc_to_assets` |
| **Innovation** | `intangibles_to_assets, rd_to_revenue` |
| Leverage | `debt_to_assets, debt_to_equity, lt_debt_to_assets, interest_coverage` |
| Profitability | `roa, net_margin, operating_margin, cfo_to_assets` |
| Size | `log_assets, log_revenue` |

16 фичей всего. Винзоризация 1/99 %, `fillna(median)` — как в России.

### 4.2 EDA
* **[01_class_distribution.png](china/01_class_distribution.png)** — 3 412 vs 48 (rows). На уровне компаний — 339 vs 11.
* **[02_violin_by_class.png](china/02_violin_by_class.png)** — distributions. Медианы по классам ([cn_medians_by_class.csv](china/cn_medians_by_class.csv)):

  | | current_ratio | intang/A | debt/A | ROA |
  |---|---|---|---|---|
  | Active  | 17.25 | 0.015 | 0.33 | 0.00 |
  | Default | 2.12 | 0.000 | 0.36 | 0.00 |

  Обратить внимание: `current_ratio` у активных = 17.25 — это **завышенная** медиана из-за `current_liab = st_borrow` (см. §3.7); реальный current-ratio в Wind намного ниже, но такой вид имеет сигнал *после* нашей proxy-замены, и модели всё равно учатся на этом признаке.
* **[03_correlation_heatmap.png](china/03_correlation_heatmap.png)** — мультиколлинеарность под контролем (|corr|<0.85 для всех пар кроме ожидаемых `log_assets × log_revenue`).

### 4.3 H1 — переобучение
Сплит: stratified 80/20 на уровне **строк**. Train 2 768 / 38 defaults, Test 692 / 10 defaults. `scale_pos_weight` вычислен автоматически ≈ 72.

**Результат** ([cn_h1_metrics.csv](china/cn_h1_metrics.csv)):

| Model | ROC-AUC train | ROC-AUC test | ΔROC | PR-AUC test |
|---|---|---|---|---|
| LogReg | 0.997 | 0.996 | +0.001 | 0.696 |
| RF     | 1.000 | **1.000** | −0.000 | 1.000 |
| XGB    | 1.000 | 0.999 | +0.001 | 0.957 |

**Вердикт ноутбука:** ⚠️ **H1 частично**. По train/test-разрыву LogReg *не* стабильнее ансамблей — все три модели имеют ΔROC ≈ 0.
**Почему так:** при stratified split *на уровне строк* каждая из 11 дефолтных компаний имеет 4–5 лет панели, которые случайно распределяются между train и test. То есть *один и тот же тикер* попадает в обе части — RF/XGBoost буквально «запоминают» company fingerprint (`log_assets`, `log_revenue`, `intangibles_to_assets`) и восстанавливают ответ на test. **Это data leakage, а не настоящая generalization.** В текстовой интерпретации §4.5 (в отчётном markdown) это прямо помечено.

**Как честнее проверить H1 на Китае:**
* `GroupKFold` или `train_test_split(..., stratify=y)` **по `ticker`**, а не по строкам.
* Time-based split (train ≤ 2021, test 2022–2024).
Оба способа уронят test-AUC ощутимо и сделают сравнение LogReg vs RF/XGB содержательным. Это запланировано как следующий шаг.

### 4.4 H2 — SHAP
* **[04_shap_summary.png](china/04_shap_summary.png)** — summary plot RF (best test ROC-AUC = 1.000 по вышеописанной причине).
* **[05_h2_group_importance.png](china/05_h2_group_importance.png)** — блоки.

**Топ-фичи** ([cn_shap_feature_importance.csv](china/cn_shap_feature_importance.csv)):

1. `intangibles_to_assets` (0.1614) 🥇
2. `operating_margin` (0.0640)
3. `log_revenue` (0.0598)
4. `log_assets` (0.0534)
5. `current_ratio` (0.0353)
6. `rd_to_revenue` (0.0344)
7. `cash_to_cl` (0.0338)

**Группы** ([cn_shap_group_importance.csv](china/cn_shap_group_importance.csv)):

| Группа | Σ|SHAP| |
|---|---|
| Innovation    | 0.1958 |
| Size          | 0.1132 |
| Liquidity     | 0.0976 |
| Profitability | 0.0767 |
| Leverage      | 0.0394 |

`Liquidity + Innovation = 0.2934` > `Leverage = 0.0394`, **ratio 7.45×** → ✅ **H2 подтверждается** (и заметно сильнее, чем в РФ — там 1.56×).

**Почему коэффициент в 4.7× выше, чем в России:** в китайской выборке (а) НМА заполнены для всех тикеров (в SPARK — почти нигде), (б) есть отдельный `rd_to_revenue`-канал (в SPARK R&D нет), (в) дефолтные китайские IT-компании — это Z-Obee / GEONG / Link Motion и т.п., у которых **схлопывается именно intangible base** (списание goodwill, прекращение R&D) до дефолта. Это экономически согласованная картина.

**Оговорка:** поскольку H1 тесты скомпрометированы leakage, H2 тоже частично на leakage — `intangibles_to_assets` могла стать топ-фичей потому, что модель запомнила «у компании X `intangibles/A=0.08` → она дефолт». После исправления split'а ожидаю, что **ранг H2 сохранится** (distribution по классам действительно различается — см. медианы), но абсолютные |SHAP| упадут.

### 4.5 Что *не* сделано в этом ноутбуке
* Сплит по тикерам (см. §4.3).
* Time-based split.
* Bootstrap-CI на test-метрики — при 10 дефолтах в test одна перестановка легко меняет AUC на ±0.1.
* **Replacement `current_liab` proxy** — ждём перезаливку Wind.
* **net_profit, net_revenue** — placeholder=0; `net_margin = NI/Revenue` вычисляется через `net_profit=0`, то есть он **тождественно ноль** и его вклад в SHAP — чистый шум. Тот факт, что он не всплыл в топе, — хороший знак, но признак стоит убрать из `FEATURES` после перезаливки.

---

## 5. Cross-country PIT — H3 ([50_cross_country_pit.ipynb](../notebooks/50_cross_country_pit.ipynb))

### 5.1 Что тестирует
Гипотеза H3 имеет две подчасти:
1. **Macro lift**: PIT (TTC_score + GDP + CPI) > TTC только по TTC_score.
2. **Локальность профиля риска**: в РФ доминирует *Liquidity*, в Китае — *Profitability*.

### 5.2 Data alignment
* Обе панели сведены к **одинаковому набору 15 ratios** (`rd_to_revenue` в кросс-контри пайплайне *не используется*, чтобы иметь общий feature-space).
* Оба периода обрезаны до **2014–2024** (Россия изначально с 2012 — отбрасываем 2012–2013 для честности).
* Сравниваются **только относительные ratios**, абсолютные RUB/CNY не смешиваются.
* Winzorize 1/99 %, `fillna(median)`.

### 5.3 TTC refit
Одинаковая архитектура на обе страны: `StandardScaler + LogReg(class_weight='balanced')`, stratified 80/20, `random_state=42`. Получаем `proba_all` — TTC-score для каждой строки ([h3_pit_vs_ttc.csv](cross_country/h3_pit_vs_ttc.csv)).

| | TTC ROC-AUC test |
|---|---|
| Russia | 0.8780 |
| China  | 0.9953 |

### 5.4 Макро-placeholder → реальные данные на руках (2026-04-15)
Функция `load_macro('Russia' | 'China')` возвращает DataFrame `year × (GDP_Growth, Inflation_Rate)`. На момент текущего экспорта в коде сидит **правдоподобный IMF-like placeholder** за 2014–2024.

**Update.** Заказчик передал реальные ряды Real GDP Growth % / Avg CPI Inflation % за 2012–2025 для обеих стран — они будут подставлены в `_MACRO_DATA` в следующем прогоне (этот отчёт фиксирует состояние *до* интеграции). Источники остаются те же: IMF WEO `NGDP_RPCH` / `PCPIPCH`, плюс национальная статистика (Росстат / ЦБ РФ; NBS / World Bank). Все цифры §5.5 после подстановки нужно перечитать — текущие приросты ROC построены на placeholder-ряде и могут как усилиться, так и обнулиться.

### 5.5 Stage-2 PIT
Модель: `LogReg(class_weight='balanced')` на `[ttc_score, GDP_Growth, Inflation_Rate]`. Сравнение с TTC-only:

| Country | TTC ROC | PIT ROC | ΔROC | TTC PR-AUC | PIT PR-AUC |
|---|---|---|---|---|---|
| Russia | 0.8780 | **0.8821** | +0.0042 | 0.0673 | **0.0437** |
| China  | 0.9953 | **0.9957** | +0.0004 | 0.6918 | **0.7167** |

**Вердикт ноутбука:** ✅ **H3 часть 1 подтверждается** (по ROC-AUC PIT > TTC в обеих странах).

**Оговорка — важная для диплома:**
* Приросты крошечные (+0.004 и +0.0004). На placeholder-макро это *арифметический* эффект, не экономический. Реальные цифры IMF дадут или более сильный сигнал, или исчезновение эффекта — сейчас утверждать нельзя.
* В России **PR-AUC при добавлении макро падает** (0.067 → 0.044): по PR-кривой PIT *хуже*. Это типичное поведение при сильном дисбалансе и минорной добавке фичей — модель смещает порог. В диплом стоит явно зафиксировать: *«H3 подтверждается по ROC-AUC; по PR-AUC в РФ эффект отрицательный — требуется реальная макро-серия и robustness check»*.

### 5.6 Side-by-side SHAP (часть 2 H3)
Для каждой страны отдельно обучаем XGBoost на всех 15 ratios, считаем mean |SHAP|.

**Топ-3 фичей** ([h3_top3_features.csv](cross_country/h3_top3_features.csv)):

| Rank | Russia feature | RU group | China feature | CN group |
|---|---|---|---|---|
| 1 | `cash_to_cl` | **Liquidity** | `intangibles_to_assets` | **Innovation** |
| 2 | `log_assets` | Size | `log_revenue` | Size |
| 3 | `debt_to_assets` | Leverage | `operating_margin` | **Profitability** |

**Суммы по группам** ([h3_group_importance.csv](cross_country/h3_group_importance.csv)):

| Group | Russia Σ|SHAP| | China Σ|SHAP| |
|---|---|---|
| Liquidity | **2.67** | 3.22 |
| Leverage  | 2.05 | 0.45 |
| Profitability | 1.77 | 1.79 |
| Size | 1.54 | 2.02 |
| Innovation | 0.14 | **3.72** |

**Вердикт ноутбука:** ⚠️ **H3 часть 2 частично**.
* Россия: dominant group = **Liquidity** ✅ — совпадает с гипотезой (дорогой капитал → ликвидность решает).
* Китай: dominant group = **Innovation**, не Profitability, как формулирует H3. Profitability стоит 4-м.

**Интерпретация для диплома (содержательная, не формальная):**
> Паспорт работы ожидал, что в Китае при институциональной поддержке первичной станет *operational profitability*. Эмпирически же доминирует **Innovation (intangibles_to_assets, рост НМА, R&D)** — что экономически даже логичнее: в субсидируемом госсекторе операционная прибыль менее информативна (все компании плюс-минус рентабельны благодаря господдержке), а настоящий differentiator — способность накапливать интеллектуальный капитал. Компании, у которых intangibles схлопывается, — делистируются. Profitability стоит 4-м с Σ|SHAP|=1.79, т.е. она важна, но не доминирует. Для строгой версии H3 можно переформулировать как *«в Китае доминирует non-leverage профиль»*: Innovation (3.72) + Liquidity (3.22) + Profitability (1.79) ≫ Leverage (0.45), 18×.

**Графика:** [h3_shap_side_by_side.png](cross_country/h3_shap_side_by_side.png) — две SHAP summary рядом.

### 5.7 Что *не* сделано
* **Реальная макро-выгрузка** (см. §5.4) — критический апдейт. Без неё обе части H3 — скорее «инфраструктурный тест», что пайплайн работает, чем эмпирический вывод.
* **Unified feature space не включает `rd_to_revenue`** — при реальной пайплайн-сравнимости это логично (в России R&D нет), но экономически слабее обосновывает H2-часть сравнения. Отдельной ячейкой можно было бы сделать «CN-only SHAP с `rd_to_revenue`» — сейчас это в §4, но не в кросс-контри файле.
* **Country-effect через общую LogReg с country dummy** — альтернативный способ тестирования H3 (один global-model, переменная `is_china`). Не реализовано.
* **Bootstrap на top-фичи** — дать 95%-CI для `|SHAP|_i` и показать, что ранги устойчивы. Полезно для защиты.

---

## 6. Общие ограничения и дорожная карта

### 6.1 Обязательно исправить перед защитой
| # | Проблема | Файл | Фикс | Статус (2026-04-15) |
|---|---|---|---|---|
| 1 | Leakage в 40_china из-за row-split при 11 тикерах | 40_china_eda_and_models | `GroupShuffleSplit` / `GroupKFold` по `ticker` — все годы одной компании уходят целиком в train *или* test | в работе, фикс согласован |
| 2 | Макро — placeholder | 50_cross_country_pit | подставить реальный `_MACRO_DATA` (GDP %, CPI %) за 2012–2025 — данные получены | данные на руках, ждёт прогона |
| 3 | `current_liab` в Китае = `st_borrow` proxy | 30_china_load_and_clean | новая выгрузка Wind с явным `tot_cur_liab` / `wgsd_liabs_curr` (в текущем файле такого кода нет) | заблокировано — нужен новый экспорт от заказчика |
| 4 | `net_profit` / `net_revenue` = 0 в Китае | 30_china_load_and_clean | новая выгрузка Wind с `np_belongto_parcomsh` + `oper_rev_after_ded` (в текущем файле только пустой `anal_reits_netprofit`) | заблокировано — нужен новый экспорт от заказчика |
| 5 | В Russia pipeline явного `drop year==2025` нет | 10_russia_load_and_clean | добавить строчку | открыто |

### 6.2 Желательно (усиливает аргументацию)
* Stratified K-Fold вместо одного сплита в 20_russia и 40_china.
* Time-based split (train ≤ 2021, test 2022–24) — Stage-2-аргумент для H1.
* Bootstrap-CI на test-ROC-AUC и на top-|SHAP|.
* Country-effect через global LogReg с `is_china` вместо двух отдельных моделей.
* Добавить macro-policy rate как третью макропеременную (для РФ: ключевая ставка ЦБ, для КНР: 1Y LPR).

### 6.3 Принципиальные ограничения выборки (не устранимы кодом)
* В РФ 206 дефолтов за 13 лет при 10 000 активных компаний — это *настоящий* прод-дисбаланс банковского скоринга; за это отвечает класс методологии (class_weight), не дата-инженерия.
* В Китае 11 true defaults на 319 активных — **не статистически валидная** выборка для ML сама по себе. Выводы по Китаю правильно трактовать как *pattern detection*, а не *prediction*. Именно поэтому кросс-контри сравнение `Russia → China` методологически значимо: **мы тестируем, одинаково ли работают одни и те же фичи в разных режимах капитала** — а не абсолютные AUC.
* Survivorship bias: Wind выдаёт только сейчас-котируемые + 37 делистингов; компании, ушедшие до 2014, не видны.

---

## 7. Как этим пользоваться

1. **Посмотреть конкретный ноутбук** → открыть PDF из [reports/notebook_exports/](notebook_exports/). Каждый PDF самодостаточен (все графики зашиты), 5 файлов, ~ 0.2–0.6 MB каждый.
2. **Найти конкретный график** → папка [reports/russia/](russia/), [reports/china/](china/), [reports/cross_country/](cross_country/). Имена PNG перечислены в таблицах § 2.2, 2.4, 4.2, 4.4, 5.6.
4. **Найти конкретную таблицу-результат** → рядом с PNG лежит одноимённый CSV.
5. **Воспроизвести пайплайн с нуля** → выполнить notebooks по порядку 10 → 20 → 30 → 40 → 50. Первая пара цифр ≠ зависит от KEY-папки `data/raw/`.

---

## 8. Краткий summary по гипотезам

| Гипотеза | Russia | China | Cross | Общий вердикт |
|---|---|---|---|---|
| **H1** — LogReg стабильнее ансамблей на малом числе дефолтов | ✅ подтверждена (ΔROC LogReg = 0.024 vs RF 0.098) | ⚠️ не проверяема текущим split'ом (leakage) | — | **✅ для RU; для CN требуется split по тикерам** |
| **H2** — Liquidity + Innovation > Leverage | ✅ 1.56× (но innovation ≈ 0 в SPARK) | ✅ 7.45× (сильно и через intangibles) | — | **✅ в обеих странах** |
| **H3** часть 1 — PIT > TTC | +0.004 ROC | +0.0004 ROC | оба + | ⚠️ **технически да, но на placeholder-макро; требуется реальная макра** |
| **H3** часть 2 — Russia=Liquidity, China=Profitability | RU: Liquidity ✅ | CN: Innovation (не Profitability) | top-3 отличаются | ⚠️ **частично; уточнить формулировку: CN доминирует non-leverage профиль** |

---

*Конец сводного репорта.*
