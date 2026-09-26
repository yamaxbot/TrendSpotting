# On-Demand Parsing

> Историческое описание раннего парсера. Актуальные файлы, пути и тесты: [README.md](README.md).

Модуль собирает научные публикации из OpenAlex по произвольному технологическому направлению пользователя и формирует очищенный корпус для дальнейшего поиска слабых сигналов.

Запускается только:

```bash
python on_demand_parsing/parser.py
```

---

# Структура

```text
on_demand_parsing/
├── data/
│   ├── output1/
│   │   ├── raw_works.json
│   │   ├── clean_works.json
│   │   └── yearly_stats.json
│   ├── output2/
│   └── ...
│
├── parser.py
├── query_normalizer.py
├── request_validation.py
├── openalex_client.py
├── topic_discovery.py
├── topic_scoring.py
├── main_collection.py
├── normalization.py
├── relevance_filter.py
├── storage.py
└── PARSING.md
```

`parser.py` — основной оркестратор pipeline.

Остальные Python-файлы содержат отдельные части логики парсера и вызываются из основного pipeline.

`request_validation.py` присутствует в структуре проекта как отдельный модуль валидации запросов.

---

# Pipeline

```text
User Query
    ↓
Query Normalization
    ↓
Seed Search
    ↓
Topic Discovery
    ↓
Topic Scoring
    ↓
Topic Selection
    ↓
Topic Expansion / Main Collection
    +
Exploration
    ↓
Deduplication
    ↓
Normalization
    ↓
Relevance Scoring
    ↓
Relevance Filtering
    ↓
Clean Corpus
    ↓
Save Results
```

---

## 1. Query Normalization

Файл: `query_normalizer.py`

Запрос пользователя преобразуется в короткий англоязычный поисковый запрос для OpenAlex.

Пример:

```text
"Найди перспективные технологии защиты промышленных систем от кибератак"

→

"industrial control systems cybersecurity"
```

LLM используется только для преобразования запроса.

Он не генерирует технологии или слабые сигналы.

---

## 2. Seed Search

Файл: `openalex_client.py`

По нормализованному запросу загружается около 1000 публикаций OpenAlex.

Seed Works используются для первичного исследования научного направления и обнаружения связанных OpenAlex Topics.

---

## 3. Topic Discovery

Файл: `topic_discovery.py`

Из Seed Works извлекаются OpenAlex Topics.

Для каждого Topic сохраняются:

```text
Topic ID
Topic Name
Frequency
```

`frequency` показывает, в скольких Seed Works встретился Topic.

---

## 4. Topic Scoring

Файл: `topic_scoring.py`

Используется embedding-модель:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Для каждого Topic вычисляется semantic similarity между названием Topic и нормализованным запросом пользователя.

Получаем:

```text
topic
frequency
coverage
similarity
```

---

## 5. Topic Selection

Выбираются Topics, которые достаточно связаны с запросом.

Текущие baseline-правила:

```python
core = similarity >= 0.50 and coverage >= 0.01

niche = similarity >= 0.65 and frequency >= 2

supported = similarity >= 0.30 and coverage >= 0.05
```

Topic проходит, если выполняется хотя бы одно из правил.

OpenAlex Topics используются для навигации и расширения корпуса.

Они НЕ считаются найденными технологиями или слабыми сигналами.

---

## 6. Main Collection

Файл: `main_collection.py`

По ID выбранных Topics загружается основной корпус публикаций.

Текущий лимит разработки:

```text
MAIN_LIMIT = 5000
```

Общий бюджет распределяется между выбранными Topics.

---

## 7. Yearly Statistics

Для каждого выбранного Topic отдельно запрашивается количество публикаций OpenAlex за:

```text
2018–2026
```

Эти данные используются для последующего temporal analysis.

Важно: OpenAlex может ретроспективно классифицировать старые статьи современным Topic.

Поэтому наличие публикаций у Topic в определённом году не означает, что технология под современным названием Topic появилась именно тогда.

---

## 8. Exploration

Чтобы система не зависела только от OpenAlex Topics, дополнительно выполняется прямой поиск по нормализованному запросу.

Текущий лимит:

```text
EXPLORATION_LIMIT = 1000
```

Exploration помогает находить публикации, которые могли не попасть в выбранные Topics.

---

## 9. Deduplication

Результаты Main Collection и Exploration объединяются.

Сейчас дубликаты удаляются по:

```text
OpenAlex ID
```

После этого формируется RAW corpus.

---

## 10. Normalization

Файл: `normalization.py`

Все уникальные публикации преобразуются из большого OpenAlex JSON в единую компактную структуру:

```text
id
doi
title
abstract
publication_year
publication_date
topics
keywords
authors
institutions
cited_by_count
source
language
type
```

Также восстанавливается обычный текст abstract из `abstract_inverted_index`.

Нормализация выполняется для всех уникальных статей до relevance filtering.

---

## 11. Relevance Scoring

Файл: `relevance_filter.py`

Для каждой статьи используется:

```text
title + abstract
```

Embedding-модель:

```text
sentence-transformers/all-MiniLM-L6-v2
```

сравнивает содержание статьи с нормализованным запросом.

В структуру статьи добавляется:

```json
"relevance": 0.684
```

После этого статьи сортируются от наиболее релевантных к наименее релевантным.

---

## 12. Relevance Filtering

На текущем baseline удаляются нижние 15% статей по `relevance`:

```text
remove_ratio = 0.15
```

Цель — убрать очевидный нерелевантный хвост без использования фиксированного абсолютного similarity threshold.

Оставшиеся публикации образуют Clean Corpus.

---

# Сохранение результатов

Файл: `storage.py`

Все результаты сохраняются в папке:

```text
on_demand_parsing/data/
```

То есть `data` находится в той же директории, что и скрипты парсера.

Для каждого запуска внутри `data` создаётся отдельная папка:

```text
data/
├── output1/
├── output2/
├── output3/
└── ...
```

Внутри каждого `outputN`:

```text
outputN/
├── raw_works.json
├── clean_works.json
└── yearly_stats.json
```

### raw_works.json

Уникальные исходные объекты OpenAlex после:

```text
Main Collection
+ Exploration
+ Deduplication
```

Сохраняется исходная структура OpenAlex, чтобы при необходимости можно было использовать дополнительные поля без повторного скачивания данных.

### clean_works.json

Основной результат парсера.

Содержит нормализованные статьи после:

```text
Normalization
→ Relevance Scoring
→ Relevance Filtering
```

Именно этот корпус передаётся следующим этапам поиска слабых сигналов.

### yearly_stats.json

Содержит количество публикаций по выбранным OpenAlex Topics за каждый год.

Используется для дальнейшего анализа временной динамики.

---

# Что нужно доработать

## 1. Topic Selection

Текущие similarity/frequency thresholds являются baseline.

Нужно добавить Topic co-occurrence.

Для Topic рассчитывать:

```text
core_ratio =
число работ, где Topic встречается вместе с core topics
/
frequency Topic
```

После этого учитывать совокупность:

```text
semantic similarity
+ frequency / coverage
+ co-occurrence
```

Это позволит не терять узкие, но действительно связанные направления.

---

## 2. Selected Topics = 0

На узких запросах возможна ситуация:

```text
Query: SHAP
Seed Works: 1000
Unique Topics: 675
Selected Topics: 0
```

Pipeline не должен из-за этого терять найденные публикации.

Нужно добавить fallback:

```text
Selected Topics > 0
    → Topic Expansion + Exploration

Selected Topics = 0
    → Seed Works + Exploration
```

---

## 3. Deduplication

Сейчас дедупликация выполняется только по OpenAlex ID.

В тестах после неё оставались публикации с одинаковыми названиями.

Нужно улучшить дедупликацию:

```text
OpenAlex ID
    ↓
DOI
    ↓
normalized title
```

---

## 4. Relevance Filtering

Удаление нижних 15% — временный baseline.

Нужно подобрать более устойчивый метод на тестовой выборке, чтобы не удалять редкие релевантные статьи и одновременно эффективнее убирать шум.

---

## 5. Exploration Statistics

Сейчас статистика не показывает, сколько именно новых уникальных публикаций дал Exploration.

Нужно отдельно выводить:

```text
main_unique
exploration_downloaded
exploration_new_unique
final_unique
```

---

## 6. Yearly Stats Optimization

Сейчас выполняется отдельный API-запрос для каждого:

```text
Topic × Year
```

Это увеличивает время работы.

Нужно исследовать возможности aggregation/grouping OpenAlex и уменьшить количество API-запросов.

---

## 7. OpenAlex Topics ≠ Weak Signals

OpenAlex Topics нельзя использовать как финальные найденные технологии.

Topics нужны для построения релевантного корпуса.

Финальные weak signals должны обнаруживаться на следующих этапах из:

```text
titles
abstracts
keywords
phrases/entities
semantic clusters
temporal dynamics
```

---

# Текущее состояние

```text
[✓] Query Normalization
[✓] Seed Search
[✓] Topic Discovery
[✓] Topic Scoring
[✓] Baseline Topic Selection
[✓] Topic Expansion
[✓] Main Collection
[✓] Yearly Statistics
[✓] Exploration
[✓] Deduplication
[✓] Normalization
[✓] Relevance Scoring
[✓] Relevance Filtering
[✓] JSON Persistence
```

---

# Следующие этапы системы

После парсера:

```text
Clean Corpus
    ↓
Phrase / Entity Extraction
    ↓
Embeddings
    ↓
Clustering
    ↓
Temporal Features
    ↓
Weak Signal Scoring / ML
    ↓
Mature Trend / Hype / Noise Filtering
    ↓
TOP-15 Weak Signals
    ↓
Explanation + Sources
```

Парсер отвечает за получение качественного корпуса данных.

Определение самих слабых сигналов выполняется на следующих этапах системы, а не на основе названий OpenAlex Topics.
