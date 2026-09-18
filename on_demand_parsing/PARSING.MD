# On-Demand Parsing

Модуль отвечает за динамический сбор научных публикаций по произвольному технологическому запросу пользователя через OpenAlex.

## Pipeline

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
Topic Expansion
    ↓
Main Collection
    ↓
Exploration
    ↓
Normalization + Deduplication
    ↓
Relevance Filtering
    ↓
Clean Corpus
```

## Файлы

### `parser.py`
Главная точка входа.

Последовательно запускает весь parsing pipeline. Остальные модули содержат отдельные функции и используются через `parser.py`.

### `query_normalizer.py`
Нормализует пользовательский запрос через LLM (KIE API).

- переводит запрос на английский;
- удаляет пользовательское намерение;
- сохраняет предметную область и ограничения;
- не генерирует названия технологий.

Пример:

```text
"Найди перспективные технологии в кибербезопасности"
→ "cybersecurity"
```

### `request_validation.py`
Проверяет доступность KIE API и текущий баланс аккаунта. Используется для диагностики.

### `openalex_client.py`
Клиент OpenAlex API.

Реализовано:

- текстовый поиск публикаций;
- cursor pagination;
- получение Seed Works;
- получение публикаций по OpenAlex Topic ID.

Используется как для Seed Search, так и для Topic Expansion.

### `topic_discovery.py`
Извлекает Topics из Seed Works.

Для каждого Topic сохраняются:

```text
id
name
frequency
```

`frequency` показывает, в скольких Seed Works встретился Topic.

### `topic_scoring.py`
Оценивает и выбирает Topics для расширения поиска.

Для каждого Topic рассчитываются:

- `similarity` — semantic similarity с запросом пользователя;
- `frequency` — количество Seed Works с Topic;
- `coverage` — доля Topic в Seed Corpus.

Для embeddings используется:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Текущий baseline Topic Selection использует правила:

```text
CORE
NICHE
SUPPORTED
```

на основе `similarity` и `frequency / coverage`.

## Topic Expansion

После Topic Selection используются OpenAlex Topic ID для получения дополнительных публикаций.

```text
Selected Topics
      ↓
Topic IDs
      ↓
OpenAlex /works
      ↓
Works по каждому Topic
```

Это позволяет находить релевантные публикации, даже если они напрямую не содержат исходный запрос пользователя.

Topic Expansion протестирован на нескольких направлениях и корректно получает публикации по выбранным Topics.

---

## Текущий статус

```text
[✓] Query Normalization
[✓] Seed Search
[✓] Topic Discovery
[✓] Topic Scoring
[✓] Baseline Topic Selection
[✓] Topic Expansion

[ ] Main Collection
[ ] Exploration
[ ] Normalization
[ ] Deduplication
[ ] Relevance Filtering
[ ] Clean Corpus
```

## Что нужно добавить

### 1. Main Collection

Сформировать основной корпус публикаций из Selected Topics.

Нужно определить:

- общий лимит публикаций;
- распределение лимита между Topics;
- распределение по годам;
- защиту от перекоса в крупные Topics.

Целевой baseline:

```text
~20 000–30 000 works
```

### 2. Exploration

Часть корпуса получать широким поиском по исходному запросу независимо от Selected Topics.

Baseline:

```text
80% → Topic-based collection
20% → Exploration
```

Это снижает риск потери новых направлений из-за ошибок Topic Selection или ограничений OpenAlex Topics.

### 3. Normalization

Привести OpenAlex Works к единой структуре:

```text
openalex_id
doi
title
abstract
publication_date
publication_year
topics
keywords
authors
institutions
cited_by_count
source
language
type
```

### 4. Deduplication

После объединения Seed Search, Topic Expansion и Exploration удалить повторяющиеся публикации.

Основные идентификаторы:

```text
OpenAlex ID
DOI
```

### 5. Relevance Filtering

Удалить публикации, которые попали в корпус из-за слишком широкого Topic/Search.

Планируется semantic filtering относительно исходного запроса пользователя.

### 6. Clean Corpus

После очистки сформировать итоговый корпус для ML/NLP анализа слабых сигналов.

```text
OpenAlex
   ↓
RAW Works
   ↓
Normalization
   ↓
Deduplication
   ↓
Relevance Filtering
   ↓
CLEAN CORPUS
```

## Планируемые улучшения

### Topic Selection
Текущий `similarity + frequency/coverage` является baseline.

В дальнейшем добавить Topic co-occurrence:

```text
Topic
├── similarity
├── frequency / coverage
└── co-occurrence / core_ratio
```

Это позволит лучше находить редкие, но связанные направления и уменьшить количество случайных Topics.

### Query Normalization
Добавить проверку, что результат нормализации действительно получен на английском языке.

Например:

```text
"ии" → "artificial intelligence"
```

а не:

```text
"ии" → "ии"
```

---

## Итоговая цель

Результатом `on_demand_parsing` является не список слабых сигналов, а очищенный корпус публикаций:

```text
User Query
    ↓
On-Demand Parsing
    ↓
Clean Corpus
    ↓
Weak Signal Analysis
    ↓
TOP-15 Weak Signals
```