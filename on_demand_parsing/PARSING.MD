# On-Demand Parsing

Модуль отвечает за сбор научных публикаций по произвольному запросу пользователя через OpenAlex.

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
Cleaning
    ↓
Clean Corpus
```

## Файлы

### `parser.py`
Главная точка входа. Последовательно запускает все этапы парсинга. Запускать нужно только этот файл.

### `query_normalizer.py`
Нормализует запрос пользователя через LLM:
- переводит на английский;
- убирает слова вроде «найди», «перспективные», «слабые сигналы»;
- оставляет только предметную область и важные ограничения.

Пример:

```text
"Найди перспективные технологии в кибербезопасности"
→ "cybersecurity"
```

### `request_validation.py`
Проверяет подключение к KIE API и получает текущий баланс аккаунта. Используется для диагностики API.

### `openalex_client.py`
Работает с OpenAlex API. Выполняет поиск публикаций и cursor pagination.

Сейчас используется для получения `1000 Seed Works`.

### `topic_discovery.py`
Извлекает OpenAlex Topics из Seed Works и считает их частоту (`frequency`).

```text
1000 Seed Works
→ 266 Unique Topics
```

### `topic_scoring.py`
Оценивает найденные Topics.

Используются:
- `similarity` — семантическая близость Topic к запросу;
- `frequency` — количество Seed Works с этим Topic;
- `coverage` — `frequency / seed_size`.

На основе этих показателей выполняется baseline Topic Selection (`CORE / NICHE / SUPPORTED`).

## Текущий статус

```text
[✓] Query Normalization
[✓] Seed Search
[✓] Topic Discovery
[✓] Topic Scoring
[✓] Baseline Topic Selection

[ ] Topic Expansion
[ ] Main Collection
[ ] Exploration Search
[ ] Normalization
[ ] Deduplication
[ ] Relevance Filtering
[ ] Clean Corpus
```

## Что доработать

1. **Topic Selection** — добавить co-occurrence между Topics, чтобы не терять редкие релевантные направления.
2. **Topic Expansion** — получать дополнительные публикации по выбранным Topics.
3. **Exploration** — часть корпуса собирать широким поиском независимо от выбранных Topics.
4. **Cleaning** — нормализация данных, удаление дублей и нерелевантных публикаций.
5. **Export** — сохранение RAW данных и итогового Clean Corpus.