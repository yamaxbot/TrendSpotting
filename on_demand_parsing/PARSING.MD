# Поиск статей по запросу

Папка собирает ограниченный корпус OpenAlex для нового запроса сайта. Точка входа — `parser.run_parser()`, которую вызывает `website/main.py`.

## Структура

| Файлы | Роль |
|---|---|
| `request_validation.py`, `query_normalizer.py` | Проверка запроса и короткая англоязычная формулировка для поиска |
| `openalex_client.py` | Запросы к OpenAlex с ключом API |
| `topic_discovery.py`, `topic_scoring.py` | Получение, оценка и выбор тем из стартовых работ |
| `main_collection.py`, `query_recovery.py` | Расширенный сбор по темам и восстановление результатов для узких запросов |
| `normalization.py`, `relevance_filter.py`, `embedding_model.py` | Восстановление аннотаций, нормализация, семантический отбор и эмбеддинги |
| `parser.py`, `storage.py` | Сборка/экспорт корпуса Parquet и отладочных JSON |
| `time_budget.py` | Бюджет поиска 19 минут 30 секунд; при исчерпании используется уже собранное |
| `test_optimization.py`, `test_time_budget.py` | Тесты сбора и ограничения времени |

Последовательность: нормализация запроса → стартовые работы → выбор тем → расширенный сбор → удаление дублей → проверка близости → корпус `website/data/parse_corpus_<запрос>.parquet`. Основной предел сбора — 5000 работ (`MAIN_LIMIT` в `main_collection.py`). Отладочные JSON пишутся в `logics/` относительно рабочего каталога процесса; для работы сайта они не являются входными данными. Признаки и окончательные 15 карточек строятся позже в `preprocessing/` и `website/`.

Тесты из корня проекта: `python -m unittest on_demand_parsing.test_optimization on_demand_parsing.test_time_budget`.
