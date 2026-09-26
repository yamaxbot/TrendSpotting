# `on_demand_parsing/` — поиск по запросу

`parser.py::run_parser` — точка входа, которую вызывает `website/main.py`. На входе направление пользователя; на выходе очищенный Parquet `website/data/parse_corpus_<запрос>.parquet` с метаданными OpenAlex для расчёта признаков.

Ход обработки: `query_normalizer.py` формирует англоязычный запрос; `openalex_client.py` ищет исходные работы; `query_recovery.py` пробует уточняющие варианты для узких запросов; `topic_discovery.py` и `topic_scoring.py` выбирают темы; `main_collection.py` собирает публикации; `normalization.py` восстанавливает аннотации и поля; `relevance_filter.py` выполняет первичную семантическую очистку. `embedding_model.py` лениво загружает общий энкодер `all-MiniLM-L6-v2`. `storage.py` пишет диагностические JSON в каталог `logics/` относительно текущей рабочей папки.

`request_validation.py` — отдельная проверка формулировки, сейчас не вызывается из основного `run_parser`. `test_optimization.py` проверяет сценарии оптимизации/восстановления поиска без внешних API: `python -m unittest on_demand_parsing.test_optimization` из корня проекта.

Дополнительная строгая проверка соответствия запросу выполняется **после ранжирования** в `website/relevance_gate.py`; это отдельный этап, не часть Parquet парсера. Старый [`PARSING.MD`](PARSING.MD) оставлен как история ранней схемы, включая уже неактуальные пути.
