# Офлайн-сбор корпуса OpenAlex

Эта папка создаёт обучающие корпусы; сайт при пользовательском поиске её не запускает.

## Структура

| Путь | Роль |
|---|---|
| `config.json`, `pipeline/` | Первый сборщик: поиск по стратам, исторические метки (`targets.py`), экспорт и проверка |
| `new_dataset/config_new.json`, `new_dataset/pipeline/` | Новый независимый сборщик с временными группами `past_support`, `train`, `valid`, `test`, `target_support` |
| `data/`, `new_dataset/new_data/` | Генерируемые партии Parquet, SQLite-состояние, журналы, экспорт и отчёты проверки |
| `tests/` | Тесты первого сборщика |
| `FEATURES.md`, `FEATURES_NEW.md` | Заметки о данных и методике признаков; не команды запуска |

Актуальный размер нового корпуса задаёт `new_dataset/config_new.json` (`total: 1000000`), а не старое название плана на 5 млн. Результат по текущей конфигурации — `new_dataset/new_data/openalex_corpus_1m.parquet`.

## Как запустить новый сборщик

Заполните `OPENALEX_API_KEY` в корневом `.env`. Из `download_dataset/new_dataset/` выполните:

```bash
python -m pipeline collect --config config_new.json
python -m pipeline status --config config_new.json
python -m pipeline validate --config config_new.json
```

Сбор возобновляется из SQLite-состояния; `export` можно вызвать отдельно. Команда `validate` ожидает полный объём корпуса и потому может не пройти на частичной выгрузке. Для первого сборщика перейдите в `download_dataset/` и запустите `python -m pipeline collect --config config.json`; затем отдельно доступны команды `targets`, `export`, `validate` и `status`.

После выгрузки `preprocessing/create_features.py` строит таблицу признаков. Обучение модели находится в `education/`, не в этом сборщике.
