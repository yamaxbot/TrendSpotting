# `download_dataset/` — офлайн-корпуса

Эта папка нужна для подготовки обучающих данных, не для ответа на запрос сайта.

- `pipeline/`, `config.json`: первый сборщик OpenAlex, конфигурация на 1 млн работ за 2010–2026. CLI: `collect`, `targets`, `export`, `validate`, `status`; метка считается отдельной командой `targets`.
- `new_dataset/`: следующий изолированный эксперимент с корпусом 2017–2026 и явным делением годов на обучение, проверку и опорные окна. Текущая конфигурация — `new_dataset/config_new.json` (1 млн работ, батчи по 10 тыс.).
- [`FEATURES.md`](FEATURES.md): исходные определения и ограничения признаков первого корпуса; не руководство по текущему сайту.
- `tests/test_pipeline.py`: тесты первого сборщика.

Первый сборщик запускается **из этой папки**: `python -m pipeline status --config config.json`; аналогично `collect`, `targets`, `export`, `validate`. `OPENALEX_API_KEY` читается из окружения или корневого `.env`. Данные первого эксперимента пишутся в `download_dataset/data/` и не входят в Git/Docker-образ. Не смешивайте их с `new_dataset/new_data/`.

Второй сборщик запускается **из `new_dataset/`**: `python -m pipeline collect --config config_new.json`, затем `status`, `export` или `validate`. Команды `targets` там нет: метка строится позднее в `preprocessing/`. По `config_new.json` годы 2017–2019 служат исторической опорой, 2020–2021 — train, 2022 — validation, 2023 — test, 2024–2026 — будущим окном метки. Старые планы на 5 млн работ сохранены в [`FEATURES_NEW.md`](FEATURES_NEW.md) и [`RUN_NEW_DATASET.md`](RUN_NEW_DATASET.md); действующий объём задаёт конфигурация.

Проверка первого сборщика: из `download_dataset/` выполнить `python -m unittest discover -s tests`.

Для сайта скачивать миллион работ не требуется: он запрашивает публикации по пользовательскому направлению при поиске.
