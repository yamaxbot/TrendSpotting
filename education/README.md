# `education/` — обучение и проверки

Это офлайн-эксперименты; при открытии сайта обучение не происходит.

- `test/train_model.py`: бинарный CatBoost с разделением по годам; результаты по умолчанию в `outputs/catboost/`.
- `test/traing_random_temporal.py`: сопоставление случайного и временного разделений на одном наборе 2020–2023; результаты в `outputs/catboost_splits/`. Сайт берёт модель из `random/`.
- `test/experiments.py`: сравнение бинарной классификации, регрессии и ранжирования; отдельные результаты находятся в `outputs/catboost/`.
- `test/features_check.py`: диагностика таблицы признаков.
- `baseline.ipynb`, корневые `catboost_model.cbm` и `processed_ml_dataset.parquet`, артефакты в `test/`: ранние эксперименты, не загружаемые сайтом.

[`EDUCATION.md`](EDUCATION.md) сохраняет подробности прежнего baseline; его числа не являются метриками выбранной сейчас моделью сайта. Актуальное сравнение — [`../outputs/README.md`](../outputs/README.md).

После формирования `download_dataset/new_dataset/new_data/final_openalex_dataset.parquet` скрипты запускаются из корня проекта, например `python education/test/traing_random_temporal.py`. У обучающих скриптов есть `--dataset` и `--output-dir`; `experiments.py` дополнительно принимает `--mode binary|regression|ranking`. Файлы `test/metrics.json`, `test/confusion_matrix*` и локальная модель относятся к прежним пробам, не к рабочему сайту.
