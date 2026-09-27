# Обучение и эксперименты

Папка содержит офлайн-обучение на таблице признаков. Во время работы сайта код отсюда не запускается.

## Структура

| Путь | Роль |
|---|---|
| `test/traing_random_temporal.py` | Сравнение CatBoost на случайном и временном разбиении; результаты в `outputs/catboost_splits/` |
| `test/train_model.py` | Отдельный временной эксперимент с классификатором |
| `test/experiments.py` | Эксперименты с ранжированием и регрессией |
| `test/features_check.py`, `test/feature_columns.json` | Проверка и список признаков |
| `test/metrics.json`, CSV, PNG, `catboost_model.cbm` | Сохранённые результаты локального эксперимента |
| `baseline.ipynb`, `processed_ml_dataset.parquet`, `catboost_model.cbm` | Более ранний baseline и его данные/модель |

Основной сценарий: сначала собрать корпус в `download_dataset/`, затем получить признаки в `preprocessing/`, обучить и оценить модель на train/valid/test. Метрики и файлы опубликованных экспериментов описаны в [`outputs/README.md`](../outputs/README.md). Сайт сейчас загружает модель из `outputs/catboost_splits/random/`, а не файл из этой папки. Случайное разбиение даёт более оптимистичную оценку, чем проверка на новых годах.
