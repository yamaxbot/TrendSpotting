# Как запускать new_dataset pipeline

> Исторический план на 5 млн работ: пути и объём ниже устарели. Текущая конфигурация — `new_dataset/config_new.json`, краткая инструкция — [README.md](README.md).

Изоляция: только `download_dataset/new_dataset/`. Parent `download_dataset/` не трогать.

1. В `TrendSpotting/.env` — `OPENALEX_API_KEY`.
2. Collect (резюмируется):

```powershell
cd c:\Users\sevam\Desktop\papers\TrendSpotting\download_dataset\new_dataset
python -m pipeline collect --config config_new.json
```

Всё пишется в `data/new_data/` (`state.sqlite`, batches, parquet, log).
Содержимое `data/` кроме `new_data/` не трогаем.

3. Статус:

```powershell
python -m pipeline status --config config_new.json
```

4. После добора квот:

```powershell
python -m pipeline export --config config_new.json
python -m pipeline validate --config config_new.json
```

5. Финал: `download_dataset/new_dataset/data/new_data/openalex_corpus_5m.parquet`

Дальше (твои зоны):

```text
preprocessing/create_features.py  INPUT → download_dataset/new_dataset/data/new_data/openalex_corpus_5m.parquet
education/test/train_model.py     train 2020-2021 | valid 2022 | test 2023
                                  # 2017-2019 и 2024-2026 в обучение НЕ идут
```

## Схема корпуса

- Годы: **2017–2026**, 500k/год → **5 000 000** строк
- 4 домена; Physical: ≥25% Engineering (field 22)
- Quality: `has_abstract:true`, `referenced_works_count:>0`

## ML-сплит

| split | years | rows (ожид.) |
|---|---|---|
| past support | 2017–2019 | ~1 500 000 (только для past-окна) |
| train | 2020–2021 | ~1 000 000 |
| valid | 2022 | ~500 000 |
| test | 2023 | ~500 000 |
| target support | 2024–2026 | ~1 500 000 (только для future-окна) |

Бюджет OpenAlex на 5M заметно больше, чем на 3M — collect крутить с `--max-requests` и резюмом.
