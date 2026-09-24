# new_dataset (2017-2026, 500k/year)

Корпус: 10 лет × 500 000 = **5 000 000** статей.
Изолирован от parent `download_dataset/` (соавтор): всё в `new_dataset/`.

## Жёсткое правило сплита

| Годы | Роль | В CatBoost? |
|---|---|---|
| 2017–2019 | past support (окно t−3..t−1) | **нет, никогда** |
| 2020–2021 | train | да |
| 2022 | validation | да |
| 2023 | test | да |
| 2024–2026 | future support (окно t+1..t+3) | **нет** |

`training_end_year = 2023`. Pipeline пишет `split`:
`past_support` / `train_candidates` / `valid_candidates` / `test_candidates` / `target_support`.

Метки ESI считает `preprocessing/create_features.py` по всему корпусу (support года нужны для счётчиков), потом `education/` режет только 2020–2023.

## Окна таргета

```text
past:   t-3, t-2, t-1
future: t+1, t+2, t+3
```

С корпусом 2017–2026 все ML-годы имеют полные окна:

| Год статьи | past | future |
|---|---|---|
| 2020 | 2017–2019 ✓ | 2021–2023 ✓ |
| 2021 | 2018–2020 ✓ | 2022–2024 ✓ |
| 2022 | 2019–2021 ✓ | 2023–2025 ✓ |
| 2023 | 2020–2022 ✓ | 2024–2026 ✓ |

## Финальный файл

`data/new_data/openalex_corpus_5m.parquet` (остальное в `data/` не трогаем)

## Required columns для create_features.py

Все 14 колонок на месте (doc_id, source_tier, primary_topic_id, pub_year, authors_ids, authorships_json, abstract_text, title, counts_by_year, domain_id, source_id, referenced_works_ids, commercial_maturity_index, has_ref_data).

## Сознательно НЕ в этом pipeline

- `targets.py` — лейблы в create_features
- `historical_citations()` — не нужен для baseline
