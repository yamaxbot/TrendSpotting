"""Bounded-memory Parquet export and independent corpus validation."""
from __future__ import annotations

import collections
import contextlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .common import database

STRINGS = ["doc_id", "pub_date", "title", "abstract_text", "source_tier", "primary_topic",
           "primary_topic_id", "domain_id", "field_id", "counts_by_year", "authorships_json",
           "topics_json", "source_id", "doi", "fetched_at", "stratum", "target_status",
           "target_population", "split"]
INTEGERS = ["pub_year", "citations_at_cutoff", "has_ref_data", "target_emergence",
            "new_authors_baseline", "new_authors_future"]
FLOATS = ["commercial_maturity_index", "affiliation_coverage", "future_growth_ratio",
          "new_authors_growth_ratio"]
LISTS = ["authors_ids", "affiliations", "referenced_works_ids"]
SCHEMA = pa.schema(
    [(name, pa.string()) for name in STRINGS]
    + [(name, pa.int64()) for name in INTEGERS]
    + [(name, pa.float64()) for name in FLOATS]
    + [(name, pa.list_(pa.string())) for name in LISTS]
    + [("is_engineering", pa.bool_())]
)


def checkpoint_batches(config: dict) -> None:
    """Persist complete 50k shards while collection is still running."""
    connection = database(config)
    directory = Path(config["data_dir"]) / "batches"
    directory.mkdir(exist_ok=True)
    row = connection.execute("SELECT value FROM metadata WHERE key='sharded_rows'").fetchone()
    saved = int(row[0]) if row else 0
    count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
    size = config["batch_size"]
    while count - saved >= size:
        rows = connection.execute("SELECT payload FROM documents ORDER BY rowid LIMIT ? OFFSET ?",
                                  (size, saved)).fetchall()
        table = pa.Table.from_pylist([json.loads(row[0]) for row in rows], schema=SCHEMA)
        path = directory / f"part-{saved // size:05d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
        saved += size
        with connection:
            connection.execute("INSERT OR REPLACE INTO metadata VALUES ('sharded_rows',?)", (str(saved),))
    connection.close()


def export(config: dict) -> Path | None:
    connection = database(config)
    count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
    if not count:
        connection.close()
        return None
    root = Path(config["data_dir"])
    directory = root / "batches"
    directory.mkdir(exist_ok=True)
    # A partial export is never presented under the final 1m filename.
    output = root / ("openalex_corpus_1m.parquet" if count == config["total"] == 1000000
                     else "openalex_corpus_partial.parquet")
    temporary = output.with_suffix(".parquet.tmp")
    cursor = connection.execute("SELECT payload FROM documents ORDER BY rowid")
    shard_names = []
    inference = root / "inference" / "candidates.parquet"
    training = root / "training" / "labeled.parquet"
    for subset in (inference, training):
        subset.parent.mkdir(exist_ok=True)
    with contextlib.ExitStack() as stack:
        writer = stack.enter_context(pq.ParquetWriter(temporary, SCHEMA, compression="zstd"))
        inference_writer = stack.enter_context(pq.ParquetWriter(inference.with_suffix('.parquet.tmp'), SCHEMA, compression="zstd"))
        training_writer = stack.enter_context(pq.ParquetWriter(training.with_suffix('.parquet.tmp'), SCHEMA, compression="zstd"))
        index = 0
        while rows := cursor.fetchmany(config["batch_size"]):
            table = pa.Table.from_pylist([json.loads(row[0]) for row in rows], schema=SCHEMA)
            shard = directory / f"part-{index:05d}.parquet"
            shard_tmp = shard.with_suffix(".parquet.tmp")
            pq.write_table(table, shard_tmp, compression="zstd")
            shard_tmp.replace(shard)
            shard_names.append(shard.name)
            writer.write_table(table)
            inference_writer.write_table(table.filter(pc.greater(table["pub_year"], config["training_end_year"])))
            training_writer.write_table(table.filter(pc.is_valid(table["target_emergence"])))
            index += 1
    temporary.replace(output)
    for subset in (inference, training):
        subset.with_suffix('.parquet.tmp').replace(subset)
    manifest = {"rows": count, "batches": shard_names, "file": output.name,
                "expected_rows": config["total"], "collection_complete": count == config["total"]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    connection.close()
    return output


def validate(path: Path, expected: int | None = None) -> dict:
    years, domains, engineering, targets = (collections.Counter() for _ in range(4))
    nulls = collections.Counter()
    ids = set()
    duplicates = rows = invalid_targets = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=10000, columns=[
        "doc_id", "pub_year", "domain_id", "abstract_text", "is_engineering", "target_emergence"
    ]):
        for record in batch.to_pylist():
            rows += 1
            doc_id = record["doc_id"]
            duplicates += doc_id in ids
            ids.add(doc_id)
            for field in ("doc_id", "pub_year", "abstract_text"):
                if record[field] is None or record[field] == "":
                    nulls[field] += 1
            year = record["pub_year"]
            years[str(year)] += 1
            domains[str(record["domain_id"])] += 1
            if record["is_engineering"]:
                engineering[str(year)] += 1
            targets[str(record["target_emergence"])] += 1
            invalid_targets += year > 2020 and record["target_emergence"] is not None
    result = {"rows": rows, "expected_rows": expected, "by_year": dict(sorted(years.items())),
              "by_domain": dict(domains), "engineering_by_year": dict(engineering),
              "targets": dict(targets), "critical_nulls": dict(nulls),
              "duplicate_doc_ids": duplicates, "invalid_future_targets": invalid_targets,
              "passed": not nulls and not duplicates and not invalid_targets
                        and (expected is None or rows == expected)}
    path.with_suffix(".validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
