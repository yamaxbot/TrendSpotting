"""Run: python -m pipeline {collect,export,validate,status}."""
import argparse
import json
import logging
from pathlib import Path

from .collect import collect
from .common import Client, Paused, configuration, database
from .export import export, validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "export", "validate", "status"])
    parser.add_argument("--config", default="config_new.json")
    parser.add_argument("--max-requests", type=int, default=9900)
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    config = configuration(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            Path(config["data_dir"]) / "pipeline.log", encoding="utf-8")])
    client = Client(config["requests_per_second"], args.max_requests)
    if args.command == "status":
        connection = database(config)
        result = {"rows": connection.execute("SELECT count(*) FROM documents").fetchone()[0],
                  "strata": connection.execute("""
                    SELECT s.name,s.quota,count(d.doc_id) FROM strata s
                    LEFT JOIN documents d ON s.name=d.stratum GROUP BY s.name ORDER BY s.name
                  """).fetchall()}
        connection.close()
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "validate":
        path = args.file or Path(config["data_dir"]) / config.get(
            "corpus_filename", "openalex_corpus_5m.parquet"
        )
        result = validate(path, config["total"], config["training_end_year"])
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    paused = False
    try:
        if args.command == "collect":
            collect(config, client)
        elif args.command == "export":
            export(config)
    except Paused as exc:
        logging.warning("%s", exc)
        paused = True
    finally:
        path = export(config)
        if path:
            report = validate(path, training_end_year=config["training_end_year"])
            logging.info("Exported %s rows to %s; integrity passed=%s",
                         report["rows"], path, report["passed"])
    return 2 if paused else 0


if __name__ == "__main__":
    raise SystemExit(main())
