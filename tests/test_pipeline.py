import tempfile
import unittest
from pathlib import Path

from pipeline.collect import abstract, allocate, normalize, collect_stratum
from pipeline.common import database, Paused
from pipeline.export import export, validate
from pipeline.targets import growth, new_authors


class PipelineTests(unittest.TestCase):
    def test_resume_keeps_page_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"data_dir": directory, "start_year": 2010, "end_year": 2026,
                      "training_end_year": 2020}
            connection = database(config)
            with connection:
                connection.execute("INSERT INTO strata VALUES ('s','filter',150,200,42,1)")
            class FakeClient:
                paused = True
                def get(self, endpoint, **params):
                    if params['page'] == 2 and self.paused:
                        raise Paused('test interruption')
                    # The second page deliberately repeats one ID.
                    start = 0 if params['page'] == 1 else 99
                    return {'results': [{
                        'id': f'https://openalex.org/W{i}', 'publication_year': 2010,
                        'abstract_inverted_index': {'text': [0]},
                        'referenced_works': ['https://openalex.org/W999'],
                        'primary_topic': {'id': 'https://openalex.org/T1'},
                    } for i in range(start, start + 100)]}
            client = FakeClient()
            with self.assertRaises(Paused):
                collect_stratum(config, client, 's')
            self.assertEqual(connection.execute('SELECT page FROM strata').fetchone()[0], 2)
            self.assertEqual(connection.execute('SELECT count(*) FROM documents').fetchone()[0], 100)
            client.paused = False
            collect_stratum(config, client, 's')
            self.assertEqual(connection.execute('SELECT count(*) FROM documents').fetchone()[0], 150)
            connection.close()

    def test_new_authors_use_first_ever_topic_year(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = database({'data_dir': directory})
            class FakeClient:
                def get(self, endpoint, **params):
                    return {'meta': {'next_cursor': None}, 'results': [
                        {'publication_year': year, 'authorships': [
                            {'author': {'id': 'https://openalex.org/' + author}}]}
                        for year, author in [(2014,'A1'),(2000,'A1'),(2013,'A2'),(2013,'A2')]
                    ]}
            counts = new_authors(connection, FakeClient(), 'T1', 2025)
            self.assertEqual(counts, {2000: 1, 2013: 1})
            connection.close()

    def test_exact_allocation(self):
        result = allocate(1000000, {str(year): 1 for year in range(2010, 2027)})
        self.assertEqual(sum(result.values()), 1000000)
        self.assertEqual(set(result.values()), {58823, 58824})

    def test_abstract_order_and_missing_positions(self):
        self.assertEqual(abstract({"world": [1], "Hello": [0]}), "Hello world")
        self.assertIsNone(abstract({"incomplete": [1]}))
        self.assertIsNone(abstract({"a": [0], "b": [0]}))

    def test_growth_excludes_gap_and_future_outside_window(self):
        counts = {2008: 1, 2009: 1, 2010: 1, 2011: 900, 2012: 900,
                  2013: 4, 2014: 4, 2015: 4, 2016: 900}
        self.assertAlmostEqual(growth(counts, 2010, 1e-8), 4)

    def test_parquet_roundtrip_and_no_future_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"data_dir": directory, "start_year": 2010, "end_year": 2026,
                      "training_end_year": 2020, "batch_size": 50000, "total": 1000000}
            work = {"id": "https://openalex.org/W1", "publication_year": 2021,
                    "abstract_inverted_index": {"hello": [0]},
                    "referenced_works": ["https://openalex.org/W2"],
                    "primary_topic": {"id": "https://openalex.org/T1",
                                      "domain": {"id": "https://openalex.org/domains/3"}},
                    "authorships": []}
            record = normalize(work, config, "2021-3")
            self.assertIsNone(record["target_emergence"])
            self.assertEqual(record["split"], "inference")
            self.assertIsNone(record["commercial_maturity_index"])
            import json
            connection = database(config)
            with connection:
                connection.execute("INSERT INTO documents VALUES (?,?,?)", ("W1", "2021-3", json.dumps(record)))
                connection.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?)", ("W1", "2021-3", json.dumps(record)))
            connection.close()
            path = export(config)
            self.assertEqual(path.name, "openalex_corpus_partial.parquet")
            result = validate(path, expected=1)
            self.assertTrue(result["passed"])
            self.assertEqual(result["rows"], 1)
            self.assertTrue((Path(directory) / "batches" / "part-00000.parquet").exists())


if __name__ == "__main__":
    unittest.main()
