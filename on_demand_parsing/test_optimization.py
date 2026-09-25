import unittest
from unittest.mock import patch

from on_demand_parsing import main_collection, openalex_client, parser, query_recovery
from on_demand_parsing.relevance_filter import filter_relevant


class CollectionOptimizationTests(unittest.TestCase):
    def test_recovery_keeps_only_original_query_matches(self):
        def work(identifier, title):
            return {
                "id": identifier,
                "title": title,
                "abstract_inverted_index": {"implant": [0], "sensor": [1]},
            }

        calls = []

        def search(query, *, limit, years, require_references):
            calls.append((query, require_references))
            if query == "implant glucose sensor":
                return [work("W1", "Glucose monitoring"), {
                    "id": "W2", "title": "Unrelated optics",
                    "abstract_inverted_index": {"laser": [0]},
                }]
            return [work("W1", "Glucose monitoring")]

        recovered = query_recovery.recover_seed_works(
            "implant glucose sensor", (2025, 2026), search,
        )
        self.assertEqual([item["id"] for item in recovered], ["W1"])
        self.assertEqual(calls[0], ("implant glucose sensor", False))
        self.assertIn(("implant glucose", True), calls)

    def test_single_relevant_work_is_not_discarded(self):
        only_work = [{"id": "W1", "relevance": .7}]
        self.assertEqual(filter_relevant(only_work), only_work)

    def test_parser_expands_recovered_seed_and_filters_weaker_topic_works(self):
        seed = {"id": "W1"}
        weaker = {"id": "W2"}
        normalized = [
            {"id": "W1", "abstract": "implant glucose sensor", "relevance": .7},
            {"id": "W2", "abstract": "general sensors", "relevance": .2},
        ]
        with patch.object(parser, "normalize_query", return_value="implant glucose sensor"), \
             patch.object(parser, "search_works", return_value=[]), \
             patch.object(parser, "recover_seed_works", return_value=[seed]) as recover, \
             patch.object(parser, "discover_topics", return_value=[{"id": "T1"}]), \
             patch.object(parser, "score_topics", return_value=[{"id": "T1"}]), \
             patch.object(parser, "select_topics", return_value=[{"id": "T1", "name": "topic", "similarity": .8, "frequency": 1, "coverage": 1}]), \
             patch.object(parser, "collect_main_corpus", return_value=([seed, weaker], {})) as collect, \
             patch.object(parser, "normalize_works", return_value=normalized), \
             patch.object(parser, "score_relevance", return_value=normalized), \
             patch.object(parser, "save_results"), \
             patch.object(parser, "save_parquet"):
            works, _ = parser.run_parser("узкий запрос")

        recover.assert_called_once()
        self.assertEqual(collect.call_args.kwargs["seed_works"], [seed])
        self.assertEqual([work["id"] for work in works], ["W1"])

    def test_reuses_seed_and_preserves_topic_year_order(self):
        topics = [
            {"id": "T1", "name": "First"},
            {"id": "T2", "name": "Second"},
        ]
        seed = [{"id": "W-seed"}]

        def fetch(topic_id, limit, year):
            return [{"id": f"W-{topic_id}-{year}"}], year

        with patch.object(main_collection, "search_works_by_topic_with_count", side_effect=fetch) as fetch_mock:
            with patch.object(main_collection, "search_works") as duplicate_search:
                works, stats = main_collection.collect_main_corpus(
                    topics, "query", (2024, 2025), seed_works=seed,
                )

        self.assertEqual(fetch_mock.call_count, 4)
        duplicate_search.assert_not_called()
        self.assertEqual(
            [work["id"] for work in works],
            ["W-T1-2024", "W-T1-2025", "W-T2-2024", "W-T2-2025", "W-seed"],
        )
        self.assertEqual(stats, {
            "First": {2024: 2024, 2025: 2025},
            "Second": {2024: 2024, 2025: 2025},
        })

    def test_topic_count_comes_from_collection_response(self):
        class Response:
            def __init__(self, results, count, cursor):
                self.payload = {
                    "results": results,
                    "meta": {"count": count, "next_cursor": cursor},
                }

            def raise_for_status(self):
                pass

            def json(self):
                return self.payload

        class Session:
            def __init__(self):
                self.responses = iter([
                    Response([{"id": "W1"}], 7, "next"),
                    Response([{"id": "W2"}], 7, None),
                ])
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def get(self, url, params, timeout):
                self.calls.append(params)
                return next(self.responses)

        session = Session()
        with patch.object(openalex_client.requests, "Session", return_value=session):
            works, count = openalex_client.search_works_by_topic_with_count(
                "https://openalex.org/T1", limit=2, year=2025,
            )

        self.assertEqual([work["id"] for work in works], ["W1", "W2"])
        self.assertEqual(count, 7)
        self.assertEqual([call["cursor"] for call in session.calls], ["*", "next"])
        self.assertIn("topics.id:T1", session.calls[0]["filter"])


if __name__ == "__main__":
    unittest.main()
