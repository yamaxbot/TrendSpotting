import unittest
from unittest.mock import patch

from on_demand_parsing import openalex_client
from on_demand_parsing.time_budget import SEARCH_LIMIT_SECONDS, SearchBudget


class TimeBudgetTests(unittest.TestCase):
    def test_limit_is_twenty_five_minutes(self):
        self.assertEqual(SEARCH_LIMIT_SECONDS, 1500)
        with patch("on_demand_parsing.time_budget.time.monotonic", side_effect=[10, 11]):
            budget = SearchBudget.start()
            self.assertEqual(budget.remaining(), 1499)

    def test_openalex_returns_loaded_page_when_deadline_expires(self):
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": [{"id": "W1"}],
                        "meta": {"next_cursor": "next"}}

        class Session:
            def __init__(self):
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def get(self, *_args, **_kwargs):
                self.calls += 1
                return Response()

        class Budget:
            def __init__(self):
                self.checks = 0
                self.truncated = False

            def expired(self):
                self.checks += 1
                return self.checks >= 3

            def request_timeout(self):
                return 1

            def stop(self):
                self.truncated = True

        session = Session()
        budget = Budget()
        with patch.object(openalex_client.requests, "Session", return_value=session):
            works = openalex_client.search_works("query", years=[2025], budget=budget)
        self.assertEqual([work["id"] for work in works], ["W1"])
        self.assertEqual(session.calls, 1)
        self.assertTrue(budget.truncated)


if __name__ == "__main__":
    unittest.main()
