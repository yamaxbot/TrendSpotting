"""Single-worker background jobs for the local Django server.

Job state is process-local; restarting the server interrupts unfinished work.
Completed results remain in parquet files.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trend-search")
_lock = Lock()
_jobs = {}


def _run(query):
    from on_demand_parsing.time_budget import SearchBudget
    from website.main import (
        build_dataset_for_query,
        get_built_dataset_path,
        get_parsed_corpus_path,
        model_dataset_is_ready,
    )
    from .views import _dataset_is_ready, _load_trends

    with _lock:
        _jobs[query] = "running"
    budget = SearchBudget.start()
    try:
        corpus_path = get_parsed_corpus_path(query)
        path = get_built_dataset_path(query)
        if not model_dataset_is_ready(path, corpus_path):
            path = build_dataset_for_query(query, budget=budget)
        _load_trends(path, corpus_path, generate_llm_texts=True, budget=budget)
        if not _dataset_is_ready(path, corpus_path):
            raise RuntimeError("Search results remain incomplete after enrichment")
    except Exception:
        logger.exception("Search processing failed")
        state = "failed"
    else:
        state = "complete"
    with _lock:
        _jobs[query] = state


def start(query):
    with _lock:
        if _jobs.get(query) in {"queued", "running"}:
            return _jobs[query]
        if sum(s in {"queued", "running"} for s in _jobs.values()) >= 10:
            return "busy"
        for key in list(_jobs):
            if _jobs[key] not in {"queued", "running"}:
                del _jobs[key]
        _jobs[query] = "queued"
        _executor.submit(_run, query)
        return "queued"


def status(query):
    with _lock:
        return _jobs.get(query, "missing")
