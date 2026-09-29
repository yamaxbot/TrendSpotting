from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def configuration(path: str = "config_new.json") -> dict:
    config = json.loads((ROOT / path).read_text(encoding="utf-8-sig"))
    config["data_dir"] = str((ROOT / config["data_dir"]).resolve())
    Path(config["data_dir"]).mkdir(parents=True, exist_ok=True)
    return config


def api_key() -> str | None:
    value = os.environ.get("OPENALEX_API_KEY")
    if value:
        return value
    for path in (ROOT / ".env", ROOT.parent / ".env", ROOT.parent.parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.strip().partition("=")
            if separator and name.strip() == "OPENALEX_API_KEY":
                return value.strip().strip('"').strip("'") or None
    return None


class Paused(RuntimeError):
    pass


class Client:

    def __init__(self, rate: float = 5, max_requests: int = 9900):
        self.key = api_key()
        self.rate = rate
        self.max_requests = max_requests
        self.calls = 0
        self.remaining = None
        self.next_request = 0.0
        self.lock = threading.Lock()
        self.local = threading.local()

    def get(self, endpoint: str, **params) -> dict:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            if self.key:
                self.local.session.headers["Authorization"] = f"Bearer {self.key}"
        for attempt in range(6):
            with self.lock:
                if self.calls >= self.max_requests or (
                    self.remaining is not None and self.remaining <= 10
                ):
                    raise Paused("Request allowance reached; progress is saved.")
                time.sleep(max(0, self.next_request - time.monotonic()))
                self.next_request = time.monotonic() + 1 / self.rate
                self.calls += 1
            try:
                response = self.local.session.get(
                    f"https://api.openalex.org/{endpoint}", params=params, timeout=(15, 90)
                )
            except requests.RequestException:
                if attempt == 5:
                    raise Paused("Network unavailable; progress is saved.") from None
                time.sleep(min(2 ** attempt, 30))
                continue
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining is not None:
                with self.lock:
                    number = int(float(remaining))
                    self.remaining = number if self.remaining is None else min(self.remaining, number)
            if response.status_code == 429:
                raise Paused("OpenAlex rate/budget limit reached; retry later.")
            if response.status_code in (401, 403):
                raise Paused("OpenAlex rejected the API key or permissions.")
            if response.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue
            if response.status_code != 200:
                raise RuntimeError(f"OpenAlex {endpoint}: HTTP {response.status_code}")
            return response.json()
        raise Paused("OpenAlex unavailable after retries.")


def database(config: dict) -> sqlite3.Connection:
    Path(config["data_dir"]).mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(Path(config["data_dir"]) / "state.sqlite", timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS strata (
            name TEXT PRIMARY KEY, filter TEXT NOT NULL, quota INTEGER NOT NULL,
            population INTEGER NOT NULL, seed INTEGER NOT NULL, page INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY, stratum TEXT NOT NULL, payload TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS documents_stratum ON documents(stratum);
        CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS topic_authors (
            topic TEXT NOT NULL, author TEXT NOT NULL, first_year INTEGER NOT NULL,
            PRIMARY KEY(topic, author)
        );
        CREATE TABLE IF NOT EXISTS topic_cursor (
            topic TEXT PRIMARY KEY, cursor TEXT, complete INTEGER NOT NULL DEFAULT 0
        );
    """)
    return connection


def cached_get(connection, client, key: str, endpoint: str, **params) -> dict:
    row = connection.execute("SELECT payload FROM cache WHERE key=?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    result = client.get(endpoint, **params)
    with connection:
        connection.execute("INSERT INTO cache VALUES (?,?)", (key, json.dumps(result)))
    return result


def short_id(value: str | None) -> str | None:
    return value.rsplit("/", 1)[-1] if value else None
