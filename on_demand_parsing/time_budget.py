
from dataclasses import dataclass
import time


SEARCH_LIMIT_SECONDS = 30 * 60


@dataclass
class SearchBudget:
    deadline: float
    truncated: bool = False

    @classmethod
    def start(cls):
        return cls(time.monotonic() + SEARCH_LIMIT_SECONDS)

    def expired(self):
        return time.monotonic() >= self.deadline

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def request_timeout(self, maximum=30):
        return max(0.1, min(maximum, self.remaining()))

    def stop(self):
        self.truncated = True
