from __future__ import annotations

from typing import Protocol

from social_itp.schemas.types import PolicyDecision, Track1Example


class Policy(Protocol):
    name: str
    def choose(self, example: Track1Example) -> PolicyDecision: ...
