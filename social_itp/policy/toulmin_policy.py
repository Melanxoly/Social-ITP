from __future__ import annotations

from social_itp.persuader.dummy_toulmin_persuader import DummyToulminPersuader
from social_itp.schemas.types import PolicyDecision, Track1Example


class DummyToulminPolicy:
    name = "dummy_toulmin"

    def __init__(self):
        self.persuader = DummyToulminPersuader()

    def choose(self, example: Track1Example) -> PolicyDecision:
        action = self.persuader.propose(
            obs=example.observation,
            reply_to_node_id=example.action.reply_to_node_id,
            reply_to_user_id=example.action.reply_to_user_id,
        )
        return PolicyDecision(
            policy_name=self.name,
            action=action,
            strategy=action.strategy,
            meta={"source": "dummy Toulmin persuader", "toulmin": action.toulmin},
        )
