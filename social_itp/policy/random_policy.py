from __future__ import annotations

from social_itp.persuader.random_persuader import RandomPersuader
from social_itp.schemas.types import PolicyDecision, Track1Example


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 42):
        self.persuader = RandomPersuader(seed=seed)

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
            meta={"source": "random template"},
        )
