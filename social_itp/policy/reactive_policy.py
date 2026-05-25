from __future__ import annotations

from social_itp.persuader.reactive_persuader import ReactivePersuader, ReactivePersuaderConfig
from social_itp.schemas.types import PolicyDecision, Track1Example


class ReactivePolicy:
    name = "reactive"

    def __init__(self, llm_client, temperature: float = 0.4):
        self.persuader = ReactivePersuader(
            llm_client=llm_client,
            cfg=ReactivePersuaderConfig(temperature=temperature),
        )

    def choose(self, example: Track1Example) -> PolicyDecision:
        result = self.persuader.propose(
            obs=example.observation,
            reply_to_node_id=example.action.reply_to_node_id,
            reply_to_user_id=example.action.reply_to_user_id,
            author_user_id="persuader",
            author_user_name="persuader_bot",
        )
        return PolicyDecision(
            policy_name=self.name,
            action=result.action,
            strategy=result.strategy,
            raw_response=result.raw_response,
            meta={"source": "reactive persuader"},
        )
