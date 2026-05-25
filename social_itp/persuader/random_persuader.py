from __future__ import annotations

import random
from typing import Dict, List, Optional

from social_itp.schemas.types import Action, Observation


RANDOM_TEMPLATE_ACTIONS: List[Dict[str, str]] = [
    {"strategy": "empathy_question", "text": "I get where you're coming from. What makes you see it that way?"},
    {"strategy": "soft_reframe", "text": "There may be another way to look at this. Would you be open to one point?"},
    {"strategy": "clarification", "text": "Can you clarify which specific claim you mean here?"},
    {"strategy": "narrow_focus", "text": "Can we focus on the specific claim here instead of the broader identity issue?"},
    {"strategy": "de_escalation", "text": "I think this is getting heated. Can we slow down and focus on what is actually shown?"},
    {"strategy": "evidence_request", "text": "What evidence in the post makes you most confident about that conclusion?"},
]


class RandomPersuader:
    def __init__(self, seed: int = 42, templates: Optional[List[Dict[str, str]]] = None):
        self.rng = random.Random(seed)
        self.templates = templates or RANDOM_TEMPLATE_ACTIONS

    def propose(
        self,
        obs: Observation,
        reply_to_node_id: str,
        reply_to_user_id: Optional[str] = None,
        author_user_id: str = "persuader",
        author_user_name: str = "persuader_bot",
    ) -> Action:
        cand = self.rng.choice(self.templates)
        return Action(
            action_comment_node_id="random_action",
            reply_to_node_id=reply_to_node_id,
            reply_to_user_id=reply_to_user_id,
            author_user_id=author_user_id,
            author_user_name=author_user_name,
            text=cand["text"],
            strategy=cand["strategy"],
        )
