from __future__ import annotations

from typing import Optional

from social_itp.schemas.types import Action, Observation


class DummyToulminPersuader:
    """Interface placeholder for Toulmin-constrained generation.

    It returns a valid action with a simple Toulmin plan. Replace this class with a
    real LLM-based planner when running the Toulmin ablation.
    """

    def propose(
        self,
        obs: Observation,
        reply_to_node_id: str,
        reply_to_user_id: Optional[str] = None,
        author_user_id: str = "persuader",
        author_user_name: str = "persuader_bot",
    ) -> Action:
        return Action(
            action_comment_node_id="dummy_toulmin_action",
            reply_to_node_id=reply_to_node_id,
            reply_to_user_id=reply_to_user_id,
            author_user_id=author_user_id,
            author_user_name=author_user_name,
            text="I understand the concern. Could we first separate the specific claim from the broader disagreement?",
            strategy="narrow_focus",
            toulmin={
                "claim": "The discussion should first focus on the specific claim.",
                "data": "The current thread appears to mix several issues together.",
                "warrant": "Clarifying the specific claim can reduce misunderstanding and escalation.",
                "qualifier": "This may not resolve the entire disagreement immediately.",
                "rebuttal": "If the broader issue is central, it can be discussed after the claim is clarified.",
            },
        )
