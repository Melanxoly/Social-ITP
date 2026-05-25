from __future__ import annotations
from typing import List

from social_itp.schemas.types import Observation
from social_itp.schemas.types import CandidateAction


class TemplateCandidateGenerator:
    """
    第一版别急着让模型自由发挥太多。
    先给固定模板 + 少量风格变化，便于比较 look-ahead 是否有用。
    """

    def generate(self, obs: Observation) -> List[CandidateAction]:
        target_name = obs.target_user.user_name if obs.target_user else "there"

        return [
            CandidateAction(
                text=f"I get where you're coming from, {target_name}. What makes you see it that way?",
                strategy="empathy_question",
            ),
            CandidateAction(
                text="I think there may be another way to look at this. Would you be open to considering one point?",
                strategy="soft_reframe",
            ),
            CandidateAction(
                text="Can we focus on the specific claim here instead of the whole political identity issue?",
                strategy="narrow_focus",
            ),
        ]
