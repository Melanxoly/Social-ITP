from __future__ import annotations
from typing import List

from social_itp.schemas.types import Observation, Action
from social_itp.schemas.types import WorldModelPrediction as SimulatorPrediction, EvalResult as EvalScores


class RuleBasedEvaluator:
    """
    第一版先用启发式规则，后面再替换成：
    - stance classifier
    - toxicity/safety classifier
    - affect / defensiveness classifier
    """

    def score(
        self, obs: Observation, action: Action, pred: SimulatorPrediction
    ) -> EvalScores:
        target_replies = [r for r in pred.next_replies if r.role == "target_user"]
        bystander_replies = [r for r in pred.next_replies if r.role == "bystander"]

        target_engagement = float(len(target_replies))
        bystander_engagement = float(len(bystander_replies))

        # 一个极简 proxy：出现明显敌对词就降低支持性
        hostile_terms = ["idiot", "stupid", "moron", "shut up", "fuck", "bullshit"]
        hostile_count = 0
        for r in pred.next_replies:
            txt = r.text.lower()
            if any(w in txt for w in hostile_terms):
                hostile_count += 1

        target_supportiveness = max(0.0, 1.0 - 0.5 * hostile_count)
        bystander_polarization_risk = min(1.0, 0.3 * hostile_count)
        safety_risk = min(1.0, 0.3 * hostile_count)

        overall_score = (
            1.2 * target_engagement
            + 0.4 * target_supportiveness
            - 0.8 * bystander_polarization_risk
            - 1.0 * safety_risk
        )

        return EvalScores(
            target_engagement=target_engagement,
            bystander_engagement=bystander_engagement,
            target_supportiveness=target_supportiveness,
            bystander_polarization_risk=bystander_polarization_risk,
            safety_risk=safety_risk,
            overall_score=overall_score,
            notes="rule-based baseline",
        )
