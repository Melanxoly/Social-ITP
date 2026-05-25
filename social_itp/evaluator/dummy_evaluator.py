from __future__ import annotations

from social_itp.schemas.types import Action, EvalResult, Observation, WorldModelPrediction


class DummyEvaluator:
    """Evaluator for smoke tests. It is not a research metric."""

    def score(self, obs: Observation, action: Action, prediction: WorldModelPrediction) -> EvalResult:
        return EvalResult(
            target_engagement=0.0,
            bystander_engagement=0.0,
            target_supportiveness=0.5,
            bystander_polarization_risk=0.0,
            safety_risk=0.0,
            overall_score=0.5,
            target_effect=0.5,
            bystander_externality=0.0,
            cost=0.0,
            notes="dummy evaluator",
        )
