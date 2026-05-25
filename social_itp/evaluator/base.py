from __future__ import annotations

from typing import Protocol

from social_itp.schemas.types import Action, EvalResult, Observation, WorldModelPrediction


class Evaluator(Protocol):
    def score(self, obs: Observation, action: Action, prediction: WorldModelPrediction) -> EvalResult: ...
