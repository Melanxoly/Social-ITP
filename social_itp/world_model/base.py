from __future__ import annotations

from typing import Protocol

from social_itp.schemas.types import Action, Observation, WorldModelPrediction


class WorldModel(Protocol):
    def predict(self, obs: Observation, action: Action) -> WorldModelPrediction: ...
