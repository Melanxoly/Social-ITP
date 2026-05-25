from __future__ import annotations

from social_itp.schemas.types import Action, Observation, SimulatedReply, WorldModelPrediction


class DummyWorldModel:
    """Cheap deterministic world model for smoke tests and interface tests."""

    def __init__(self):
        self.last_raw_response = None

    def predict(self, obs: Observation, action: Action) -> WorldModelPrediction:
        target = obs.target_user
        reply = SimulatedReply(
            node_id="dummy_r1",
            parent_id=action.action_comment_node_id,
            user_id=target.user_id if target else None,
            user_name=target.user_name if target else None,
            role="target_user",
            text="I see your point, but I am not fully convinced yet.",
            depth_from_action=1,
        )
        self.last_raw_response = '{"next_replies":[{"role":"target_user","text":"I see your point, but I am not fully convinced yet."}]}'
        return WorldModelPrediction(next_replies=[reply], meta={"world_model": "dummy"})
