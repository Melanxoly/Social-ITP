from __future__ import annotations

import json
from typing import Any, Dict, Optional

from social_itp.schemas.types import Observation, Action
from social_itp.schemas.types import SimulatedReply, WorldModelPrediction as SimulatorPrediction
from social_itp.world_model.prompt_builder import PromptBuilder
from social_itp.llm.client import LLMClient


class PromptWorldSimulator:
    """
    Prompt-based world model:
    输入 observation + action，输出结构化 next_replies。
    关键修复：
    1. parent_id 不再允许保留占位符 TO_FILL / PRED_ACTION；
       若模型未给出合法父节点，则默认挂到当前 action.action_comment_node_id。
    2. 每次 predict 后，保留该次调用对应的 raw response，供 look-ahead 调试。
    """

    VALID_ROLES = {"target_user", "bystander", "action_author", "other"}
    INVALID_PARENT_IDS = {
        None,
        "",
        "TO_FILL",
        "to_fill",
        "PENDING",
        "PRED_ACTION",
        "pred_action",
    }

    def __init__(self, llm_client: LLMClient, prompt_builder: PromptBuilder):
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.last_raw_response: Optional[str] = None
        self.last_response_obj: Optional[Any] = None

    def _validate_prediction_dict(self, raw: Any) -> bool:
        if not isinstance(raw, dict):
            return False
        if "next_replies" not in raw:
            return False
        if not isinstance(raw["next_replies"], list):
            return False
        return True

    def _parse_prediction(
        self, raw: Dict[str, Any], action: Action
    ) -> SimulatorPrediction:
        replies = []

        for i, r in enumerate(raw.get("next_replies", []), start=1):
            raw_parent_id = r.get("parent_id")
            raw_node_id = r.get("node_id")
            raw_role = r.get("role")
            raw_text = str(r.get("text") or "").strip()

            if not raw_text:
                continue

            node_id = str(raw_node_id or f"pred_r{i}")

            # 修复点 1：不要再把占位符保留到输出里
            if raw_parent_id in self.INVALID_PARENT_IDS:
                parent_id = str(action.action_comment_node_id)
            else:
                parent_id = str(raw_parent_id)

            role = str(raw_role or "other").strip()
            if role not in self.VALID_ROLES:
                role = "other"

            replies.append(
                SimulatedReply(
                    node_id=node_id,
                    parent_id=parent_id,
                    user_id=r.get("user_id"),
                    user_name=r.get("user_name"),
                    role=role,
                    text=raw_text,
                    t=r.get("t"),
                    depth_from_action=int(r.get("depth_from_action") or 1),
                )
            )

        return SimulatorPrediction(next_replies=replies)

    def predict(self, obs: Observation, action: Action) -> SimulatorPrediction:
        messages = self.prompt_builder.build_messages(obs, action)
        raw = self.llm_client.generate_json(messages=messages, temperature=0.7)

        # 修复点 2：保留本次 candidate 调用对应的 raw response
        self.last_response_obj = raw
        raw_text = getattr(self.llm_client, "last_raw_text", None)
        if raw_text is None:
            try:
                raw_text = json.dumps(raw, ensure_ascii=False)
            except Exception:
                raw_text = str(raw)
        self.last_raw_response = raw_text

        if not self._validate_prediction_dict(raw):
            return SimulatorPrediction(next_replies=[])

        return self._parse_prediction(raw, action)
