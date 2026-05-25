from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


class LLMClient(Protocol):
    def generate_json(
        self, messages: List[Dict[str, str]], temperature: float = 0.7
    ) -> Dict[str, Any]: ...


class DummyLLMClient:
    """本地联调用占位客户端，不访问任何外部 API。"""

    def __init__(self):
        self.last_raw_text: Optional[str] = None
        self.last_response_obj: Optional[Any] = None

    def generate_json(
        self, messages: List[Dict[str, str]], temperature: float = 0.7
    ) -> Dict[str, Any]:
        out = {
            "next_replies": [
                {
                    "node_id": "pred_r1",
                    "parent_id": "TO_FILL",
                    "user_id": "dummy_target",
                    "user_name": "dummy_target_user",
                    "role": "target_user",
                    "text": "I disagree with that, but I see what you're saying.",
                    "depth_from_action": 1,
                    "t": None,
                }
            ]
        }
        self.last_raw_text = json.dumps(out, ensure_ascii=False)
        self.last_response_obj = out
        return out


@dataclass
class QwenClientConfig:
    model_name: str = "qwen-plus-latest"
    max_retries: int = 2
    retry_sleep_sec: float = 1.5
    request_timeout_sec: float = 120.0
    seed: Optional[int] = 42
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url_env: str = "DASHSCOPE_BASE_URL"
    default_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenJSONClient:
    """
    千问（阿里云百炼）OpenAI-compatible 客户端。

    设计说明：
    1. 使用 OpenAI Python SDK + DashScope OpenAI-compatible base_url。
    2. 使用 response_format=json_schema，尽量让模型直接输出结构化 JSON。
    3. 保持 generate_json(messages, temperature) 接口不变，便于与现有 simulator 无缝衔接。
    """

    _REPLY_ROLE_SET = {"target_user", "bystander", "action_author", "other"}

    def __init__(self, cfg: Optional[QwenClientConfig] = None):
        self.cfg = cfg or QwenClientConfig()
        self.last_raw_text: Optional[str] = None
        self.last_response_obj: Optional[Any] = None

        api_key = os.getenv(self.cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.cfg.api_key_env} is not set. "
                f"Please set your DashScope API key first."
            )

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "The `openai` Python package is required. Install it with: pip install openai"
            ) from e

        base_url = os.getenv(self.cfg.base_url_env, self.cfg.default_base_url)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _response_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "next_replies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "node_id": {"type": "string"},
                            "parent_id": {"type": ["string", "null"]},
                            "user_id": {"type": ["string", "null"]},
                            "user_name": {"type": ["string", "null"]},
                            "role": {
                                "type": "string",
                                "enum": [
                                    "target_user",
                                    "bystander",
                                    "action_author",
                                    "other",
                                ],
                            },
                            "text": {"type": "string"},
                            "depth_from_action": {"type": "integer"},
                            "t": {"type": ["string", "null"]},
                        },
                        "required": [
                            "node_id",
                            "parent_id",
                            "user_id",
                            "user_name",
                            "role",
                            "text",
                            "depth_from_action",
                            "t",
                        ],
                    },
                }
            },
            "required": ["next_replies"],
        }

    def _validate_prediction(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Model output is not a JSON object.")

        replies = data.get("next_replies")
        if not isinstance(replies, list):
            raise ValueError("`next_replies` must be a list.")

        cleaned: List[Dict[str, Any]] = []
        for i, r in enumerate(replies, start=1):
            if not isinstance(r, dict):
                raise ValueError(f"Reply #{i} is not a JSON object.")

            role = str(r.get("role") or "other").strip()
            if role not in self._REPLY_ROLE_SET:
                role = "other"

            text = str(r.get("text") or "").strip()
            if not text:
                continue

            item = {
                "node_id": str(r.get("node_id") or f"pred_r{i}"),
                "parent_id": (
                    None if r.get("parent_id") is None else str(r.get("parent_id"))
                ),
                "user_id": None if r.get("user_id") is None else str(r.get("user_id")),
                "user_name": (
                    None if r.get("user_name") is None else str(r.get("user_name"))
                ),
                "role": role,
                "text": text,
                "depth_from_action": int(r.get("depth_from_action") or 1),
                "t": None if r.get("t") is None else str(r.get("t")),
            }
            cleaned.append(item)

        return {"next_replies": cleaned}

    def _call_qwen_once(
        self, messages: List[Dict[str, str]], temperature: float
    ) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=temperature,
            timeout=self.cfg.request_timeout_sec,
            seed=self.cfg.seed,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "simulator_prediction",
                    "strict": True,
                    "schema": self._response_schema(),
                },
            },
        )
        self.last_response_obj = resp

        msg = resp.choices[0].message
        content = msg.content
        if isinstance(content, list):
            parts: List[str] = []
            for x in content:
                if isinstance(x, dict) and x.get("type") == "text":
                    parts.append(str(x.get("text") or ""))
                else:
                    parts.append(str(x))
            raw_text = "\n".join(parts)
        else:
            raw_text = str(content or "")

        self.last_raw_text = raw_text
        data = json.loads(raw_text)
        return self._validate_prediction(data)

    def generate_json(
        self, messages: List[Dict[str, str]], temperature: float = 0.7
    ) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 2):
            try:
                return self._call_qwen_once(messages=messages, temperature=temperature)
            except Exception as e:
                last_err = e
                if attempt >= self.cfg.max_retries + 1:
                    break
                time.sleep(self.cfg.retry_sleep_sec)
        raise RuntimeError(f"Qwen generate_json failed after retries: {last_err}")
