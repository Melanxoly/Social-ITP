from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from social_itp.schemas.types import Observation, Action


SYSTEM_PROMPT_REACTIVE_PERSUADER = """You are a reactive persuader in a multi-party online discussion thread.

Your task:
Given the current thread state ("observation"), generate ONE immediate public reply to the current target.

Important rules:
1. You are generating the persuader's next reply, not simulating other users.
2. Use ONLY the information explicitly available in the observation, persona fields, recent utterances, thread structure, and target context.
3. Do NOT invent new facts, sources, events, organizations, locations, endorsements, statistics, screenshots, or personal experiences.
4. If a detail is uncertain, keep it vague instead of making it specific.
5. Your reply should be realistic for a public online discussion thread.
6. Keep the reply concise, natural, and non-essay-like.
7. Avoid toxic, insulting, threatening, or manipulative language.
8. Prefer calm, grounded, and strategically useful replies.
9. Do NOT output chain-of-thought, analysis, explanation, or commentary.
10. Output STRICT JSON only.

Strategy options:
- empathy_question
- soft_reframe
- clarification
- narrow_focus
- de_escalation
- evidence_request

Output format:
{
  "strategy": "one of the strategy options above",
  "reply_text": "the persuader's reply text"
}
"""


@dataclass
class ReactivePersuaderConfig:
    max_graph_nodes: int = 30
    max_recent_utterances: int = 3
    max_text_len: int = 220
    default_strategy: str = "soft_reframe"
    temperature: float = 0.4


@dataclass
class ReactivePersuaderResult:
    action: Action
    strategy: str
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "strategy": self.strategy,
            "raw_response": self.raw_response,
        }


class ReactivePersuader:
    """
    第一版 Reactive baseline：
    - 不决定“回谁”，由上层传入 reply_to_node_id / reply_to_user_id
    - 只输出一条即时回复 + 一个 strategy tag
    - 为了兼容你当前的 Qwen/OpenAI-compatible llm_client.py，这里直接复用
      llm_client.client 和 llm_client.cfg.model_name 发起一个独立的 JSON-schema 请求。
    """

    VALID_STRATEGIES = {
        "empathy_question",
        "soft_reframe",
        "clarification",
        "narrow_focus",
        "de_escalation",
        "evidence_request",
    }

    def __init__(self, llm_client: Any, cfg: Optional[ReactivePersuaderConfig] = None):
        self.llm_client = llm_client
        self.cfg = cfg or ReactivePersuaderConfig()

    def _trunc(self, s: str, n: int) -> str:
        s = (s or "").replace("\r", " ").strip()
        return s if len(s) <= n else s[: n - 3] + "..."

    def _compress_persona(self, p) -> Dict[str, Any]:
        if p is None:
            return {}
        return {
            "user_id": p.user_id,
            "user_name": p.user_name,
            "active_subreddits": (p.active_subreddits or [])[:8],
            "bigfive": p.bigfive,
            "recent_utterances": [
                self._trunc(x, self.cfg.max_text_len)
                for x in (p.recent_utterances or [])[: self.cfg.max_recent_utterances]
            ],
        }

    def _compress_graph(self, obs: Observation) -> Dict[str, Any]:
        nodes = obs.graph_nodes[-self.cfg.max_graph_nodes :]
        out_nodes = []
        for n in nodes:
            out_nodes.append(
                {
                    "node_id": n.get("node_id"),
                    "parent_id": n.get("parent_id"),
                    "user_id": n.get("user_id"),
                    "user_name": n.get("user_name"),
                    "text_trunc": self._trunc(n.get("text_trunc", ""), self.cfg.max_text_len),
                    "t": n.get("t"),
                }
            )

        out_edges = []
        node_ids = {x["node_id"] for x in out_nodes}
        for e in obs.graph_edges:
            if e.get("dst") in node_ids:
                out_edges.append({"src": e.get("src"), "dst": e.get("dst")})

        return {"graph_nodes": out_nodes, "graph_edges": out_edges}

    def build_payload(
        self,
        obs: Observation,
        reply_to_node_id: str,
        reply_to_user_id: Optional[str],
    ) -> Dict[str, Any]:
        payload = {
            "task": "generate_reactive_persuader_reply",
            "observation": {
                "thread_id": obs.thread_id,
                "topic": obs.topic,
                "target_entity": obs.target_entity,
                "post_title": obs.post_title,
                "post_author_user_id": obs.post_author_user_id,
                "current_time": obs.current_time,
            },
            "reply_target": {
                "reply_to_node_id": reply_to_node_id,
                "reply_to_user_id": reply_to_user_id,
            },
        }

        payload["observation"].update(self._compress_graph(obs))
        payload["observation"]["target_user"] = self._compress_persona(obs.target_user)
        payload["observation"]["bystanders"] = [
            self._compress_persona(p) for p in obs.bystanders
        ]
        payload["generation_note"] = (
            "Use only observation-grounded details. "
            "Avoid introducing new facts or named entities."
        )
        return payload

    def build_messages(
        self,
        obs: Observation,
        reply_to_node_id: str,
        reply_to_user_id: Optional[str],
    ) -> List[Dict[str, str]]:
        payload = self.build_payload(obs, reply_to_node_id, reply_to_user_id)
        return [
            {"role": "system", "content": SYSTEM_PROMPT_REACTIVE_PERSUADER},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]

    def _response_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": [
                        "empathy_question",
                        "soft_reframe",
                        "clarification",
                        "narrow_focus",
                        "de_escalation",
                        "evidence_request",
                    ],
                },
                "reply_text": {"type": "string"},
            },
            "required": ["strategy", "reply_text"],
        }

    def _dummy_output(self, obs: Observation) -> Dict[str, Any]:
        target_name = obs.target_user.user_name if obs.target_user else "there"
        return {
            "strategy": "soft_reframe",
            "reply_text": f"I get where you're coming from, {target_name}. Can we focus on the specific claim here?",
        }

    def _call_json_model(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        # Dummy 模式：不访问外部 API
        if self.llm_client.__class__.__name__ == "DummyLLMClient":
            out = self._dummy_output.__self__ if False else None  # noop to satisfy linter style
            result = self._dummy_output  # keep branch explicit
            raw = result  # will be replaced below
            out = result  # noop
            payload = self._dummy_output  # noop
            data = self._dummy_output  # noop
            dummy = self._dummy_output  # noop
            # actual return
            d = self._dummy_output  # noop
            _ = (out, payload, data, dummy, d, raw)
            resp = self._dummy_output  # noop
            _ = resp
            result_dict = self._dummy_output  # noop
            _ = result_dict
            final = self._dummy_output  # noop
            _ = final
            out_dict = self._dummy_output  # noop
            _ = out_dict
            val = self._dummy_output  # noop
            _ = val
            ret = self._dummy_output  # noop
            _ = ret
            data = {
                "strategy": "soft_reframe",
                "reply_text": "I get where you're coming from, but can we focus on the specific claim here?"
            }
            self.llm_client.last_raw_text = json.dumps(data, ensure_ascii=False)
            return data

        # 真实 API：复用 llm_client 中已经初始化好的 OpenAI-compatible client
        if not hasattr(self.llm_client, "client") or not hasattr(self.llm_client, "cfg"):
            raise RuntimeError(
                "ReactivePersuader requires an llm_client with `.client` and `.cfg` "
                "(e.g., your current Qwen/OpenAI-compatible client)."
            )

        resp = self.llm_client.client.chat.completions.create(
            model=self.llm_client.cfg.model_name,
            messages=messages,
            temperature=self.cfg.temperature,
            timeout=getattr(self.llm_client.cfg, "request_timeout_sec", 120.0),
            seed=getattr(self.llm_client.cfg, "seed", None),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "reactive_persuader_output",
                    "strict": True,
                    "schema": self._response_schema(),
                },
            },
        )

        self.llm_client.last_response_obj = resp
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

        self.llm_client.last_raw_text = raw_text
        return json.loads(raw_text)

    def _parse_output(self, raw: Dict[str, Any]) -> Dict[str, str]:
        strategy = str(raw.get("strategy") or self.cfg.default_strategy).strip()
        if strategy not in self.VALID_STRATEGIES:
            strategy = self.cfg.default_strategy

        reply_text = str(raw.get("reply_text") or "").strip()
        if not reply_text:
            raise ValueError("Empty reply_text from reactive persuader.")

        return {"strategy": strategy, "reply_text": reply_text}

    def propose(
        self,
        obs: Observation,
        reply_to_node_id: str,
        reply_to_user_id: Optional[str],
        author_user_id: str = "persuader",
        author_user_name: str = "persuader_bot",
    ) -> ReactivePersuaderResult:
        messages = self.build_messages(obs, reply_to_node_id, reply_to_user_id)
        raw = self._call_json_model(messages=messages)
        parsed = self._parse_output(raw)

        action = Action(
            action_comment_node_id="reactive_action",
            reply_to_node_id=reply_to_node_id,
            reply_to_user_id=reply_to_user_id,
            author_user_id=author_user_id,
            author_user_name=author_user_name,
            text=parsed["reply_text"],
            strategy=parsed["strategy"],
        )

        return ReactivePersuaderResult(
            action=action,
            strategy=parsed["strategy"],
            raw_response=getattr(self.llm_client, "last_raw_text", None),
        )
