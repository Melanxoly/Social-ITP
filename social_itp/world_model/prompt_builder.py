from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from social_itp.schemas.types import Observation, Action


SYSTEM_PROMPT_SIMULATOR = """You are a simulation model of local social state transitions in a multi-party online discussion thread.

Your task:
Given the current thread state ("observation") and a new persuader message ("action"),
predict the most likely public replies in the next short interaction window.

You are NOT helping the persuader.
You are ONLY simulating how other participants would likely react.

Core requirements:
1. Use ONLY the information explicitly available in the observation, persona fields, recent utterances, thread structure, and the current action.
2. Do NOT invent new facts, sources, events, organizations, locations, endorsements, statistics, reports, or personal experiences unless they are clearly grounded in the observation.
3. If a detail is uncertain, keep it vague. It is better to be underspecified than to fabricate specifics.
4. Do NOT output stance labels, persuasion scores, emotion scores, explanations, reasoning steps, or commentary.
5. Output STRICT JSON only. No markdown, no code fence, no extra text.
6. Replies should sound realistic for a public online discussion thread and should match the likely tone, style, and context of each participant.
7. Not everyone needs to reply. It is valid to return zero replies.
8. Some cases may produce only a target_user reply, only bystander replies, or both.
9. Keep replies concise unless the context strongly suggests a longer response.
10. Prefer plausible conversational reactions over polished essays.
11. Do not force agreement, escalation, or engagement if silence or a short dismissive reply is more likely.
12. Do not introduce new named entities unless they already appear in the observation.
13. Do not cite news outlets, screenshots, relatives, friends, or external evidence unless such evidence is explicitly present in the observation.
14. If the action is ambiguous or weak, a minimal reaction or no reaction is acceptable.

Return valid JSON that can be parsed directly by json.loads().

Output format:
{
  "next_replies": [
    {
      "node_id": "pred_r1",
      "parent_id": "parent comment node id",
      "user_id": "user id",
      "user_name": "user name",
      "role": "target_user|bystander|action_author|other",
      "text": "reply text",
      "depth_from_action": 1
    }
  ]
}
"""


@dataclass
class PromptBuildConfig:
    max_graph_nodes: int = 30
    max_recent_utterances: int = 3
    max_text_len: int = 220
    include_persona: bool = True
    include_bystanders: bool = True
    include_graph_edges: bool = True


class PromptBuilder:
    def __init__(self, cfg: PromptBuildConfig):
        self.cfg = cfg

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
                    "text_trunc": self._trunc(
                        n.get("text_trunc", ""), self.cfg.max_text_len
                    ),
                    "t": n.get("t"),
                }
            )

        out = {"graph_nodes": out_nodes}
        if self.cfg.include_graph_edges:
            node_ids = {x["node_id"] for x in out_nodes}
            edges = []
            for e in obs.graph_edges:
                if e.get("dst") in node_ids:
                    edges.append({"src": e.get("src"), "dst": e.get("dst")})
            out["graph_edges"] = edges
        return out

    def build_user_payload(self, obs: Observation, action: Action) -> Dict[str, Any]:
        payload = {
            "task": "predict_next_replies",
            "time_horizon_hours": 48,
            "max_depth": 1,
            "observation": {
                "thread_id": obs.thread_id,
                "topic": obs.topic,
                "target_entity": obs.target_entity,
                "post_title": obs.post_title,
                "post_author_user_id": obs.post_author_user_id,
                "current_time": obs.current_time,
            },
            "action": {
                "action_comment_node_id": action.action_comment_node_id,
                "reply_to_node_id": action.reply_to_node_id,
                "reply_to_user_id": action.reply_to_user_id,
                "author_user_id": action.author_user_id,
                "author_user_name": action.author_user_name,
                "text": self._trunc(action.text, self.cfg.max_text_len),
                "strategy": action.strategy,
            },
        }

        payload["observation"].update(self._compress_graph(obs))

        if self.cfg.include_persona:
            payload["observation"]["target_user"] = self._compress_persona(
                obs.target_user
            )

        if self.cfg.include_bystanders:
            payload["observation"]["bystanders"] = [
                self._compress_persona(p) for p in obs.bystanders
            ]

        return payload

    def build_messages(self, obs: Observation, action: Action) -> List[Dict[str, str]]:
        payload = self.build_user_payload(obs, action)
        return [
            {"role": "system", "content": SYSTEM_PROMPT_SIMULATOR},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ]
