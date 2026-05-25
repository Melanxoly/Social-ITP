from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class Persona:
    user_id: str
    user_name: Optional[str] = None
    karma: Optional[int] = None
    created_utc: Optional[str] = None
    active_subreddits: Optional[List[str]] = None
    bigfive: Optional[Dict[str, Any]] = None
    recent_utterances: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    thread_id: str
    topic: str
    target_entity: Optional[str]
    post_title: str
    post_text: str
    post_url: Optional[str]
    media_links: Optional[Any]
    is_media_post: bool
    post_author_user_id: Optional[str]
    current_time: str
    graph_nodes: List[Dict[str, Any]]
    graph_edges: List[Dict[str, str]]
    target_user: Optional[Persona]
    bystanders: List[Persona]
    # Reserved for later ablations. Current builders may keep it empty.
    state_features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["target_user"] = None if self.target_user is None else self.target_user.to_dict()
        d["bystanders"] = [p.to_dict() for p in self.bystanders]
        return d


@dataclass
class Action:
    action_comment_node_id: str
    reply_to_node_id: str
    reply_to_user_id: Optional[str]
    author_user_id: Optional[str]
    author_user_name: Optional[str]
    text: str
    strategy: Optional[str] = None
    # Reserved for Toulmin-constrained generation.
    toulmin: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LabeledReply:
    node_id: str
    parent_id: str
    user_id: Optional[str]
    user_name: Optional[str]
    t: str
    text: str
    role: str
    is_target_user: bool = False
    is_bystander: bool = False
    is_action_author: bool = False
    depth_from_action: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Track1Label:
    next_replies: List[LabeledReply]

    def to_dict(self) -> Dict[str, Any]:
        return {"next_replies": [r.to_dict() for r in self.next_replies]}


@dataclass
class Track1Example:
    example_id: str
    observation: Observation
    action: Action
    label: Track1Label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
            "label": self.label.to_dict(),
        }


@dataclass
class SimulatedReply:
    node_id: str
    parent_id: str
    user_id: Optional[str]
    user_name: Optional[str]
    role: str  # target_user | bystander | action_author | other
    text: str
    t: Optional[str] = None
    depth_from_action: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorldModelPrediction:
    next_replies: List[SimulatedReply]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_replies": [r.to_dict() for r in self.next_replies],
            "meta": self.meta,
        }


# Backward-compatible alias for old code names.
SimulatorPrediction = WorldModelPrediction


@dataclass
class EvalResult:
    # Old rule-based fields; kept for backward compatibility.
    target_engagement: float = 0.0
    bystander_engagement: float = 0.0
    target_supportiveness: float = 0.0
    bystander_polarization_risk: float = 0.0
    safety_risk: float = 0.0
    overall_score: float = 0.0
    notes: Optional[str] = None
    # New generic fields; later evaluators can populate these.
    target_effect: Optional[float] = None
    bystander_externality: Optional[float] = None
    cost: Optional[float] = None
    subscores: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


EvalScores = EvalResult


@dataclass
class CandidateAction:
    text: str
    strategy: Optional[str] = None

    def to_action(
        self,
        obs: Observation,
        author_user_id: str = "persuader",
        author_user_name: Optional[str] = "persuader_bot",
        reply_to_node_id: Optional[str] = None,
        action_comment_node_id: str = "candidate_action",
    ) -> Action:
        reply_to_user_id = obs.target_user.user_id if obs.target_user else None
        return Action(
            action_comment_node_id=action_comment_node_id,
            reply_to_node_id=reply_to_node_id or "TO_FILL",
            reply_to_user_id=reply_to_user_id,
            author_user_id=author_user_id,
            author_user_name=author_user_name,
            text=self.text,
            strategy=self.strategy,
        )


@dataclass
class SearchResult:
    chosen_action: CandidateAction
    chosen_prediction: WorldModelPrediction
    chosen_scores: EvalResult
    all_candidates: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen_action": asdict(self.chosen_action),
            "chosen_prediction": self.chosen_prediction.to_dict(),
            "chosen_scores": self.chosen_scores.to_dict(),
            "all_candidates": self.all_candidates,
        }


@dataclass
class PolicyDecision:
    policy_name: str
    action: Action
    strategy: Optional[str] = None
    lookahead_trace: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "action": self.action.to_dict(),
            "strategy": self.strategy,
            "lookahead_trace": self.lookahead_trace,
            "raw_response": self.raw_response,
            "meta": self.meta,
        }


class WorldModel(Protocol):
    def predict(self, obs: Observation, action: Action) -> WorldModelPrediction: ...


class Evaluator(Protocol):
    def score(self, obs: Observation, action: Action, prediction: WorldModelPrediction) -> EvalResult: ...


class Policy(Protocol):
    name: str
    def choose(self, example: Track1Example) -> PolicyDecision: ...
