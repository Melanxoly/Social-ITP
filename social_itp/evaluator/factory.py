from __future__ import annotations

from typing import Any, Dict

from social_itp.evaluator.dummy_evaluator import DummyEvaluator
from social_itp.evaluator.rule_based_evaluator import RuleBasedEvaluator


def make_evaluator(cfg: Dict[str, Any]):
    ev_type = str(cfg.get("type", "rule_based")).lower()
    if ev_type in {"dummy", "none"}:
        return DummyEvaluator()
    if ev_type in {"rule", "rule_based", "rulebased"}:
        return RuleBasedEvaluator()
    raise ValueError(f"Unknown evaluator type: {ev_type}")
