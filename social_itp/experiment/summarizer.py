from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_experiment(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    result_path = output_dir / "raw" / "results.jsonl"
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not result_path.exists():
        raise FileNotFoundError(f"results.jsonl not found: {result_path}")

    records = _read_jsonl(result_path)
    flat = []
    for r in records:
        ev = r.get("evaluation", {}) or {}
        dec = r.get("decision", {}) or {}
        action = dec.get("action", {}) or {}
        pred = r.get("world_model_prediction", {}) or {}
        replies = pred.get("next_replies", []) or []
        timing = r.get("timing", {}) or {}
        flat.append({
            "exp_id": r.get("exp_id"),
            "topic": r.get("topic"),
            "example_id": r.get("example_id"),
            "policy": r.get("policy"),
            "strategy": dec.get("strategy") or action.get("strategy"),
            "action_text": action.get("text"),
            "pred_reply_count": len(replies),
            "pred_target_reply_count": sum(1 for x in replies if x.get("role") == "target_user"),
            "pred_bystander_reply_count": sum(1 for x in replies if x.get("role") == "bystander"),
            "target_engagement": ev.get("target_engagement"),
            "bystander_engagement": ev.get("bystander_engagement"),
            "target_supportiveness": ev.get("target_supportiveness"),
            "bystander_polarization_risk": ev.get("bystander_polarization_risk"),
            "safety_risk": ev.get("safety_risk"),
            "overall_score": ev.get("overall_score"),
            "target_effect": ev.get("target_effect"),
            "bystander_externality": ev.get("bystander_externality"),
            "cost": ev.get("cost"),
            "decision_sec": timing.get("decision_sec"),
            "world_model_sec": timing.get("world_model_sec"),
            "evaluator_sec": timing.get("evaluator_sec"),
            "total_sec": timing.get("total_sec"),
        })

    df = pd.DataFrame(flat)
    df.to_csv(summary_dir / "all_rows.csv", index=False)

    numeric_cols = [
        "pred_reply_count",
        "pred_target_reply_count",
        "pred_bystander_reply_count",
        "target_engagement",
        "bystander_engagement",
        "target_supportiveness",
        "bystander_polarization_risk",
        "safety_risk",
        "overall_score",
        "target_effect",
        "bystander_externality",
        "cost",
        "decision_sec",
        "world_model_sec",
        "evaluator_sec",
        "total_sec",
    ]
    existing_numeric = [c for c in numeric_cols if c in df.columns]
    if len(df):
        df.groupby("policy")[existing_numeric].agg(["mean", "std", "count"]).reset_index().to_csv(
            summary_dir / "summary_by_policy.csv", index=False
        )
        df.groupby(["topic", "policy"])[existing_numeric].agg(["mean", "std", "count"]).reset_index().to_csv(
            summary_dir / "summary_by_topic_policy.csv", index=False
        )
        df.groupby(["policy", "strategy"]).size().reset_index(name="count").sort_values(
            ["policy", "count"], ascending=[True, False]
        ).to_csv(summary_dir / "strategy_distribution.csv", index=False)

    print(f"[OK] wrote summary to: {summary_dir}")
    return summary_dir
