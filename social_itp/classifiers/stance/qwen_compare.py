from __future__ import annotations

import csv
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

STANCE_LABELS = ["favor", "against", "none"]
PRED_LABELS_WITH_INVALID = ["favor", "against", "none", "invalid"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_sample_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    n_per_topic_label: int = 5,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample rows by topic × stance.

    If max_samples is supplied and the stratified sample is larger than max_samples,
    the sample is shuffled and truncated. This keeps the command cheap for API use.
    """
    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        topic = str(r.get("topic") or "")
        label = str(r.get("stance") or "")
        if label not in STANCE_LABELS:
            continue
        groups.setdefault((topic, label), []).append(r)

    sampled: List[Dict[str, Any]] = []
    for key in sorted(groups.keys()):
        group = list(groups[key])
        rng.shuffle(group)
        take = min(n_per_topic_label, len(group))
        sampled.extend(group[:take])

    rng.shuffle(sampled)
    if max_samples is not None and max_samples > 0 and len(sampled) > max_samples:
        sampled = sampled[:max_samples]
    return sampled


def _shorten(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    s = re.sub(r"\s+", " ", str(text)).strip()
    if max_chars and max_chars > 0 and len(s) > max_chars:
        return s[:max_chars].rstrip() + " ..."
    return s


def build_qwen_prompt(row: Dict[str, Any], *, max_context_chars: int = 1200, max_comment_chars: int = 800) -> str:
    """Build a deterministic target-conditioned stance classification prompt."""
    target = _shorten(row.get("target_entity") or "the target entity", 120)
    topic = _shorten(row.get("topic"), 120)
    post_title = _shorten(row.get("post_title"), 240)

    # Prefer explicit ancestor_texts if available; otherwise use parent_text.
    ancestor_texts = row.get("ancestor_texts")
    context_lines: List[str] = []
    if isinstance(ancestor_texts, list) and ancestor_texts:
        for i, t in enumerate(ancestor_texts[-2:], start=1):
            if str(t).strip():
                context_lines.append(f"Ancestor comment {i}: {_shorten(t, max_context_chars // 2)}")
    else:
        parent = _shorten(row.get("parent_text"), max_context_chars)
        if parent:
            context_lines.append(f"Parent comment: {parent}")

    comment = _shorten(row.get("comment_text") or row.get("input_text"), max_comment_chars)
    context = "\n".join(context_lines) if context_lines else "(no parent context provided)"

    return f"""You are a strict stance classifier for online discussion comments.

Task: Classify the stance of the COMMENT toward the TARGET ENTITY.

Labels:
- favor: the comment supports, praises, agrees with, or defends the target entity.
- against: the comment criticizes, rejects, attacks, or opposes the target entity.
- none: the comment does not express a clear stance toward the target entity, or is mainly factual, joking, off-topic, or unclear.

Important:
- Judge stance toward the TARGET ENTITY, not merely the sentiment of the text.
- A comment attacking the target's opponent may count as favor only if it clearly defends/supports the target.
- If the stance is ambiguous, choose none.
- Reply with JSON only, exactly in this format: {{"label":"favor"}} or {{"label":"against"}} or {{"label":"none"}}.

TARGET ENTITY: {target}
TOPIC: {topic}
POST TITLE: {post_title}
CONVERSATION CONTEXT:
{context}
COMMENT:
{comment}
""".strip()


def parse_label_from_llm(text: str) -> Tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return "invalid", raw

    # Remove common markdown fences.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            lab = str(obj.get("label") or obj.get("stance") or "").strip().lower()
            lab = normalize_label(lab)
            return lab, raw
    except Exception:
        pass

    # Fallback: search for a standalone label, but keep it strict.
    m = re.search(r"\b(favor|favour|against|none|neutral|no stance|no_stance)\b", cleaned, flags=re.IGNORECASE)
    if m:
        return normalize_label(m.group(1)), raw
    return "invalid", raw


def normalize_label(x: str) -> str:
    y = str(x or "").strip().lower()
    mapping = {
        "favour": "favor",
        "support": "favor",
        "supports": "favor",
        "positive": "favor",
        "oppose": "against",
        "opposes": "against",
        "negative": "against",
        "neutral": "none",
        "no stance": "none",
        "no_stance": "none",
        "nostance": "none",
    }
    y = mapping.get(y, y)
    return y if y in STANCE_LABELS else "invalid"


def evaluate_label_predictions(
    rows: Sequence[Dict[str, Any]],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] = STANCE_LABELS,
) -> Dict[str, Any]:
    y_true = [str(r.get("stance")) for r in rows]
    y_pred_norm = [normalize_label(p) for p in y_pred]
    metrics = {
        "rows": len(rows),
        "invalid_predictions": int(sum(1 for p in y_pred_norm if p == "invalid")),
        "accuracy": float(accuracy_score(y_true, y_pred_norm)) if rows else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred_norm, labels=list(labels), average="macro", zero_division=0)) if rows else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred_norm, labels=list(labels), average="weighted", zero_division=0)) if rows else 0.0,
        "classification_report": classification_report(y_true, y_pred_norm, labels=list(labels), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred_norm, labels=list(labels)).tolist(),
        "confusion_matrix_with_invalid": confusion_matrix(y_true, y_pred_norm, labels=PRED_LABELS_WITH_INVALID).tolist(),
        "labels": list(labels),
        "labels_with_invalid": PRED_LABELS_WITH_INVALID,
    }
    return metrics


def topic_metrics_dataframe(rows: Sequence[Dict[str, Any]], y_pred: Sequence[str]) -> pd.DataFrame:
    pred_rows = []
    for r, p in zip(rows, y_pred):
        rr = dict(r)
        rr["pred"] = normalize_label(p)
        pred_rows.append(rr)
    df = pd.DataFrame(pred_rows)
    if df.empty:
        return pd.DataFrame()
    outs = []
    for topic, sub in df.groupby("topic"):
        y_true = list(sub["stance"])
        y_p = list(sub["pred"])
        m = evaluate_label_predictions(sub.to_dict(orient="records"), y_p)
        row = {
            "topic": topic,
            "support": len(sub),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "invalid_predictions": m["invalid_predictions"],
        }
        for lab in STANCE_LABELS:
            rep = m["classification_report"].get(lab, {})
            row[f"{lab}_precision"] = rep.get("precision", 0.0)
            row[f"{lab}_recall"] = rep.get("recall", 0.0)
            row[f"{lab}_f1"] = rep.get("f1-score", 0.0)
            row[f"{lab}_support"] = rep.get("support", 0.0)
        outs.append(row)
    return pd.DataFrame(outs).sort_values("topic")


def write_metrics_bundle(out_dir: Path, model_name: str, rows: Sequence[Dict[str, Any]], y_pred: Sequence[str]) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_label_predictions(rows, y_pred)
    with (out_dir / f"{model_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"gold_{x}" for x in STANCE_LABELS],
        columns=[f"pred_{x}" for x in STANCE_LABELS],
    )
    cm.to_csv(out_dir / f"{model_name}_confusion_matrix.csv")

    cm_invalid = pd.DataFrame(
        metrics["confusion_matrix_with_invalid"],
        index=[f"gold_{x}" for x in PRED_LABELS_WITH_INVALID],
        columns=[f"pred_{x}" for x in PRED_LABELS_WITH_INVALID],
    )
    cm_invalid.to_csv(out_dir / f"{model_name}_confusion_matrix_with_invalid.csv")

    topic_df = topic_metrics_dataframe(rows, y_pred)
    topic_df.to_csv(out_dir / f"{model_name}_metrics_by_topic.csv", index=False)
    return metrics
