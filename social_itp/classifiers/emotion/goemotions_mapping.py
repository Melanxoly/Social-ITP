from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EMOTION_LABELS = ["negative", "neutral", "positive"]

# GoEmotions simplified taxonomy: 27 emotions + neutral.
# User-requested special case: confusion -> negative.
POSITIVE_FINE = {
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
}

NEGATIVE_FINE = {
    "anger",
    "annoyance",
    "confusion",  # user-requested mapping
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
}

NEUTRAL_FINE = {
    "neutral",
    "curiosity",
    "realization",
    "surprise",
}

FINE_TO_COARSE: Dict[str, str] = {}
FINE_TO_COARSE.update({k: "positive" for k in POSITIVE_FINE})
FINE_TO_COARSE.update({k: "negative" for k in NEGATIVE_FINE})
FINE_TO_COARSE.update({k: "neutral" for k in NEUTRAL_FINE})


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def map_fine_labels_to_coarse(
    fine_labels: Sequence[str],
    *,
    multi_label_policy: str = "skip_conflict",
) -> Optional[str]:
    """Map one GoEmotions multi-label annotation to negative/neutral/positive.

    Policies:
      - skip_conflict: skip examples containing both positive and negative coarse labels.
      - priority_negative: if any negative label exists, output negative; else positive; else neutral.
      - priority_non_neutral: if any positive/negative and no conflict, output that; conflicts -> negative.

    Neutral labels co-occurring with non-neutral labels do not override non-neutral emotion.
    """
    coarse = {FINE_TO_COARSE[x] for x in fine_labels if x in FINE_TO_COARSE}
    if not coarse:
        return None
    non_neutral = coarse - {"neutral"}
    if not non_neutral:
        return "neutral"

    has_pos = "positive" in non_neutral
    has_neg = "negative" in non_neutral

    if has_pos and has_neg:
        if multi_label_policy == "skip_conflict":
            return None
        return "negative"
    if has_neg:
        return "negative"
    if has_pos:
        return "positive"
    return "neutral"


def load_goemotions_dataset(dataset_name: str = "google-research-datasets/go_emotions", config_name: str = "simplified"):
    """Load GoEmotions with fallback to the legacy datasets name."""
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("Please install datasets: pip install -r requirements_emotion_goemotions.txt") from e

    errors = []
    for name in [dataset_name, "go_emotions"]:
        try:
            return load_dataset(name, config_name)
        except Exception as e:
            errors.append(f"{name}: {repr(e)}")
    raise RuntimeError("Failed to load GoEmotions. Tried dataset names:\n" + "\n".join(errors))


def get_goemotions_label_names(ds_split) -> List[str]:
    features = ds_split.features
    labels_feature = features["labels"]
    # datasets.Sequence(datasets.ClassLabel)
    if hasattr(labels_feature, "feature") and hasattr(labels_feature.feature, "names"):
        return list(labels_feature.feature.names)
    # fallback
    return [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
    ]


def convert_split_to_rows(
    ds_split,
    *,
    split_name: str,
    label_names: Sequence[str],
    multi_label_policy: str = "skip_conflict",
    prefer_single_label: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    skipped_conflict = 0
    skipped_no_label = 0
    multi_label_kept = 0
    multi_label_skipped_by_prefer_single = 0
    for i, ex in enumerate(ds_split):
        label_ids = list(ex.get("labels") or [])
        fine = [label_names[j] for j in label_ids if 0 <= int(j) < len(label_names)]
        if not fine:
            skipped_no_label += 1
            continue
        if prefer_single_label and len({FINE_TO_COARSE.get(x) for x in fine if x in FINE_TO_COARSE}) > 1:
            # This removes examples containing mixed coarse labels, but keeps cases like anger+annoyance.
            coarse_set = {FINE_TO_COARSE.get(x) for x in fine if x in FINE_TO_COARSE}
            non_null = {x for x in coarse_set if x is not None}
            if len(non_null) > 1:
                multi_label_skipped_by_prefer_single += 1
                continue
        emotion = map_fine_labels_to_coarse(fine, multi_label_policy=multi_label_policy)
        if emotion is None:
            skipped_conflict += 1
            continue
        if len(fine) > 1:
            multi_label_kept += 1
        rid = str(ex.get("id") or f"{split_name}-{i}")
        text = str(ex.get("text") or "").strip()
        rows.append(
            {
                "row_id": f"goemotions::{split_name}::{rid}",
                "source_dataset": "go_emotions",
                "source_split": split_name,
                "text": text,
                "input_text": f"Comment: {text}",
                "emotion": emotion,
                "fine_emotions": fine,
                "fine_label_ids": label_ids,
                "num_fine_labels": len(fine),
            }
        )
    meta = {
        "split": split_name,
        "input_rows": len(ds_split),
        "output_rows": len(rows),
        "label_counts": dict(Counter(r["emotion"] for r in rows)),
        "skipped_conflict": skipped_conflict,
        "skipped_no_label": skipped_no_label,
        "multi_label_kept": multi_label_kept,
        "multi_label_skipped_by_prefer_single": multi_label_skipped_by_prefer_single,
    }
    return rows, meta


def balanced_sample(
    rows: Sequence[Dict[str, Any]],
    *,
    per_class: int,
    seed: int = 42,
    allow_less: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("emotion") in EMOTION_LABELS:
            by_label[str(r["emotion"])].append(r)
    sampled: List[Dict[str, Any]] = []
    counts_before = {lab: len(by_label.get(lab, [])) for lab in EMOTION_LABELS}
    counts_after = {}
    for lab in EMOTION_LABELS:
        group = list(by_label.get(lab, []))
        rng.shuffle(group)
        if len(group) < per_class and not allow_less:
            raise RuntimeError(f"Not enough rows for {lab}: need {per_class}, got {len(group)}")
        take = min(per_class, len(group))
        sampled.extend(group[:take])
        counts_after[lab] = take
    rng.shuffle(sampled)
    meta = {
        "per_class_requested": per_class,
        "allow_less": allow_less,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "total_after": len(sampled),
    }
    return sampled, meta


def make_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "label_counts": dict(Counter(str(r.get("emotion")) for r in rows)),
        "source_split_counts": dict(Counter(str(r.get("source_split")) for r in rows)),
    }
