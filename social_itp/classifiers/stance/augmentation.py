from __future__ import annotations

import copy
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

PREFIXES_DEFAULT = [
    "oh,",
    "well,",
    "actually,",
    "literally,",
    "to be fair,",
    "honestly,",
]


def _parse_list(x: str | Iterable[str] | None, default: Optional[List[str]] = None, sep: str = "|") -> List[str]:
    if x is None:
        return list(default or [])
    if isinstance(x, str):
        # Support both pipe and comma for convenience.
        if sep in x:
            return [t.strip() for t in x.split(sep) if t.strip()]
        return [t.strip() for t in x.split(",") if t.strip()]
    return [str(t).strip() for t in x if str(t).strip()]


def label_from_row_for_augmentation(row: Dict[str, Any], task: str) -> str:
    stance = str(row.get("stance"))
    if task == "stage1":
        return "none" if stance == "none" else "has_stance"
    return stance


def _prefix_comment_in_input(input_text: str, comment_text: str, prefix: str) -> str:
    input_text = str(input_text or "")
    comment_text = str(comment_text or "")
    if not input_text:
        return f"{prefix} {comment_text}".strip()
    if not comment_text:
        return f"{prefix} {input_text}".strip()
    idx = input_text.rfind(comment_text)
    if idx >= 0:
        new_comment = f"{prefix} {comment_text}".strip()
        return input_text[:idx] + new_comment + input_text[idx + len(comment_text) :]
    return f"{prefix} {input_text}".strip()


def augment_row(row: Dict[str, Any], *, method: str, prefix: str = "", augment_index: int = 0) -> Dict[str, Any]:
    new = copy.deepcopy(row)
    source_id = str(row.get("row_id") or row.get("comment_id") or augment_index)
    new["row_id"] = f"{source_id}::aug{augment_index:06d}"
    new["is_augmented"] = True
    new["augment_method"] = method
    new["augment_source_row_id"] = source_id
    if method == "prefix":
        old_comment = str(new.get("comment_text") or "")
        new_comment = f"{prefix} {old_comment}".strip()
        new["comment_text"] = new_comment
        new["input_text"] = _prefix_comment_in_input(str(new.get("input_text") or ""), old_comment, prefix)
        new["augment_prefix"] = prefix
    else:
        new["augment_prefix"] = ""
    return new


def summarize_counts(rows: List[Dict[str, Any]], *, task: str, scope: str = "topic") -> Dict[str, Any]:
    if scope == "global":
        c = Counter(label_from_row_for_augmentation(r, task) for r in rows)
        return {"global": dict(c)}
    out: Dict[str, Dict[str, int]] = {}
    for r in rows:
        topic = str(r.get("topic") or "__unknown__")
        lab = label_from_row_for_augmentation(r, task)
        out.setdefault(topic, {})[lab] = out.setdefault(topic, {}).get(lab, 0) + 1
    return out


def augment_training_rows(
    rows: List[Dict[str, Any]],
    *,
    task: str = "three_class",
    enabled: bool = False,
    scope: str = "topic",
    target: str = "max",
    methods: str | Iterable[str] = "duplicate,prefix",
    prefixes: str | Iterable[str] | None = None,
    seed: int = 42,
    max_multiplier: float = 3.0,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Oversample minority classes for training only.

    scope:
      - topic: balance labels independently inside each topic.
      - global: balance labels over the whole training set.
    target:
      - max: oversample each group to the largest label count in that scope.
      - median: oversample to median label count.
      - none: no balancing.

    max_multiplier caps each label group's final count to avoid extreme duplication.
    """
    if not enabled or target == "none":
        return list(rows), {
            "enabled": False,
            "original_rows": len(rows),
            "augmented_rows_added": 0,
            "final_rows": len(rows),
            "counts_before": summarize_counts(rows, task=task, scope=scope),
            "counts_after": summarize_counts(rows, task=task, scope=scope),
        }

    rng = random.Random(seed)
    methods_list = _parse_list(methods, default=["duplicate", "prefix"], sep=",")
    prefixes_list = _parse_list(prefixes, default=PREFIXES_DEFAULT, sep="|")
    if not methods_list:
        methods_list = ["duplicate"]

    if scope not in {"topic", "global"}:
        raise ValueError("augmentation scope must be 'topic' or 'global'")

    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = "__global__" if scope == "global" else str(row.get("topic") or "__unknown__")
        lab = label_from_row_for_augmentation(row, task)
        groups[key][lab].append(row)

    out = list(rows)
    aug_index = 0
    per_scope_meta = {}
    for key, label_groups in sorted(groups.items()):
        counts = {lab: len(rs) for lab, rs in label_groups.items()}
        if not counts:
            continue
        vals = sorted(counts.values())
        if target == "max":
            target_count = max(vals)
        elif target == "median":
            target_count = vals[len(vals) // 2]
        else:
            raise ValueError("augmentation target must be max, median, or none")

        label_meta = {}
        for lab, rs in sorted(label_groups.items()):
            if not rs:
                continue
            cap = int(max(len(rs), round(len(rs) * float(max_multiplier)))) if max_multiplier > 0 else target_count
            desired = min(target_count, cap)
            needed = max(0, desired - len(rs))
            label_meta[lab] = {"original": len(rs), "target": desired, "added": needed}
            for _ in range(needed):
                base = rng.choice(rs)
                method = rng.choice(methods_list)
                if method == "prefix" and prefixes_list:
                    prefix = rng.choice(prefixes_list)
                    aug = augment_row(base, method="prefix", prefix=prefix, augment_index=aug_index)
                else:
                    aug = augment_row(base, method="duplicate", prefix="", augment_index=aug_index)
                out.append(aug)
                aug_index += 1
        per_scope_meta[key] = {"target_count": target_count, "labels": label_meta}

    meta = {
        "enabled": True,
        "task": task,
        "scope": scope,
        "target": target,
        "methods": methods_list,
        "prefixes": prefixes_list,
        "seed": seed,
        "max_multiplier": max_multiplier,
        "original_rows": len(rows),
        "augmented_rows_added": len(out) - len(rows),
        "final_rows": len(out),
        "counts_before": summarize_counts(rows, task=task, scope=scope),
        "counts_after": summarize_counts(out, task=task, scope=scope),
        "per_scope": per_scope_meta,
    }
    return out, meta
