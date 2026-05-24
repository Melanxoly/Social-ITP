from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

VALID_STANCES = {"favor", "against", "none"}
DEFAULT_LABELS = ["favor", "against", "none"]
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
ONLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
MARKDOWN_MEDIA_ONLY_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)$|^!\[[^\]]*\]$", re.UNICODE)


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def norm_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def is_meaningful_text(text: str, min_len: int = 2) -> bool:
    text = norm_text(text)
    if len(text) < min_len:
        return False
    if text.lower() in {"[deleted]", "[removed]", "deleted", "removed", "null", "none"}:
        return False
    if URL_RE.sub("", text).strip() == "":
        return False
    if ONLY_PUNCT_RE.match(text):
        return False
    if MARKDOWN_MEDIA_ONLY_RE.match(text):
        return False
    return True


def parse_time(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return datetime.fromtimestamp(float(s))
    except Exception:
        return None


def normalize_stance(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    mapping = {
        "fav": "favor",
        "favour": "favor",
        "pro": "favor",
        "support": "favor",
        "supports": "favor",
        "supporting": "favor",
        "positive": "favor",
        "for": "favor",
        "against": "against",
        "anti": "against",
        "oppose": "against",
        "opposes": "against",
        "opposing": "against",
        "negative": "against",
        "con": "against",
        "neutral": "none",
        "none": "none",
        "no_stance": "none",
        "nostance": "none",
        "no stance": "none",
    }
    s = mapping.get(s, s)
    return s if s in VALID_STANCES else None


def _find_dir_case_insensitive(root: Path, name: str) -> Optional[Path]:
    if not root.exists() or not root.is_dir():
        return None
    want = name.lower()
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == want:
            return p
    return None


def resolve_data_root(data_root: Path, topics: List[str]) -> Tuple[Path, List[str]]:
    """Resolve common accidental roots and return warnings.

    Expected root is the directory that directly contains topic folders, e.g.
    data/MCSD/biden, data/MCSD/trump, ...
    """
    warnings: List[str] = []
    if not data_root.exists():
        raise FileNotFoundError(
            f"data_root does not exist: {data_root}\n"
            "Please pass the directory that directly contains topic folders, for example: data/MCSD"
        )

    if any(_find_dir_case_insensitive(data_root, t) for t in topics):
        return data_root, warnings

    mcsd = _find_dir_case_insensitive(data_root, "MCSD")
    if mcsd and any(_find_dir_case_insensitive(mcsd, t) for t in topics):
        warnings.append(f"data_root looked like parent data dir; auto-resolved to {mcsd}")
        return mcsd, warnings

    warnings.append(
        "No requested topic directory was found under data_root. "
        "Expected folders such as biden, Bitcoin, BMW, costco, tesla, trump directly under data_root."
    )
    return data_root, warnings


def iter_thread_json_files(data_root: Path, topics: List[str]) -> Iterable[Tuple[str, Path]]:
    for topic in topics:
        topic_dir = _find_dir_case_insensitive(data_root, topic)
        if topic_dir is None:
            continue
        for p in sorted(topic_dir.rglob("*.json")):
            yield topic, p


def _comment_sort_key(c: Dict[str, Any]) -> Tuple[str, str]:
    dt = parse_time(c.get("created_time")) or parse_time(c.get("created_utc"))
    tkey = dt.isoformat() if dt is not None else ""
    return (tkey, str(c.get("id") or c.get("comment_id") or ""))


def _collect_ancestor_texts(
    *,
    comment: Dict[str, Any],
    id2c: Dict[str, Dict[str, Any]],
    max_depth: int,
) -> List[str]:
    """Return ancestor comments from far to near, capped by max_depth."""
    out: List[str] = []
    if max_depth <= 0:
        return out
    pid = str(comment.get("parent_id") or "")
    depth = 0
    visited = set()
    while pid and pid != "0" and pid in id2c and depth < max_depth and pid not in visited:
        visited.add(pid)
        pc = id2c[pid]
        txt = norm_text(pc.get("text"))
        if txt:
            out.append(txt)
        pid = str(pc.get("parent_id") or "")
        depth += 1
    return list(reversed(out))


def _truncate(x: str, n: int) -> str:
    x = norm_text(x)
    if n is None or n <= 0 or len(x) <= n:
        return x
    return x[:n].rstrip() + " ..."


def build_input_text(
    *,
    topic: str,
    target_entity: Optional[str],
    post_title: str,
    post_text: str,
    ancestor_texts: List[str],
    comment_text: str,
    include_context: bool,
    input_template: str,
    max_post_chars: int,
    max_parent_chars: int,
) -> str:
    """Build model input.

    input_template:
      - plain: simple field concatenation.
      - task: original task-style template with label definitions and post text.
      - stage1: simplified template for none-vs-has_stance diagnostics.

    The stage1 template is intentionally shorter: target + topic + post title +
    selected ancestors + comment. It excludes long post text and full label
    definitions, because Stage 1 only asks whether a clear stance is expressed.
    """
    comment_text = norm_text(comment_text)
    if not include_context:
        return comment_text

    template = (input_template or "task").lower()
    target = norm_text(target_entity) or "the target entity"
    post_title = _truncate(post_title, max_post_chars)
    post_text = _truncate(post_text, max_post_chars)
    parent_context = "\n".join(
        f"Ancestor comment {i + 1}: {_truncate(t, max_parent_chars)}" for i, t in enumerate(ancestor_texts)
    )

    if template == "plain":
        parts = [f"Topic: {topic}", f"Target: {target}"]
        if post_title:
            parts.append(f"Post title: {post_title}")
        if post_text:
            parts.append(f"Post text: {post_text}")
        if parent_context:
            parts.append(parent_context)
        parts.append(f"Comment: {comment_text}")
        return "\n".join(parts)

    if template in {"stage1", "simple_stage1", "stance_presence"}:
        parts = [
            "Task: Determine whether the comment expresses a clear stance toward the target entity.",
            f"Target entity: {target}",
            f"Discussion topic: {topic}",
        ]
        if post_title:
            parts.append(f"Post title: {post_title}")
        if parent_context:
            parts.append("Conversation context:")
            parts.append(parent_context)
        parts.append(f"Comment: {comment_text}")
        return "\n".join(parts)

    if template != "task":
        raise ValueError(f"Unknown input_template: {input_template}. Use 'task', 'plain', or 'stage1'.")

    parts = [
        "Task: Classify the stance of the comment toward the target entity.",
        "Label definitions:",
        "- favor: the comment supports, praises, agrees with, or defends the target entity.",
        "- against: the comment criticizes, rejects, attacks, or opposes the target entity.",
        "- none: the comment has no clear stance toward the target entity.",
        f"Target entity: {target}",
        f"Discussion topic: {topic}",
    ]
    if post_title:
        parts.append(f"Post title: {post_title}")
    if post_text:
        parts.append(f"Post text: {post_text}")
    if parent_context:
        parts.append("Conversation context:")
        parts.append(parent_context)
    parts.append(f"Comment to classify: {comment_text}")
    return "\n".join(parts)


@dataclass
class StanceDatasetConfig:
    data_root: str
    topics: List[str]
    out_dir: str
    seed: int = 42
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    include_context: bool = True
    input_template: str = "task"
    split_by_topic: bool = False
    min_text_len: int = 2
    max_post_chars: int = 600
    max_parent_chars: int = 360
    max_ancestor_depth: int = 1
    fail_on_empty: bool = True


def scan_stance_sources(data_root: str, topics: List[str]) -> Dict[str, Any]:
    root, warnings = resolve_data_root(Path(data_root), topics)
    report: Dict[str, Any] = {
        "input_data_root": str(data_root),
        "resolved_data_root": str(root),
        "warnings": warnings,
        "topics": {},
        "total_json_files": 0,
        "total_comments": 0,
        "total_labeled_comments": 0,
        "total_supported_labels": 0,
    }
    for topic in topics:
        topic_dir = _find_dir_case_insensitive(root, topic)
        topic_report = {
            "topic_dir": str(topic_dir) if topic_dir else None,
            "json_files": 0,
            "comments": 0,
            "labeled_comments": 0,
            "supported_labels": 0,
            "raw_label_counts": {},
            "normalized_label_counts": {},
        }
        if topic_dir:
            for path in sorted(topic_dir.rglob("*.json")):
                topic_report["json_files"] += 1
                obj = safe_load_json(path)
                if not obj:
                    continue
                comments = obj.get("comments") or []
                if not isinstance(comments, list):
                    continue
                topic_report["comments"] += len(comments)
                for c in comments:
                    if not isinstance(c, dict):
                        continue
                    raw = (c.get("anno_label") or {}).get("stance")
                    if raw is not None:
                        topic_report["labeled_comments"] += 1
                        rs = str(raw)
                        topic_report["raw_label_counts"][rs] = topic_report["raw_label_counts"].get(rs, 0) + 1
                    norm = normalize_stance(raw)
                    if norm is not None:
                        topic_report["supported_labels"] += 1
                        topic_report["normalized_label_counts"][norm] = topic_report["normalized_label_counts"].get(norm, 0) + 1
        report["topics"][topic] = topic_report
        report["total_json_files"] += topic_report["json_files"]
        report["total_comments"] += topic_report["comments"]
        report["total_labeled_comments"] += topic_report["labeled_comments"]
        report["total_supported_labels"] += topic_report["supported_labels"]
    return report


def extract_stance_rows(cfg: StanceDatasetConfig) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root, warnings = resolve_data_root(Path(cfg.data_root), cfg.topics)
    rows: List[Dict[str, Any]] = []
    scan = scan_stance_sources(str(root), cfg.topics)
    scan["input_data_root"] = cfg.data_root
    scan["resolved_data_root"] = str(root)
    scan["warnings"] = warnings + scan.get("warnings", [])

    for topic, thread_path in iter_thread_json_files(root, cfg.topics):
        thread = safe_load_json(thread_path)
        if not isinstance(thread, dict):
            continue

        thread_id = str(thread.get("post_id") or thread.get("id") or thread_path.stem)
        post_title = norm_text(thread.get("post_title") or thread.get("title"))
        post_text = norm_text(thread.get("text") or thread.get("post_text") or thread.get("selftext"))
        target_entity = thread.get("target")
        comments = [c for c in (thread.get("comments") or []) if isinstance(c, dict)]
        id2c = {str(c.get("id")): c for c in comments if c.get("id") is not None}

        for c in sorted(comments, key=_comment_sort_key):
            label = normalize_stance((c.get("anno_label") or {}).get("stance"))
            if label is None:
                continue
            text = norm_text(c.get("text"))
            if not is_meaningful_text(text, min_len=cfg.min_text_len):
                continue

            parent_id = str(c.get("parent_id") or "0")
            ancestor_texts = _collect_ancestor_texts(comment=c, id2c=id2c, max_depth=cfg.max_ancestor_depth)
            parent_text = ancestor_texts[-1] if ancestor_texts else ""
            user_obj = c.get("user", {}) or {}
            comment_id = str(c.get("id") or c.get("comment_id") or "")
            row = {
                "row_id": f"{thread_id}::{comment_id}",
                "thread_id": thread_id,
                "topic": topic,
                "target_entity": target_entity,
                "comment_id": comment_id,
                "reddit_comment_id": str(c.get("comment_id") or ""),
                "parent_id": parent_id,
                "user_id": str(user_obj.get("user_id") or ""),
                "user_name": user_obj.get("user_name"),
                "created_time": c.get("created_time") or c.get("created_utc"),
                "post_title": post_title,
                "post_text": post_text,
                "parent_text": parent_text,
                "ancestor_texts": ancestor_texts,
                "comment_text": text,
                "input_text": build_input_text(
                    topic=topic,
                    target_entity=target_entity,
                    post_title=post_title,
                    post_text=post_text,
                    ancestor_texts=ancestor_texts,
                    comment_text=text,
                    include_context=cfg.include_context,
                    input_template=cfg.input_template,
                    max_post_chars=cfg.max_post_chars,
                    max_parent_chars=cfg.max_parent_chars,
                ),
                "stance": label,
                "source_file": str(thread_path),
            }
            rows.append(row)
    return rows, scan


def _split_thread_ids(
    thread_ids: List[str],
    *,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> Tuple[set[str], set[str], set[str]]:
    rng = random.Random(seed)
    ids = list(thread_ids)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return set(), set(), set()
    if n == 1:
        return set(ids), set(), set()
    if n == 2:
        return set(ids[:1]), set(), set(ids[1:])

    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    n_train = max(1, min(n_train, n - 2))
    n_dev = max(1 if dev_ratio > 0 else 0, min(n_dev, n - n_train - 1))
    train_ids = set(ids[:n_train])
    dev_ids = set(ids[n_train : n_train + n_dev])
    test_ids = set(ids[n_train + n_dev :])
    return train_ids, dev_ids, test_ids


def split_rows_by_thread_global(
    rows: List[Dict[str, Any]],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(str(r["thread_id"]), []).append(r)
    train_ids, dev_ids, test_ids = _split_thread_ids(
        list(grouped.keys()), seed=seed, train_ratio=train_ratio, dev_ratio=dev_ratio
    )
    splits = {"train": [], "dev": [], "test": []}
    for tid, rs in grouped.items():
        if tid in train_ids:
            splits["train"].extend(rs)
        elif tid in dev_ids:
            splits["dev"].extend(rs)
        else:
            splits["test"].extend(rs)
    return splits


def split_rows_by_thread_within_topic(
    rows: List[Dict[str, Any]],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> Dict[str, List[Dict[str, Any]]]:
    topic_thread_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in rows:
        topic = str(r.get("topic") or "")
        tid = str(r.get("thread_id") or "")
        topic_thread_rows.setdefault(topic, {}).setdefault(tid, []).append(r)

    splits = {"train": [], "dev": [], "test": []}
    for i, (topic, grouped) in enumerate(sorted(topic_thread_rows.items())):
        train_ids, dev_ids, test_ids = _split_thread_ids(
            list(grouped.keys()), seed=seed + 9973 * i, train_ratio=train_ratio, dev_ratio=dev_ratio
        )
        for tid, rs in grouped.items():
            if tid in train_ids:
                splits["train"].extend(rs)
            elif tid in dev_ids:
                splits["dev"].extend(rs)
            else:
                splits["test"].extend(rs)
    return splits


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
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


def _summarize_split(split_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_counts: Dict[str, int] = {}
    topic_counts: Dict[str, int] = {}
    topic_label_counts: Dict[str, Dict[str, int]] = {}
    threads = set()
    for r in split_rows:
        lab = str(r["stance"])
        topic = str(r["topic"])
        label_counts[lab] = label_counts.get(lab, 0) + 1
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        topic_label_counts.setdefault(topic, {})[lab] = topic_label_counts.setdefault(topic, {}).get(lab, 0) + 1
        threads.add(r["thread_id"])
    return {
        "rows": len(split_rows),
        "threads": len(threads),
        "label_counts": label_counts,
        "topic_counts": topic_counts,
        "topic_label_counts": topic_label_counts,
    }


def write_dataset(cfg: StanceDatasetConfig) -> Dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, scan = extract_stance_rows(cfg)

    with (out_dir / "scan_report.json").open("w", encoding="utf-8") as f:
        json.dump(scan, f, ensure_ascii=False, indent=2)

    if not rows and cfg.fail_on_empty:
        raise RuntimeError(
            "No stance rows were extracted. See scan_report.json for directory and label diagnostics. "
            f"Output dir: {out_dir}"
        )

    if cfg.split_by_topic:
        splits = split_rows_by_thread_within_topic(
            rows,
            seed=cfg.seed,
            train_ratio=cfg.train_ratio,
            dev_ratio=cfg.dev_ratio,
        )
        split_mode = "thread_split_within_each_topic"
    else:
        splits = split_rows_by_thread_global(
            rows,
            seed=cfg.seed,
            train_ratio=cfg.train_ratio,
            dev_ratio=cfg.dev_ratio,
        )
        split_mode = "global_thread_split"

    for split, split_rows in splits.items():
        write_jsonl(out_dir / f"{split}.jsonl", split_rows)

    summary = {
        "data_root": cfg.data_root,
        "resolved_data_root": scan.get("resolved_data_root"),
        "topics": cfg.topics,
        "seed": cfg.seed,
        "train_ratio": cfg.train_ratio,
        "dev_ratio": cfg.dev_ratio,
        "include_context": cfg.include_context,
        "input_template": cfg.input_template,
        "split_by_topic": cfg.split_by_topic,
        "split_mode": split_mode,
        "max_ancestor_depth": cfg.max_ancestor_depth,
        "total_rows": len(rows),
        "scan_report_path": str(out_dir / "scan_report.json"),
        "splits": {split: _summarize_split(split_rows) for split, split_rows in splits.items()},
    }
    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
