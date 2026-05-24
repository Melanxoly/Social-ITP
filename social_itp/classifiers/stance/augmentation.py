from __future__ import annotations

import copy
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

PREFIXES_DEFAULT = [
    "oh,",
    "well,",
    "actually,",
    "literally,",
    "to be fair,",
    "honestly,",
    "tbh,",
    "imo,",
]

# Lightweight, conservative substitutions intended for social-media stance text.
# They are applied only to the current comment, not to the target/post/parent fields.
SYNONYMS_DEFAULT: List[Tuple[str, str]] = [
    ("you", "u"),
    ("your", "ur"),
    ("are", "r"),
    ("to be honest", "tbh"),
    ("honestly", "to be honest"),
    ("in my opinion", "imo"),
    ("because", "bc"),
    ("people", "folks"),
    ("agree", "support"),
    ("support", "back"),
    ("disagree", "oppose"),
    ("oppose", "disagree"),
    ("opposed to", "against"),
    ("think", "believe"),
    ("wrong", "incorrect"),
    ("right", "correct"),
    ("bad", "terrible"),
    ("good", "great"),
    ("do not", "don't"),
    ("does not", "doesn't"),
    ("cannot", "can't"),
]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _parse_list(x: str | Iterable[str] | None, default: Optional[List[str]] = None, sep: str = "|") -> List[str]:
    if x is None:
        return list(default or [])
    if isinstance(x, str):
        # Support both pipe and comma for convenience.
        if sep in x:
            return [t.strip() for t in x.split(sep) if t.strip()]
        return [t.strip() for t in x.split(",") if t.strip()]
    return [str(t).strip() for t in x if str(t).strip()]


def _parse_synonym_pairs(x: str | Iterable[str] | None) -> List[Tuple[str, str]]:
    if x is None or (isinstance(x, str) and not x.strip()):
        return list(SYNONYMS_DEFAULT)
    items = _parse_list(x, default=[], sep="|")
    pairs: List[Tuple[str, str]] = []
    for item in items:
        if "=>" in item:
            a, b = item.split("=>", 1)
        elif ":" in item:
            a, b = item.split(":", 1)
        else:
            continue
        a, b = a.strip(), b.strip()
        if a and b:
            pairs.append((a, b))
    return pairs or list(SYNONYMS_DEFAULT)


def label_from_row_for_augmentation(row: Dict[str, Any], task: str) -> str:
    stance = str(row.get("stance"))
    if task == "stage1":
        return "none" if stance == "none" else "has_stance"
    return stance


def _replace_comment_in_input(input_text: str, old_comment: str, new_comment: str) -> str:
    input_text = str(input_text or "")
    old_comment = str(old_comment or "")
    new_comment = str(new_comment or "")
    if not input_text:
        return new_comment
    if not old_comment:
        return input_text
    idx = input_text.rfind(old_comment)
    if idx >= 0:
        return input_text[:idx] + new_comment + input_text[idx + len(old_comment) :]
    # Fallback: avoid accidentally changing target/post/parent; append transformed comment.
    return f"{input_text}\nAugmented comment: {new_comment}".strip()


def _random_case_words(text: str, rng: random.Random, case_modes: List[str]) -> str:
    modes = case_modes or ["lower", "upper", "title"]

    def repl(m: re.Match) -> str:
        token = m.group(0)
        mode = rng.choice(modes)
        if mode == "lower":
            return token.lower()
        if mode == "upper":
            return token.upper()
        if mode in {"title", "capital", "capitalize"}:
            return token[:1].upper() + token[1:].lower()
        return token

    return WORD_RE.sub(repl, text)


def _synonym_substitute(
    text: str,
    rng: random.Random,
    synonym_pairs: List[Tuple[str, str]],
    *,
    max_replacements: int = 2,
) -> tuple[str, List[Tuple[str, str]]]:
    if not text or not synonym_pairs:
        return text, []
    pairs = list(synonym_pairs)
    rng.shuffle(pairs)
    out = text
    used: List[Tuple[str, str]] = []
    for src, tgt in pairs:
        if len(used) >= max_replacements:
            break
        # Phrase-safe case-insensitive replacement. Use word boundaries for alphanumeric phrases.
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(src) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        if pat.search(out):
            out = pat.sub(tgt, out, count=1)
            used.append((src, tgt))
    return out, used


def _apply_one_transform(
    comment: str,
    *,
    method: str,
    rng: random.Random,
    prefixes: List[str],
    synonym_pairs: List[Tuple[str, str]],
    case_modes: List[str],
) -> tuple[str, Dict[str, Any]]:
    method = method.strip().lower()
    meta: Dict[str, Any] = {"method": method}
    if method == "prefix":
        prefix = rng.choice(prefixes) if prefixes else "honestly,"
        meta["prefix"] = prefix
        return f"{prefix} {comment}".strip(), meta
    if method in {"casing", "case", "random_case"}:
        meta["case_modes"] = case_modes
        return _random_case_words(comment, rng, case_modes), meta
    if method in {"synonym", "synonyms", "replace"}:
        new_text, used = _synonym_substitute(comment, rng, synonym_pairs)
        meta["synonyms_used"] = used
        return new_text, meta
    # duplicate or unknown -> no textual change.
    meta["method"] = "duplicate"
    return comment, meta


def augment_row(
    row: Dict[str, Any],
    *,
    method: str,
    prefix: str = "",
    augment_index: int = 0,
    rng: Optional[random.Random] = None,
    prefixes: Optional[List[str]] = None,
    synonym_pairs: Optional[List[Tuple[str, str]]] = None,
    case_modes: Optional[List[str]] = None,
    chain_min: int = 1,
    chain_max: int = 2,
) -> Dict[str, Any]:
    rng = rng or random.Random(augment_index)
    prefixes = prefixes or PREFIXES_DEFAULT
    synonym_pairs = synonym_pairs or SYNONYMS_DEFAULT
    case_modes = case_modes or ["lower", "upper", "title"]

    new = copy.deepcopy(row)
    source_id = str(row.get("row_id") or row.get("comment_id") or augment_index)
    old_comment = str(new.get("comment_text") or "")
    new_comment = old_comment
    method_l = str(method or "duplicate").strip().lower()
    applied: List[Dict[str, Any]] = []

    if method_l == "mixed":
        candidates = ["prefix", "casing", "synonym"]
        k = rng.randint(max(1, chain_min), max(max(1, chain_min), chain_max))
        chosen = rng.sample(candidates, k=min(k, len(candidates)))
        for m in chosen:
            new_comment, meta = _apply_one_transform(
                new_comment,
                method=m,
                rng=rng,
                prefixes=prefixes,
                synonym_pairs=synonym_pairs,
                case_modes=case_modes,
            )
            applied.append(meta)
    elif method_l == "prefix" and prefix:
        new_comment, meta = _apply_one_transform(
            new_comment,
            method="prefix",
            rng=rng,
            prefixes=[prefix],
            synonym_pairs=synonym_pairs,
            case_modes=case_modes,
        )
        applied.append(meta)
    else:
        new_comment, meta = _apply_one_transform(
            new_comment,
            method=method_l,
            rng=rng,
            prefixes=prefixes,
            synonym_pairs=synonym_pairs,
            case_modes=case_modes,
        )
        applied.append(meta)

    new["row_id"] = f"{source_id}::aug{augment_index:06d}"
    new["is_augmented"] = True
    new["augment_method"] = method_l
    new["augment_applied"] = applied
    new["augment_source_row_id"] = source_id
    new["comment_text"] = new_comment
    new["input_text"] = _replace_comment_in_input(str(new.get("input_text") or ""), old_comment, new_comment)
    # Backward-compatible metadata fields.
    new["augment_prefix"] = next((m.get("prefix", "") for m in applied if m.get("prefix")), "")
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
    synonyms: str | Iterable[str] | None = None,
    case_modes: str | Iterable[str] | None = None,
    seed: int = 42,
    max_multiplier: float = 3.0,
    final_multiplier: float = 1.0,
    chain_min: int = 1,
    chain_max: int = 2,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Oversample training rows only.

    The default mode balances minority classes inside each scope. The new
    `final_multiplier` supports stronger augmentation after balancing:

      counts 80, 200, 300; target=max; final_multiplier=3
      => final desired count for each label is 900.

    Important: `max_multiplier` still caps final count relative to each label's
    original count when it is > 0. Set `max_multiplier <= 0` to disable this cap
    for aggressive augmentation such as 80 -> 900.

    methods:
      duplicate: exact copy
      prefix: prepend safe discourse marker to current comment
      casing: random per-word lower/upper/title casing
      synonym: lightweight phrase/word substitutions
      mixed: compose 1-2 random transforms from prefix/casing/synonym
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
    synonym_pairs = _parse_synonym_pairs(synonyms)
    case_modes_list = _parse_list(case_modes, default=["lower", "upper", "title"], sep=",")
    if not methods_list:
        methods_list = ["duplicate"]

    if scope not in {"topic", "global"}:
        raise ValueError("augmentation scope must be 'topic' or 'global'")
    if final_multiplier <= 0:
        raise ValueError("final_multiplier must be > 0")

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
            base_target_count = max(vals)
        elif target == "median":
            base_target_count = vals[len(vals) // 2]
        else:
            raise ValueError("augmentation target must be max, median, or none")

        final_target_count = int(round(base_target_count * float(final_multiplier)))
        label_meta = {}
        for lab, rs in sorted(label_groups.items()):
            if not rs:
                continue
            desired = max(len(rs), final_target_count)
            if max_multiplier and max_multiplier > 0:
                cap = int(max(len(rs), round(len(rs) * float(max_multiplier))))
                desired = min(desired, cap)
            needed = max(0, desired - len(rs))
            label_meta[lab] = {
                "original": len(rs),
                "base_target": base_target_count,
                "final_target_before_cap": final_target_count,
                "final_target": desired,
                "added": needed,
            }
            for _ in range(needed):
                base = rng.choice(rs)
                method = rng.choice(methods_list)
                prefix = rng.choice(prefixes_list) if prefixes_list else ""
                aug = augment_row(
                    base,
                    method=method,
                    prefix=prefix,
                    augment_index=aug_index,
                    rng=rng,
                    prefixes=prefixes_list,
                    synonym_pairs=synonym_pairs,
                    case_modes=case_modes_list,
                    chain_min=chain_min,
                    chain_max=chain_max,
                )
                out.append(aug)
                aug_index += 1
        per_scope_meta[key] = {
            "base_target_count": base_target_count,
            "final_multiplier": final_multiplier,
            "final_target_count_before_cap": final_target_count,
            "labels": label_meta,
        }

    meta = {
        "enabled": True,
        "task": task,
        "scope": scope,
        "target": target,
        "methods": methods_list,
        "prefixes": prefixes_list,
        "synonyms": synonym_pairs,
        "case_modes": case_modes_list,
        "seed": seed,
        "max_multiplier": max_multiplier,
        "final_multiplier": final_multiplier,
        "chain_min": chain_min,
        "chain_max": chain_max,
        "original_rows": len(rows),
        "augmented_rows_added": len(out) - len(rows),
        "final_rows": len(out),
        "counts_before": summarize_counts(rows, task=task, scope=scope),
        "counts_after": summarize_counts(out, task=task, scope=scope),
        "per_scope": per_scope_meta,
    }
    return out, meta
