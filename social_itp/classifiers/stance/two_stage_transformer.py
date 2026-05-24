from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import Dataset

THREE_LABELS = ["favor", "against", "none"]
STAGE1_LABELS = ["none", "has_stance"]
STAGE2_LABELS = ["favor", "against"]


@dataclass
class TwoStageConfig:
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 384
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.06
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    stage1_threshold: float = 0.5
    balance_train: bool = False
    balance_scope: str = "global"  # global | topic
    balance_max_multiplier: float = 3.0


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


def rows_to_texts(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(r.get("input_text") or r.get("comment_text") or "") for r in rows]


def stage1_label(row: Dict[str, Any]) -> str:
    return "none" if str(row.get("stance")) == "none" else "has_stance"


def stage2_label(row: Dict[str, Any]) -> str:
    y = str(row.get("stance"))
    if y not in STAGE2_LABELS:
        raise ValueError(f"stage2 expects favor/against rows, got {y}")
    return y


class TextLabelDataset(Dataset):
    def __init__(self, encodings: Dict[str, Any], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_dataset(tokenizer: Any, rows: Sequence[Dict[str, Any]], labels: List[str], label_getter, max_length: int) -> TextLabelDataset:
    label2id = {lab: i for i, lab in enumerate(labels)}
    texts = rows_to_texts(rows)
    ys = [label2id[label_getter(r)] for r in rows]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return TextLabelDataset(enc, ys)


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def make_training_args(output_dir: str, cfg: TwoStageConfig, has_eval: bool = True):
    from transformers import TrainingArguments

    common = dict(
        output_dir=output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=50,
        save_total_limit=2,
        seed=cfg.seed,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        report_to="none",
        save_strategy="epoch",
    )
    if has_eval:
        common.update(dict(load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True))
        try:
            return TrainingArguments(evaluation_strategy="epoch", **common)
        except TypeError:
            return TrainingArguments(eval_strategy="epoch", **common)
    return TrainingArguments(**common)


def trainer_kwargs_with_tokenizer(Trainer, *, model, args, train_dataset=None, eval_dataset=None, tokenizer=None, compute_metrics=None):
    import inspect

    kwargs = dict(model=model, args=args)
    if train_dataset is not None:
        kwargs["train_dataset"] = train_dataset
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset
    if compute_metrics is not None:
        kwargs["compute_metrics"] = compute_metrics
    sig = inspect.signature(Trainer.__init__).parameters
    if tokenizer is not None:
        if "processing_class" in sig:
            kwargs["processing_class"] = tokenizer
        elif "tokenizer" in sig:
            kwargs["tokenizer"] = tokenizer
    return kwargs


def make_compute_metrics(labels: List[str]):
    id2label = {i: lab for i, lab in enumerate(labels)}

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        pred_ids = np.argmax(logits, axis=1)
        y_true = [id2label[int(i)] for i in label_ids]
        y_pred = [id2label[int(i)] for i in pred_ids]
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        }
        if labels == STAGE1_LABELS and len(set(y_true)) > 1:
            p = softmax(logits)[:, labels.index("has_stance")]
            y_int = np.array([1 if y == "has_stance" else 0 for y in y_true])
            try:
                out["roc_auc"] = float(roc_auc_score(y_int, p))
                out["average_precision"] = float(average_precision_score(y_int, p))
            except Exception:
                pass
        return out

    return compute_metrics


def balance_rows(
    rows: List[Dict[str, Any]],
    *,
    label_getter,
    scope: str = "global",
    seed: int = 42,
    max_multiplier: float = 3.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if scope not in {"global", "topic"}:
        raise ValueError("balance_scope must be global or topic")
    rng = random.Random(seed)
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = "__global__" if scope == "global" else str(r.get("topic") or "__unknown__")
        groups[key][label_getter(r)].append(r)

    out = list(rows)
    added = 0
    meta = {"enabled": True, "scope": scope, "max_multiplier": max_multiplier, "groups": {}}
    for key, labs in sorted(groups.items()):
        counts = {lab: len(rs) for lab, rs in labs.items()}
        if not counts:
            continue
        target = max(counts.values())
        meta["groups"][key] = {"counts_before": counts, "target": target, "counts_after": {}}
        for lab, rs in sorted(labs.items()):
            desired = target
            if max_multiplier and max_multiplier > 0:
                desired = min(desired, int(max(len(rs), round(len(rs) * max_multiplier))))
            need = max(0, desired - len(rs))
            for i in range(need):
                base = dict(rng.choice(rs))
                base["row_id"] = f"{base.get('row_id', base.get('comment_id', 'row'))}::two_stage_balance{added:06d}"
                base["is_balanced_duplicate"] = True
                base["balance_source_label"] = lab
                out.append(base)
                added += 1
            meta["groups"][key]["counts_after"][lab] = desired
    meta["original_rows"] = len(rows)
    meta["added_rows"] = added
    meta["final_rows"] = len(out)
    return out, meta


def stage1_threshold_sweep(rows: Sequence[Dict[str, Any]], p_has: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    y_true = [stage1_label(r) for r in rows]
    y_int = np.array([1 if y == "has_stance" else 0 for y in y_true])
    roc_auc = None
    avg_precision = None
    if len(set(y_int.tolist())) > 1:
        try:
            roc_auc = float(roc_auc_score(y_int, p_has))
            avg_precision = float(average_precision_score(y_int, p_has))
        except Exception:
            pass
    out = []
    for th in thresholds:
        pred_int = (p_has >= float(th)).astype(int)
        pred_lab = ["has_stance" if x else "none" for x in pred_int]
        cm = confusion_matrix(y_int, pred_int, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
        none_to_stance = fp / max(1, tn + fp)
        stance_to_none = fn / max(1, fn + tp)
        out.append(
            {
                "threshold": float(th),
                "accuracy": float(accuracy_score(y_int, pred_int)),
                "balanced_accuracy": float(balanced_accuracy_score(y_int, pred_int)),
                "macro_f1": float(f1_score(y_true, pred_lab, labels=STAGE1_LABELS, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, pred_lab, labels=STAGE1_LABELS, average="weighted", zero_division=0)),
                "none_to_stance_rate": float(none_to_stance),
                "stance_to_none_rate": float(stance_to_none),
                "boundary_avg": float((none_to_stance + stance_to_none) / 2.0),
                "boundary_max": float(max(none_to_stance, stance_to_none)),
                "boundary_weighted": float((2.0 * none_to_stance + stance_to_none) / 3.0),
                "roc_auc": roc_auc,
                "average_precision": avg_precision,
                "tn_none_as_none": tn,
                "fp_none_as_stance": fp,
                "fn_stance_as_none": fn,
                "tp_stance_as_stance": tp,
            }
        )
    return pd.DataFrame(out)


def combine_two_stage_predictions(
    rows: Sequence[Dict[str, Any]],
    *,
    stage1_logits: np.ndarray,
    stage2_logits: Optional[np.ndarray],
    stage1_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    p1 = softmax(stage1_logits)
    p_has = p1[:, STAGE1_LABELS.index("has_stance")]
    pass_stage1 = p_has >= float(stage1_threshold)

    if stage2_logits is None:
        p2 = np.zeros((len(rows), 2), dtype=np.float32)
        pred2 = np.array([0] * len(rows))
    else:
        p2 = softmax(stage2_logits)
        pred2 = np.argmax(p2, axis=1)

    final_labels: List[str] = []
    pred_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        stage1_pred = "has_stance" if pass_stage1[i] else "none"
        if stage1_pred == "none":
            final = "none"
        else:
            final = STAGE2_LABELS[int(pred2[i])]
        final_labels.append(final)
        pred_rows.append(
            {
                "row_id": r.get("row_id"),
                "thread_id": r.get("thread_id"),
                "topic": r.get("topic"),
                "target_entity": r.get("target_entity"),
                "comment_id": r.get("comment_id"),
                "gold_stance": r.get("stance"),
                "gold_stage1": stage1_label(r),
                "stage1_pred": stage1_pred,
                "stage1_threshold": float(stage1_threshold),
                "p_none": float(p1[i, STAGE1_LABELS.index("none")]),
                "p_has_stance": float(p_has[i]),
                "stage2_pred": STAGE2_LABELS[int(pred2[i])],
                "p_favor_given_stance": float(p2[i, STAGE2_LABELS.index("favor")]) if p2.shape[0] else None,
                "p_against_given_stance": float(p2[i, STAGE2_LABELS.index("against")]) if p2.shape[0] else None,
                "pred_stance": final,
                "correct": str(r.get("stance")) == final,
                "post_title": r.get("post_title"),
                "parent_text": r.get("parent_text"),
                "comment_text": r.get("comment_text"),
                "input_text": r.get("input_text"),
            }
        )
    return pred_rows, final_labels


def evaluate_three_class(rows: Sequence[Dict[str, Any]], y_pred: Sequence[str]) -> Dict[str, Any]:
    y_true = [str(r.get("stance")) for r in rows]
    cm = confusion_matrix(y_true, y_pred, labels=THREE_LABELS)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=THREE_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=THREE_LABELS, average="weighted", zero_division=0)),
        "labels": THREE_LABELS,
        "classification_report": classification_report(y_true, y_pred, labels=THREE_LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }
    # Important boundary rates.
    idx = {lab: i for i, lab in enumerate(THREE_LABELS)}
    none_total = max(1, int(cm[idx["none"]].sum()))
    stance_total = max(1, int(cm[idx["favor"]].sum() + cm[idx["against"]].sum()))
    none_to_stance = (cm[idx["none"], idx["favor"]] + cm[idx["none"], idx["against"]]) / none_total
    stance_to_none = (cm[idx["favor"], idx["none"]] + cm[idx["against"], idx["none"]]) / stance_total
    favor_against = (cm[idx["favor"], idx["against"]] + cm[idx["against"], idx["favor"]]) / stance_total
    metrics.update(
        {
            "none_to_stance_rate": float(none_to_stance),
            "stance_to_none_rate": float(stance_to_none),
            "favor_against_confusion_rate": float(favor_against),
            "boundary_avg": float((none_to_stance + stance_to_none) / 2.0),
            "boundary_max": float(max(none_to_stance, stance_to_none)),
        }
    )
    return metrics


def write_eval_artifacts(out_dir: Path, split_name: str, rows: Sequence[Dict[str, Any]], pred_rows: List[Dict[str, Any]], y_pred: Sequence[str]) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_three_class(rows, y_pred)
    with (out_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(metrics["confusion_matrix"], index=[f"gold_{x}" for x in THREE_LABELS], columns=[f"pred_{x}" for x in THREE_LABELS]).to_csv(
        out_dir / f"{split_name}_confusion_matrix.csv"
    )
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / f"{split_name}_predictions.csv", index=False)

    topic_rows = []
    if not pred_df.empty:
        for topic, sub in pred_df.groupby("topic"):
            y_true = list(sub["gold_stance"])
            yp = list(sub["pred_stance"])
            rep = classification_report(y_true, yp, labels=THREE_LABELS, output_dict=True, zero_division=0)
            row = {
                "topic": topic,
                "support": len(sub),
                "accuracy": float(accuracy_score(y_true, yp)),
                "macro_f1": float(f1_score(y_true, yp, labels=THREE_LABELS, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, yp, labels=THREE_LABELS, average="weighted", zero_division=0)),
            }
            for lab in THREE_LABELS:
                r = rep.get(lab, {})
                row[f"{lab}_precision"] = r.get("precision", 0.0)
                row[f"{lab}_recall"] = r.get("recall", 0.0)
                row[f"{lab}_f1"] = r.get("f1-score", 0.0)
                row[f"{lab}_support"] = r.get("support", 0.0)
            topic_rows.append(row)
    pd.DataFrame(topic_rows).sort_values("topic").to_csv(out_dir / f"{split_name}_metrics_by_topic.csv", index=False)
    return metrics
