from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

THREE_CLASS_LABELS = ["favor", "against", "none"]
STAGE1_LABELS = ["none", "has_stance"]


@dataclass
class TransformerFinetuneConfig:
    model_name: str = "microsoft/deberta-v3-base"
    task: str = "three_class"  # three_class | stage1
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
    stance_confidence_threshold: float = 0.0
    none_confidence_policy: str = "keep"


def label_names_for_task(task: str) -> List[str]:
    if task == "three_class":
        return THREE_CLASS_LABELS
    if task == "stage1":
        return STAGE1_LABELS
    raise ValueError(f"Unsupported task={task}. Use three_class or stage1.")


def label_from_row(row: Dict[str, Any], task: str) -> str:
    stance = str(row.get("stance"))
    if task == "three_class":
        if stance not in THREE_CLASS_LABELS:
            raise ValueError(f"Unexpected stance label: {stance}")
        return stance
    if task == "stage1":
        return "none" if stance == "none" else "has_stance"
    raise ValueError(f"Unsupported task={task}")


def rows_to_texts(rows: List[Dict[str, Any]]) -> List[str]:
    return [str(r.get("input_text") or r.get("comment_text") or "") for r in rows]


def rows_to_labels(rows: List[Dict[str, Any]], task: str) -> List[str]:
    return [label_from_row(r, task) for r in rows]


class TransformerTextDataset(Dataset):
    def __init__(self, encodings: Dict[str, Any], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_dataset(tokenizer: Any, rows: List[Dict[str, Any]], task: str, label2id: Dict[str, int], max_length: int) -> TransformerTextDataset:
    texts = rows_to_texts(rows)
    labels = [label2id[x] for x in rows_to_labels(rows, task)]
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return TransformerTextDataset(enc, labels)


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_basic_metrics_from_ids(pred_ids: np.ndarray, label_ids: np.ndarray, id2label: Dict[int, str]) -> Dict[str, float]:
    labels = [id2label[i] for i in sorted(id2label.keys())]
    y_true = [id2label[int(x)] for x in label_ids]
    y_pred = [id2label[int(x)] for x in pred_ids]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def make_compute_metrics(task: str, id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        m = compute_basic_metrics_from_ids(preds, labels, id2label)
        if task == "stage1":
            probs = softmax(logits)
            # Label ids are STAGE1_LABELS: none=0, has_stance=1 by default.
            pos_id = None
            for i, lab in id2label.items():
                if lab == "has_stance":
                    pos_id = i
                    break
            if pos_id is not None and len(set(labels.tolist() if hasattr(labels, 'tolist') else list(labels))) > 1:
                y_int = np.array([1 if id2label[int(x)] == "has_stance" else 0 for x in labels], dtype=int)
                p_pos = probs[:, pos_id]
                try:
                    m["roc_auc"] = float(roc_auc_score(y_int, p_pos))
                    m["average_precision"] = float(average_precision_score(y_int, p_pos))
                except Exception:
                    pass
        return m

    return compute_metrics


def prediction_details_from_logits(logits: np.ndarray, id2label: Dict[int, str]) -> List[Dict[str, Any]]:
    probs = softmax(logits)
    pred_ids = np.argmax(probs, axis=1)
    out: List[Dict[str, Any]] = []
    for pred_id, prob in zip(pred_ids, probs):
        prob_map = {id2label[i]: float(prob[i]) for i in range(len(prob))}
        out.append(
            {
                "pred_label": id2label[int(pred_id)],
                "probabilities": prob_map,
                "confidence": float(prob[int(pred_id)]),
            }
        )
    return out


def decode_three_class_details(
    details: List[Dict[str, Any]],
    *,
    stance_confidence_threshold: float = 0.0,
    none_confidence_policy: str = "keep",
) -> List[Dict[str, Any]]:
    """Decode raw three-class probabilities with conservative stance fallback.

    If the raw prediction is favor/against and its confidence is below
    stance_confidence_threshold, the final label is forced to none.

    If raw prediction is none but confidence is low, the default policy is still
    to keep none. This matches the conservative assumption that predicting stance
    is riskier than predicting none. Low-confidence none rows are only flagged
    for possible manual/Qwen review.
    """
    out: List[Dict[str, Any]] = []
    th = float(stance_confidence_threshold or 0.0)
    for d in details:
        probs = d.get("probabilities") or {}
        raw = str(d.get("pred_label"))
        raw_conf = float(d.get("confidence") or 0.0)
        final = raw
        conservative_to_none = False
        low_confidence_none = False
        if raw in {"favor", "against"} and th > 0 and raw_conf < th:
            final = "none"
            conservative_to_none = True
        elif raw == "none" and th > 0 and raw_conf < th:
            # Keep none by default; flag only.
            low_confidence_none = True
            final = "none"
        out.append({
            **d,
            "raw_pred_label": raw,
            "raw_confidence": raw_conf,
            "final_pred_label": final,
            "final_confidence": raw_conf if final == raw else float(probs.get("none", 0.0)),
            "conservative_to_none": conservative_to_none,
            "low_confidence_none": low_confidence_none,
            "stance_confidence_threshold": th,
            "none_confidence_policy": none_confidence_policy,
        })
    return out


def _three_class_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def _write_three_class_artifacts(
    *,
    out_dir: Path,
    split_name: str,
    rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    suffix: str = "",
) -> Dict[str, Any]:
    metrics = _three_class_metrics(y_true, y_pred, labels)
    name = f"{split_name}{suffix}"
    with (out_dir / f"{name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(metrics["confusion_matrix"], index=[f"gold_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]).to_csv(
        out_dir / f"{name}_confusion_matrix.csv"
    )
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / f"{name}_predictions.csv", index=False)
    topic_rows = []
    if not pred_df.empty and "topic" in pred_df.columns:
        for topic, sub in pred_df.groupby("topic"):
            yt = list(sub["gold_stance"])
            yp = list(sub["pred_stance"])
            topic_m = classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0)
            row = {
                "topic": topic,
                "support": len(sub),
                "accuracy": float(accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)),
            }
            for lab in labels:
                rep = topic_m.get(lab, {})
                row[f"{lab}_precision"] = rep.get("precision", 0.0)
                row[f"{lab}_recall"] = rep.get("recall", 0.0)
                row[f"{lab}_f1"] = rep.get("f1-score", 0.0)
                row[f"{lab}_support"] = rep.get("support", 0.0)
            topic_rows.append(row)
    pd.DataFrame(topic_rows).sort_values("topic").to_csv(out_dir / f"{name}_metrics_by_topic.csv", index=False)
    return metrics


def evaluate_three_class_predictions(
    *,
    rows: List[Dict[str, Any]],
    logits: np.ndarray,
    out_dir: Path,
    split_name: str,
    id2label: Dict[int, str],
    stance_confidence_threshold: float = 0.0,
    none_confidence_policy: str = "keep",
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = THREE_CLASS_LABELS
    raw_details = prediction_details_from_logits(logits, id2label)
    final_details = decode_three_class_details(
        raw_details,
        stance_confidence_threshold=stance_confidence_threshold,
        none_confidence_policy=none_confidence_policy,
    )
    y_true = [str(r.get("stance")) for r in rows]
    raw_pred = [d["raw_pred_label"] for d in final_details]
    final_pred = [d["final_pred_label"] for d in final_details]

    pred_rows_final = []
    pred_rows_raw = []
    for r, raw, final, d in zip(rows, raw_pred, final_pred, final_details):
        probs = d["probabilities"]
        base = {
            "row_id": r.get("row_id"),
            "thread_id": r.get("thread_id"),
            "topic": r.get("topic"),
            "target_entity": r.get("target_entity"),
            "comment_id": r.get("comment_id"),
            "gold_stance": r.get("stance"),
            "raw_pred_stance": raw,
            "final_pred_stance": final,
            "raw_confidence": d.get("raw_confidence"),
            "confidence": d.get("final_confidence"),
            "prob_favor": probs.get("favor"),
            "prob_against": probs.get("against"),
            "prob_none": probs.get("none"),
            "conservative_to_none": d.get("conservative_to_none"),
            "low_confidence_none": d.get("low_confidence_none"),
            "stance_confidence_threshold": stance_confidence_threshold,
            "post_title": r.get("post_title"),
            "parent_text": r.get("parent_text"),
            "comment_text": r.get("comment_text"),
            "input_text": r.get("input_text"),
        }
        pred_rows_final.append({**base, "pred_stance": final, "correct": r.get("stance") == final})
        pred_rows_raw.append({**base, "pred_stance": raw, "correct": r.get("stance") == raw})

    # Always write raw artifacts for comparison.
    raw_metrics = _write_three_class_artifacts(
        out_dir=out_dir, split_name=split_name, rows=rows, pred_rows=pred_rows_raw, y_true=y_true, y_pred=raw_pred, labels=labels, suffix="_raw"
    )
    final_metrics = _write_three_class_artifacts(
        out_dir=out_dir, split_name=split_name, rows=rows, pred_rows=pred_rows_final, y_true=y_true, y_pred=final_pred, labels=labels, suffix=""
    )
    final_metrics["raw_metrics"] = raw_metrics
    final_metrics["stance_confidence_threshold"] = float(stance_confidence_threshold or 0.0)
    final_metrics["none_confidence_policy"] = none_confidence_policy
    final_metrics["num_conservative_to_none"] = int(sum(1 for d in final_details if d.get("conservative_to_none")))
    final_metrics["num_low_confidence_none"] = int(sum(1 for d in final_details if d.get("low_confidence_none")))
    # Rewrite final metrics with extra metadata.
    with (out_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    return final_metrics


def evaluate_stage1_predictions(
    *,
    rows: List[Dict[str, Any]],
    logits: np.ndarray,
    out_dir: Path,
    split_name: str,
    id2label: Dict[int, str],
    thresholds: Iterable[float],
    prediction_threshold: float = 0.5,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    details = prediction_details_from_logits(logits, id2label)
    probs = softmax(logits)
    pos_id = [i for i, lab in id2label.items() if lab == "has_stance"][0]
    p_pos = probs[:, pos_id]
    y_true = ["none" if str(r.get("stance")) == "none" else "has_stance" for r in rows]
    y_int = np.array([1 if y == "has_stance" else 0 for y in y_true], dtype=int)

    sweep_rows = []
    roc_auc = None
    avg_precision = None
    if len(set(y_int.tolist())) > 1:
        roc_auc = float(roc_auc_score(y_int, p_pos))
        avg_precision = float(average_precision_score(y_int, p_pos))
    for th in thresholds:
        pred_int = (p_pos >= float(th)).astype(int)
        pred_lab = ["has_stance" if x else "none" for x in pred_int]
        cm = confusion_matrix(y_int, pred_int, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
        none_to_stance = fp / max(1, tn + fp)
        stance_to_none = fn / max(1, fn + tp)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, pred_lab, labels=STAGE1_LABELS, zero_division=0
        )
        row = {
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
            "none_f1": float(f1[0]),
            "stance_f1": float(f1[1]),
            "roc_auc": roc_auc,
            "average_precision": avg_precision,
            "tn_none_as_none": tn,
            "fp_none_as_stance": fp,
            "fn_stance_as_none": fn,
            "tp_stance_as_stance": tp,
        }
        sweep_rows.append(row)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_dir / f"{split_name}_threshold_sweep.csv", index=False)
    metrics = {
        "rows": len(rows),
        "none_count": int(sum(1 for y in y_true if y == "none")),
        "stance_count": int(sum(1 for y in y_true if y == "has_stance")),
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "threshold_sweep": sweep_rows,
    }
    if len(sweep_df):
        metrics["best_by_boundary_max"] = sweep_df.sort_values(["boundary_max", "boundary_avg", "threshold"]).iloc[0].to_dict()
        metrics["best_by_balanced_accuracy"] = sweep_df.sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False]).iloc[0].to_dict()
        metrics["best_by_macro_f1"] = sweep_df.sort_values(["macro_f1", "balanced_accuracy"], ascending=[False, False]).iloc[0].to_dict()
    with (out_dir / f"{split_name}_stage1_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_lab = ["has_stance" if x >= prediction_threshold else "none" for x in p_pos]
    pred_rows = []
    for r, gold, pred, pp in zip(rows, y_true, pred_lab, p_pos):
        pred_rows.append(
            {
                "row_id": r.get("row_id"),
                "thread_id": r.get("thread_id"),
                "topic": r.get("topic"),
                "comment_id": r.get("comment_id"),
                "gold_stance": r.get("stance"),
                "gold_binary": gold,
                "pred_binary": pred,
                "p_has_stance": float(pp),
                "correct_binary": gold == pred,
                "comment_text": r.get("comment_text"),
                "input_text": r.get("input_text"),
            }
        )
    pd.DataFrame(pred_rows).to_csv(out_dir / f"{split_name}_predictions_thr{prediction_threshold:.2f}.csv", index=False)
    return metrics


def make_training_args(output_dir: str, cfg: TransformerFinetuneConfig, has_eval: bool = True):
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
    )
    if has_eval:
        common.update(
            dict(
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="macro_f1",
                greater_is_better=True,
            )
        )
        # transformers has used both `evaluation_strategy` and newer `eval_strategy` names.
        try:
            return TrainingArguments(evaluation_strategy="epoch", **common)
        except TypeError:
            return TrainingArguments(eval_strategy="epoch", **common)
    return TrainingArguments(save_strategy="epoch", **common)
