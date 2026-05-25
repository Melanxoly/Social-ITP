from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset

EMOTION_LABELS = ["negative", "neutral", "positive"]


@dataclass
class EmotionTransformerConfig:
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 256
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
    emotion_confidence_threshold: float = 0.0


class TextDataset(Dataset):
    def __init__(self, encodings: Dict[str, Any], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rows_to_texts(rows: List[Dict[str, Any]]) -> List[str]:
    return [str(r.get("input_text") or r.get("text") or "") for r in rows]


def rows_to_labels(rows: List[Dict[str, Any]]) -> List[str]:
    return [str(r.get("emotion")) for r in rows]


def build_dataset(tokenizer: Any, rows: List[Dict[str, Any]], label2id: Dict[str, int], max_length: int) -> TextDataset:
    texts = rows_to_texts(rows)
    labels = [label2id[x] for x in rows_to_labels(rows)]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return TextDataset(enc, labels)


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def make_compute_metrics(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        y_true = [id2label[int(x)] for x in labels]
        y_pred = [id2label[int(x)] for x in preds]
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=EMOTION_LABELS, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=EMOTION_LABELS, average="weighted", zero_division=0)),
        }
    return compute_metrics


def decode_emotion_predictions(
    logits: np.ndarray,
    id2label: Dict[int, str],
    *,
    emotion_confidence_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    probs = softmax(logits)
    raw_ids = probs.argmax(axis=1)
    out: List[Dict[str, Any]] = []
    th = float(emotion_confidence_threshold or 0.0)
    for raw_id, prob in zip(raw_ids, probs):
        raw = id2label[int(raw_id)]
        conf = float(prob[int(raw_id)])
        final = raw
        conservative_to_neutral = False
        low_confidence_neutral = False
        if raw in {"negative", "positive"} and th > 0 and conf < th:
            final = "neutral"
            conservative_to_neutral = True
        elif raw == "neutral" and th > 0 and conf < th:
            # Keep neutral by default. Low-confidence neutral can be reviewed but not flipped to pos/neg.
            low_confidence_neutral = True
        out.append({
            "raw_pred_emotion": raw,
            "final_pred_emotion": final,
            "raw_confidence": conf,
            "final_confidence": conf if final == raw else float(prob[EMOTION_LABELS.index("neutral")]),
            "prob_negative": float(prob[EMOTION_LABELS.index("negative")]),
            "prob_neutral": float(prob[EMOTION_LABELS.index("neutral")]),
            "prob_positive": float(prob[EMOTION_LABELS.index("positive")]),
            "conservative_to_neutral": conservative_to_neutral,
            "low_confidence_neutral": low_confidence_neutral,
            "emotion_confidence_threshold": th,
        })
    return out


def _metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=EMOTION_LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=EMOTION_LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=EMOTION_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=EMOTION_LABELS, average="weighted", zero_division=0)),
        "labels": EMOTION_LABELS,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_emotion_logits(
    *,
    rows: List[Dict[str, Any]],
    logits: np.ndarray,
    out_dir: Path,
    split_name: str,
    id2label: Dict[int, str],
    emotion_confidence_threshold: float = 0.0,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded = decode_emotion_predictions(logits, id2label, emotion_confidence_threshold=emotion_confidence_threshold)
    y_true = rows_to_labels(rows)
    raw_pred = [d["raw_pred_emotion"] for d in decoded]
    final_pred = [d["final_pred_emotion"] for d in decoded]

    raw_metrics = _metrics(y_true, raw_pred)
    final_metrics = _metrics(y_true, final_pred)
    final_metrics["raw_metrics"] = raw_metrics
    final_metrics["emotion_confidence_threshold"] = float(emotion_confidence_threshold or 0.0)
    final_metrics["num_conservative_to_neutral"] = int(sum(d["conservative_to_neutral"] for d in decoded))
    final_metrics["num_low_confidence_neutral"] = int(sum(d["low_confidence_neutral"] for d in decoded))

    with (out_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    with (out_dir / f"{split_name}_raw_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(raw_metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame(final_metrics["confusion_matrix"], index=[f"gold_{x}" for x in EMOTION_LABELS], columns=[f"pred_{x}" for x in EMOTION_LABELS]).to_csv(out_dir / f"{split_name}_confusion_matrix.csv")
    pd.DataFrame(raw_metrics["confusion_matrix"], index=[f"gold_{x}" for x in EMOTION_LABELS], columns=[f"pred_{x}" for x in EMOTION_LABELS]).to_csv(out_dir / f"{split_name}_raw_confusion_matrix.csv")

    pred_rows = []
    for r, d in zip(rows, decoded):
        pred_rows.append({
            "row_id": r.get("row_id"),
            "gold_emotion": r.get("emotion"),
            "pred_emotion": d["final_pred_emotion"],
            "raw_pred_emotion": d["raw_pred_emotion"],
            "correct": r.get("emotion") == d["final_pred_emotion"],
            "raw_correct": r.get("emotion") == d["raw_pred_emotion"],
            "raw_confidence": d["raw_confidence"],
            "confidence": d["final_confidence"],
            "prob_negative": d["prob_negative"],
            "prob_neutral": d["prob_neutral"],
            "prob_positive": d["prob_positive"],
            "conservative_to_neutral": d["conservative_to_neutral"],
            "low_confidence_neutral": d["low_confidence_neutral"],
            "fine_emotions": r.get("fine_emotions"),
            "source_split": r.get("source_split"),
            "text": r.get("text"),
            "input_text": r.get("input_text"),
        })
    pd.DataFrame(pred_rows).to_csv(out_dir / f"{split_name}_predictions.csv", index=False)
    return final_metrics


def make_training_args(output_dir: str, cfg: EmotionTransformerConfig, has_eval: bool = True):
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
        common.update(dict(save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True))
        try:
            return TrainingArguments(evaluation_strategy="epoch", **common)
        except TypeError:
            return TrainingArguments(eval_strategy="epoch", **common)
    return TrainingArguments(save_strategy="epoch", **common)
