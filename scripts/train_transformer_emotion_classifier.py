from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.classifiers.emotion.transformer_emotion import (
    EMOTION_LABELS,
    EmotionTransformerConfig,
    build_dataset,
    evaluate_emotion_logits,
    make_compute_metrics,
    make_training_args,
    read_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a transformer emotion classifier on 3-class GoEmotions.")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no_safetensors", action="store_true")
    ap.add_argument("--no_force_fp32_model", action="store_true")
    ap.add_argument("--emotion_confidence_threshold", type=float, default=0.0, help="If raw positive/negative confidence below this threshold, output neutral. 0 disables.")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    model_dir = out_dir / "model"
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = EmotionTransformerConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        emotion_confidence_threshold=args.emotion_confidence_threshold,
    )

    label2id = {lab: i for i, lab in enumerate(EMOTION_LABELS)}
    id2label = {i: lab for lab, i in label2id.items()}

    train_rows = read_jsonl(dataset_dir / "train.jsonl")
    dev_rows = read_jsonl(dataset_dir / "dev.jsonl") if (dataset_dir / "dev.jsonl").exists() else []
    test_rows = read_jsonl(dataset_dir / "test.jsonl") if (dataset_dir / "test.jsonl").exists() else []
    if not train_rows:
        raise RuntimeError(f"No training rows found: {dataset_dir / 'train.jsonl'}")

    print("[INFO] loading tokenizer/model:", cfg.model_name)
    print("[INFO] use_safetensors:", not args.no_safetensors)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(EMOTION_LABELS),
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
        use_safetensors=not args.no_safetensors,
    )
    first_dtype = next(model.parameters()).dtype
    print(f"[INFO] model first parameter dtype after load: {first_dtype}")
    if args.fp16 and not args.no_force_fp32_model:
        model.float()
        print(f"[INFO] model first parameter dtype before Trainer: {next(model.parameters()).dtype}")

    print("[INFO] tokenizing")
    train_ds = build_dataset(tokenizer, train_rows, label2id, cfg.max_length)
    dev_ds = build_dataset(tokenizer, dev_rows, label2id, cfg.max_length) if dev_rows else None
    test_ds = build_dataset(tokenizer, test_rows, label2id, cfg.max_length) if test_rows else None

    training_args = make_training_args(str(model_dir), cfg, has_eval=dev_ds is not None)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=make_compute_metrics(id2label),
    )
    sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    meta = {
        "model_role": "transformer_emotion_classifier_goemotions_3class",
        "dataset_dir": str(dataset_dir),
        "config": asdict(cfg),
        "labels": EMOTION_LABELS,
        "label2id": label2id,
        "id2label": id2label,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
    }
    with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[INFO] training")
    trainer.train()
    trainer.save_model(str(model_dir / "final"))
    tokenizer.save_pretrained(str(model_dir / "final"))

    metrics_all: Dict[str, Dict] = {}
    for split, rows, ds in [("train", train_rows, train_ds), ("dev", dev_rows, dev_ds), ("test", test_rows, test_ds)]:
        if not rows or ds is None:
            continue
        print(f"[INFO] predicting {split}")
        pred = trainer.predict(ds)
        metrics = evaluate_emotion_logits(
            rows=rows,
            logits=pred.predictions,
            out_dir=reports_dir,
            split_name=split,
            id2label=id2label,
            emotion_confidence_threshold=args.emotion_confidence_threshold,
        )
        metrics_all[split] = metrics
        print(f"[OK] {split}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}")
    with (reports_dir / "metrics_all.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)
    print(f"[OK] model: {model_dir / 'final'}")
    print(f"[OK] reports: {reports_dir}")


if __name__ == "__main__":
    main()
