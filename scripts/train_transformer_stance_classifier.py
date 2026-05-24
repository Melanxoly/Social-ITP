from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List
import inspect

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.classifiers.stance.dataset import read_jsonl
from social_itp.classifiers.stance.augmentation import augment_training_rows
from social_itp.classifiers.stance.transformer_finetune import (
    TransformerFinetuneConfig,
    build_dataset,
    evaluate_stage1_predictions,
    evaluate_three_class_predictions,
    label_names_for_task,
    make_compute_metrics,
    make_training_args,
)


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _flatten_stage1_summary(split_name: str, metrics: dict) -> dict:
    row = {
        "split": split_name,
        "rows": metrics.get("rows"),
        "none_count": metrics.get("none_count"),
        "stance_count": metrics.get("stance_count"),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
    }
    for key in ["best_by_boundary_max", "best_by_balanced_accuracy", "best_by_macro_f1"]:
        best = metrics.get(key) or {}
        prefix = key.replace("best_by_", "best_")
        for k, v in best.items():
            if k in {
                "threshold",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "none_to_stance_rate",
                "stance_to_none_rate",
                "boundary_avg",
                "boundary_max",
                "boundary_weighted",
                "none_f1",
                "stance_f1",
            }:
                row[f"{prefix}_{k}"] = v
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune a transformer encoder for stance classification. Supports three_class and stage1."
    )
    ap.add_argument("--dataset_dir", required=True, help="Directory containing train/dev/test JSONL files.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--task", choices=["three_class", "stage1"], default="three_class")
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--max_length", type=int, default=384)
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
    # Training-set augmentation / oversampling. Applied only to train split.
    ap.add_argument("--augment_train", action="store_true", help="Oversample minority training classes. Evaluation splits are never augmented.")
    ap.add_argument("--augment_scope", choices=["topic", "global"], default="topic", help="Balance labels inside each topic or globally.")
    ap.add_argument("--augment_target", choices=["max", "median", "none"], default="max", help="Target count within each augmentation scope.")
    ap.add_argument("--augment_methods", default="duplicate,prefix", help="Comma-separated methods: duplicate,prefix,casing,synonym,mixed")
    ap.add_argument("--augment_prefixes", default="oh,|well,|actually,|literally,|to be fair,|honestly,|tbh,|imo,", help="Pipe-separated safe prefixes for prefix augmentation.")
    ap.add_argument("--augment_synonyms", default="", help="Optional pipe-separated substitutions such as you=>u|to be honest=>tbh. Empty uses built-in safe defaults.")
    ap.add_argument("--augment_case_modes", default="lower,upper,title", help="Comma-separated case modes for casing augmentation: lower,upper,title.")
    ap.add_argument("--augment_max_multiplier", type=float, default=3.0, help="Cap final per-label count to original_count * this multiplier. Set <=0 to disable cap.")
    ap.add_argument("--augment_final_multiplier", type=float, default=1.0, help="Multiplier applied after balancing target. Example: counts 80,200,300 with target=max and final_multiplier=3 -> desired 900,900,900 before cap.")
    ap.add_argument("--augment_chain_min", type=int, default=1, help="For method=mixed, minimum number of transforms to compose.")
    ap.add_argument("--augment_chain_max", type=int, default=2, help="For method=mixed, maximum number of transforms to compose.")
    # Conservative decoding for three_class evaluation.
    ap.add_argument("--stance_confidence_threshold", type=float, default=0.0, help="For three_class: if raw favor/against confidence is below this threshold, output none. 0 disables.")
    ap.add_argument("--none_confidence_policy", choices=["keep"], default="keep", help="For low-confidence raw none predictions, keep none and only flag them.")
    ap.add_argument("--threshold_values", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7", help="Only used for stage1.")
    ap.add_argument("--prediction_threshold", type=float, default=0.5, help="Only used for stage1 prediction CSV.")
    ap.add_argument("--no_safetensors", action="store_true", help="Disable safetensors loading. Not recommended for torch<2.6.")
    ap.add_argument("--no_force_fp32_model", action="store_true", help="Do not force trainable model parameters back to FP32 before AMP fp16 training. Not recommended.")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    model_dir = out_dir / "model"
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = TransformerFinetuneConfig(
        model_name=args.model_name,
        task=args.task,
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
        stance_confidence_threshold=args.stance_confidence_threshold,
        none_confidence_policy=args.none_confidence_policy,
    )

    labels = label_names_for_task(args.task)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    train_rows = read_jsonl(dataset_dir / "train.jsonl")
    dev_rows = read_jsonl(dataset_dir / "dev.jsonl") if (dataset_dir / "dev.jsonl").exists() else []
    test_rows = read_jsonl(dataset_dir / "test.jsonl") if (dataset_dir / "test.jsonl").exists() else []
    if not train_rows:
        raise RuntimeError(f"No training rows found in {dataset_dir / 'train.jsonl'}")

    original_train_rows_count = len(train_rows)
    train_rows, augmentation_meta = augment_training_rows(
        train_rows,
        task=args.task,
        enabled=args.augment_train,
        scope=args.augment_scope,
        target=args.augment_target,
        methods=args.augment_methods,
        prefixes=args.augment_prefixes,
        synonyms=args.augment_synonyms,
        case_modes=args.augment_case_modes,
        seed=args.seed,
        max_multiplier=args.augment_max_multiplier,
        final_multiplier=args.augment_final_multiplier,
        chain_min=args.augment_chain_min,
        chain_max=args.augment_chain_max,
    )
    if args.augment_train:
        print("[INFO] training augmentation enabled")
        print(json.dumps({
            "original_train_rows": original_train_rows_count,
            "final_train_rows": len(train_rows),
            "added": len(train_rows) - original_train_rows_count,
            "scope": args.augment_scope,
            "target": args.augment_target,
            "final_multiplier": args.augment_final_multiplier,
            "methods": args.augment_methods,
            "max_multiplier": args.augment_max_multiplier,
        }, ensure_ascii=False, indent=2))

    use_safetensors = not args.no_safetensors
    print("[INFO] loading tokenizer/model:", cfg.model_name)
    print("[INFO] use_safetensors:", use_safetensors)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(labels),
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
        use_safetensors=use_safetensors,
    )

    # Important for Trainer fp16/AMP training:
    # trainable parameters should remain FP32. AMP will cast operations as needed.
    # If the model is already half precision, GradScaler may fail with
    # ValueError: Attempting to unscale FP16 gradients.
    first_param_dtype = next(model.parameters()).dtype
    print(f"[INFO] model first parameter dtype after load: {first_param_dtype}")
    if args.fp16 and not args.no_force_fp32_model:
        if first_param_dtype != torch.float32:
            print("[INFO] fp16=True but model parameters are not FP32; casting model.float() for AMP compatibility")
        else:
            print("[INFO] fp16=True; keeping model parameters in FP32 for AMP compatibility")
        model.float()
        print(f"[INFO] model first parameter dtype before Trainer: {next(model.parameters()).dtype}")

    print("[INFO] tokenizing splits")
    train_ds = build_dataset(tokenizer, train_rows, args.task, label2id, cfg.max_length)
    dev_ds = build_dataset(tokenizer, dev_rows, args.task, label2id, cfg.max_length) if dev_rows else None
    test_ds = build_dataset(tokenizer, test_rows, args.task, label2id, cfg.max_length) if test_rows else None

    training_args = make_training_args(str(model_dir), cfg, has_eval=dev_ds is not None)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=make_compute_metrics(args.task, id2label),
    )
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_sig:
        # Newer Transformers versions replaced/deprecated `tokenizer` with `processing_class`.
        trainer_kwargs["processing_class"] = tokenizer
        print("[INFO] Trainer uses processing_class")
    elif "tokenizer" in trainer_sig:
        # Older Transformers versions still use `tokenizer`.
        trainer_kwargs["tokenizer"] = tokenizer
        print("[INFO] Trainer uses tokenizer")
    else:
        # Some future versions may not need either argument for this script.
        print("[INFO] Trainer accepts neither tokenizer nor processing_class; skipping processor argument")

    trainer = Trainer(**trainer_kwargs)

    meta = {
        "model_role": "transformer_finetuned_stance_classifier",
        "dataset_dir": str(dataset_dir),
        "config": asdict(cfg),
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "use_safetensors": use_safetensors,
        "train_rows": len(train_rows),
        "original_train_rows": original_train_rows_count,
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
        "augmentation": augmentation_meta,
    }
    with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[INFO] training")
    trainer.train()
    print("[INFO] saving best/final model")
    trainer.save_model(str(model_dir / "final"))
    tokenizer.save_pretrained(str(model_dir / "final"))

    metrics_all: Dict[str, Dict] = {}
    stage1_summary_rows = []
    thresholds = _parse_floats(args.threshold_values)
    for split_name, rows, ds in [("train", train_rows, train_ds), ("dev", dev_rows, dev_ds), ("test", test_rows, test_ds)]:
        if not rows or ds is None:
            continue
        print(f"[INFO] predicting {split_name}")
        pred = trainer.predict(ds)
        logits = pred.predictions
        if args.task == "three_class":
            metrics = evaluate_three_class_predictions(
                rows=rows,
                logits=logits,
                out_dir=reports_dir,
                split_name=split_name,
                id2label=id2label,
                stance_confidence_threshold=args.stance_confidence_threshold,
                none_confidence_policy=args.none_confidence_policy,
            )
        else:
            metrics = evaluate_stage1_predictions(
                rows=rows,
                logits=logits,
                out_dir=reports_dir,
                split_name=split_name,
                id2label=id2label,
                thresholds=thresholds,
                prediction_threshold=args.prediction_threshold,
            )
            stage1_summary_rows.append(_flatten_stage1_summary(split_name, metrics))
        metrics_all[split_name] = metrics
        if args.task == "three_class":
            print(f"[OK] {split_name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}")
        else:
            bb = metrics.get("best_by_boundary_max", {})
            print(f"[OK] {split_name}: roc_auc={metrics.get('roc_auc')} best_boundary_thr={bb.get('threshold')} best_boundary_max={bb.get('boundary_max')}")

    with (reports_dir / "metrics_all.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)
    if args.task == "stage1":
        pd.DataFrame(stage1_summary_rows).to_csv(reports_dir / "stage1_summary.csv", index=False)

    print("[OK] finished transformer fine-tuning")
    print(f"model: {model_dir / 'final'}")
    print(f"reports: {reports_dir}")


if __name__ == "__main__":
    main()
