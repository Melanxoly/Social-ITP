from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.classifiers.stance.two_stage_transformer import (
    STAGE1_LABELS,
    STAGE2_LABELS,
    TwoStageConfig,
    balance_rows,
    build_dataset,
    combine_two_stage_predictions,
    make_compute_metrics,
    make_training_args,
    read_jsonl,
    stage1_label,
    stage1_threshold_sweep,
    stage2_label,
    softmax,
    trainer_kwargs_with_tokenizer,
    write_eval_artifacts,
)


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _load_model_tokenizer(model_name: str, *, num_labels: int, labels: List[str], use_safetensors: bool = True):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        use_safetensors=use_safetensors,
    )
    return model, tokenizer


def _train_one_stage(
    *,
    stage_name: str,
    model_name: str,
    labels: List[str],
    train_rows: List[dict],
    dev_rows: List[dict],
    label_getter,
    out_dir: Path,
    cfg: TwoStageConfig,
    use_safetensors: bool = True,
):
    from transformers import Trainer

    print(f"[INFO] loading {stage_name} model: {model_name}")
    model, tokenizer = _load_model_tokenizer(model_name, num_labels=len(labels), labels=labels, use_safetensors=use_safetensors)
    if cfg.fp16:
        model.float()
    train_ds = build_dataset(tokenizer, train_rows, labels, label_getter, cfg.max_length)
    dev_ds = build_dataset(tokenizer, dev_rows, labels, label_getter, cfg.max_length) if dev_rows else None
    args = make_training_args(str(out_dir / "checkpoints"), cfg, has_eval=dev_ds is not None)
    trainer = Trainer(
        **trainer_kwargs_with_tokenizer(
            Trainer,
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            tokenizer=tokenizer,
            compute_metrics=make_compute_metrics(labels),
        )
    )
    print(f"[INFO] training {stage_name}")
    trainer.train()
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[OK] saved {stage_name} to {final_dir}")
    return trainer, tokenizer, final_dir


def _predict_logits(trainer, tokenizer, rows: List[dict], labels: List[str], label_getter, max_length: int):
    ds = build_dataset(tokenizer, rows, labels, label_getter, max_length)
    return trainer.predict(ds).predictions


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a two-stage transformer stance classifier: none/has_stance then favor/against.")
    ap.add_argument("--dataset_dir", required=True, help="Directory with train/dev/test JSONL files.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--stage1_model_name", default=None, help="Optional different model for stage1.")
    ap.add_argument("--stage2_model_name", default=None, help="Optional different model for stage2.")
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
    ap.add_argument("--no_safetensors", action="store_true")
    ap.add_argument("--stage1_threshold", type=float, default=0.5, help="Predict has_stance only when p_has_stance >= this threshold; otherwise merge to none.")
    ap.add_argument("--threshold_values", default="0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7", help="Stage1 threshold sweep values.")
    ap.add_argument("--balance_train", action="store_true", help="Oversample labels in each stage's train split.")
    ap.add_argument("--balance_scope", choices=["global", "topic"], default="global")
    ap.add_argument("--balance_max_multiplier", type=float, default=3.0)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = TwoStageConfig(
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
        stage1_threshold=args.stage1_threshold,
        balance_train=args.balance_train,
        balance_scope=args.balance_scope,
        balance_max_multiplier=args.balance_max_multiplier,
    )

    train_rows = read_jsonl(dataset_dir / "train.jsonl")
    dev_rows = read_jsonl(dataset_dir / "dev.jsonl") if (dataset_dir / "dev.jsonl").exists() else []
    test_rows = read_jsonl(dataset_dir / "test.jsonl") if (dataset_dir / "test.jsonl").exists() else []
    if not train_rows:
        raise RuntimeError(f"No train rows found in {dataset_dir / 'train.jsonl'}")

    stage1_train = list(train_rows)
    stage2_train = [r for r in train_rows if str(r.get("stance")) in STAGE2_LABELS]
    stage2_dev = [r for r in dev_rows if str(r.get("stance")) in STAGE2_LABELS]

    balance_meta = {"enabled": False}
    if args.balance_train:
        stage1_train, meta1 = balance_rows(
            stage1_train,
            label_getter=stage1_label,
            scope=args.balance_scope,
            seed=args.seed,
            max_multiplier=args.balance_max_multiplier,
        )
        stage2_train, meta2 = balance_rows(
            stage2_train,
            label_getter=stage2_label,
            scope=args.balance_scope,
            seed=args.seed + 17,
            max_multiplier=args.balance_max_multiplier,
        )
        balance_meta = {"enabled": True, "stage1": meta1, "stage2": meta2}
        print("[INFO] train balancing enabled")
        print(json.dumps({"stage1_final_rows": len(stage1_train), "stage2_final_rows": len(stage2_train)}, ensure_ascii=False, indent=2))

    config_meta = {
        "config": asdict(cfg),
        "dataset_dir": str(dataset_dir),
        "stage1_model_name": args.stage1_model_name or args.model_name,
        "stage2_model_name": args.stage2_model_name or args.model_name,
        "rows": {
            "train_original": len(train_rows),
            "dev": len(dev_rows),
            "test": len(test_rows),
            "stage1_train": len(stage1_train),
            "stage2_train": len(stage2_train),
            "stage2_dev": len(stage2_dev),
        },
        "balance": balance_meta,
    }
    with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_meta, f, ensure_ascii=False, indent=2)

    use_safetensors = not args.no_safetensors
    stage1_trainer, stage1_tok, stage1_final = _train_one_stage(
        stage_name="stage1_none_vs_has_stance",
        model_name=args.stage1_model_name or args.model_name,
        labels=STAGE1_LABELS,
        train_rows=stage1_train,
        dev_rows=dev_rows,
        label_getter=stage1_label,
        out_dir=out_dir / "stage1_model",
        cfg=cfg,
        use_safetensors=use_safetensors,
    )
    stage2_trainer, stage2_tok, stage2_final = _train_one_stage(
        stage_name="stage2_favor_vs_against",
        model_name=args.stage2_model_name or args.model_name,
        labels=STAGE2_LABELS,
        train_rows=stage2_train,
        dev_rows=stage2_dev,
        label_getter=stage2_label,
        out_dir=out_dir / "stage2_model",
        cfg=cfg,
        use_safetensors=use_safetensors,
    )

    metrics_all: Dict[str, dict] = {}
    for split_name, rows in [("train", train_rows), ("dev", dev_rows), ("test", test_rows)]:
        if not rows:
            continue
        print(f"[INFO] evaluating {split_name}")
        s1_logits = _predict_logits(stage1_trainer, stage1_tok, rows, STAGE1_LABELS, stage1_label, cfg.max_length)
        p_has = softmax(s1_logits)[:, STAGE1_LABELS.index("has_stance")]
        sweep = stage1_threshold_sweep(rows, p_has, _parse_floats(args.threshold_values))
        sweep.to_csv(reports_dir / f"{split_name}_stage1_threshold_sweep.csv", index=False)

        s2_logits = _predict_logits(stage2_trainer, stage2_tok, rows, STAGE2_LABELS, lambda r: "favor", cfg.max_length)
        pred_rows, final_pred = combine_two_stage_predictions(
            rows,
            stage1_logits=s1_logits,
            stage2_logits=s2_logits,
            stage1_threshold=args.stage1_threshold,
        )
        metrics = write_eval_artifacts(reports_dir, split_name, rows, pred_rows, final_pred)
        metrics_all[split_name] = metrics
        print(
            f"[OK] {split_name}: acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"none_to_stance={metrics['none_to_stance_rate']:.4f} stance_to_none={metrics['stance_to_none_rate']:.4f}"
        )

    with (reports_dir / "metrics_all.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)
    print("[OK] two-stage training/evaluation finished")
    print(f"stage1 model: {stage1_final}")
    print(f"stage2 model: {stage2_final}")
    print(f"reports: {reports_dir}")


if __name__ == "__main__":
    main()
