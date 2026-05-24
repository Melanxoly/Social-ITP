from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.classifiers.stance.dataset import read_jsonl
from social_itp.classifiers.stance.transformer_finetune import (
    build_dataset,
    evaluate_stage1_predictions,
    evaluate_three_class_predictions,
    label_names_for_task,
)


def _parse_floats(s: str):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a fine-tuned transformer stance classifier without retraining.")
    ap.add_argument("--task", choices=["three_class", "stage1"], default="three_class")
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--model_dir", required=True, help="Path to saved model directory, e.g. outputs/.../model/final")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_name", default="eval")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=16)
    ap.add_argument("--threshold_values", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    ap.add_argument("--prediction_threshold", type=float, default=0.5)
    ap.add_argument("--stance_confidence_threshold", type=float, default=0.0)
    ap.add_argument("--none_confidence_policy", choices=["keep"], default="keep")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.dataset_jsonl))
    if not rows:
        raise RuntimeError(f"No rows found in {args.dataset_jsonl}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = label_names_for_task(args.task)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    ds = build_dataset(tokenizer, rows, args.task, label2id, args.max_length)

    targs = TrainingArguments(
        output_dir=str(out_dir / "tmp_trainer"),
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        report_to="none",
    )
    kwargs = dict(model=model, args=targs)
    sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in sig:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**kwargs)
    pred = trainer.predict(ds)
    logits = pred.predictions

    if args.task == "three_class":
        metrics = evaluate_three_class_predictions(
            rows=rows,
            logits=logits,
            out_dir=out_dir,
            split_name=args.split_name,
            id2label=id2label,
            stance_confidence_threshold=args.stance_confidence_threshold,
            none_confidence_policy=args.none_confidence_policy,
        )
        print(f"[OK] {args.split_name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}")
    else:
        metrics = evaluate_stage1_predictions(
            rows=rows,
            logits=logits,
            out_dir=out_dir,
            split_name=args.split_name,
            id2label=id2label,
            thresholds=_parse_floats(args.threshold_values),
            prediction_threshold=args.prediction_threshold,
        )
        bb = metrics.get("best_by_boundary_max", {})
        print(f"[OK] {args.split_name}: roc_auc={metrics.get('roc_auc')} best_boundary_thr={bb.get('threshold')} best_boundary_max={bb.get('boundary_max')}")
    with (out_dir / f"{args.split_name}_eval_metrics_all.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
