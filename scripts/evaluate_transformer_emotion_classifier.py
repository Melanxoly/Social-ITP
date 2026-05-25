from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.classifiers.emotion.transformer_emotion import (
    EMOTION_LABELS,
    build_dataset,
    evaluate_emotion_logits,
    read_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a saved emotion classifier without retraining.")
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_name", default="eval")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=16)
    ap.add_argument("--emotion_confidence_threshold", type=float, default=0.0)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.dataset_jsonl))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label2id = {lab: i for i, lab in enumerate(EMOTION_LABELS)}
    id2label = {i: lab for lab, i in label2id.items()}

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    ds = build_dataset(tokenizer, rows, label2id, args.max_length)
    targs = TrainingArguments(output_dir=str(out_dir / "tmp_trainer"), per_device_eval_batch_size=args.per_device_eval_batch_size, report_to="none")
    kwargs = dict(model=model, args=targs)
    sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in sig:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**kwargs)
    pred = trainer.predict(ds)
    metrics = evaluate_emotion_logits(
        rows=rows,
        logits=pred.predictions,
        out_dir=out_dir,
        split_name=args.split_name,
        id2label=id2label,
        emotion_confidence_threshold=args.emotion_confidence_threshold,
    )
    with (out_dir / f"{args.split_name}_eval_metrics_all.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] {args.split_name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
