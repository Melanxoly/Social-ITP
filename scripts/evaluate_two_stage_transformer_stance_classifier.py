from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import List

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.classifiers.stance.two_stage_transformer import (
    STAGE1_LABELS,
    STAGE2_LABELS,
    build_dataset,
    combine_two_stage_predictions,
    read_jsonl,
    stage1_label,
    stage1_threshold_sweep,
    softmax,
    write_eval_artifacts,
)


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _trainer(model_dir: str, out_dir: Path, batch_size: int):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    args = TrainingArguments(output_dir=str(out_dir / "tmp_trainer"), per_device_eval_batch_size=batch_size, report_to="none")
    kwargs = dict(model=model, args=args)
    sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in sig:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs), tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained two-stage stance classifier.")
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--stage1_model_dir", required=True)
    ap.add_argument("--stage2_model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_name", default="test")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=16)
    ap.add_argument("--stage1_threshold", type=float, default=0.5)
    ap.add_argument("--threshold_values", default="0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.dataset_jsonl))
    s1_tr, s1_tok = _trainer(args.stage1_model_dir, out_dir / "stage1_eval", args.per_device_eval_batch_size)
    s2_tr, s2_tok = _trainer(args.stage2_model_dir, out_dir / "stage2_eval", args.per_device_eval_batch_size)

    s1_ds = build_dataset(s1_tok, rows, STAGE1_LABELS, stage1_label, args.max_length)
    s1_logits = s1_tr.predict(s1_ds).predictions
    p_has = softmax(s1_logits)[:, STAGE1_LABELS.index("has_stance")]
    sweep = stage1_threshold_sweep(rows, p_has, _parse_floats(args.threshold_values))
    sweep.to_csv(out_dir / f"{args.split_name}_stage1_threshold_sweep.csv", index=False)

    s2_ds = build_dataset(s2_tok, rows, STAGE2_LABELS, lambda r: "favor", args.max_length)
    s2_logits = s2_tr.predict(s2_ds).predictions
    pred_rows, final_pred = combine_two_stage_predictions(
        rows,
        stage1_logits=s1_logits,
        stage2_logits=s2_logits,
        stage1_threshold=args.stage1_threshold,
    )
    metrics = write_eval_artifacts(out_dir, args.split_name, rows, pred_rows, final_pred)
    with (out_dir / f"{args.split_name}_metrics_all.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(
        f"[OK] {args.split_name}: acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"none_to_stance={metrics['none_to_stance_rate']:.4f} stance_to_none={metrics['stance_to_none_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
