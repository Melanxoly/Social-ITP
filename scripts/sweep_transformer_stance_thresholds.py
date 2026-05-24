from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.classifiers.stance.dataset import read_jsonl
from social_itp.classifiers.stance.transformer_finetune import (
    THREE_CLASS_LABELS,
    build_dataset,
    evaluate_three_class_predictions,
    label_names_for_task,
)


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _safe_name(th: float) -> str:
    return f"thr{th:.2f}".replace(".", "p")


def _confusion_rates(metrics: Dict) -> Dict[str, float]:
    labels = metrics.get("labels") or THREE_CLASS_LABELS
    cm = metrics.get("confusion_matrix") or []
    idx = {lab: i for i, lab in enumerate(labels)}
    def cell(gold: str, pred: str) -> int:
        return int(cm[idx[gold]][idx[pred]])

    favor_total = sum(int(x) for x in cm[idx["favor"]])
    against_total = sum(int(x) for x in cm[idx["against"]])
    none_total = sum(int(x) for x in cm[idx["none"]])
    stance_total = favor_total + against_total
    none_to_stance = (cell("none", "favor") + cell("none", "against")) / max(1, none_total)
    stance_to_none = (cell("favor", "none") + cell("against", "none")) / max(1, stance_total)
    favor_against_confusion = (cell("favor", "against") + cell("against", "favor")) / max(1, stance_total)
    return {
        "none_to_stance_rate": float(none_to_stance),
        "stance_to_none_rate": float(stance_to_none),
        "favor_against_confusion_rate": float(favor_against_confusion),
        "boundary_avg": float((none_to_stance + stance_to_none) / 2.0),
        "boundary_max": float(max(none_to_stance, stance_to_none)),
        "boundary_weighted": float((2.0 * none_to_stance + stance_to_none) / 3.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep conservative stance-confidence thresholds for a fine-tuned three-class transformer classifier."
    )
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--model_dir", required=True, help="Saved HF model dir, e.g. outputs/.../model/final")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_name", default="test")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=16)
    ap.add_argument("--thresholds", default="0.0,0.5,0.55,0.6,0.65,0.7,0.75")
    ap.add_argument("--none_confidence_policy", choices=["keep"], default="keep")
    ap.add_argument("--write_artifacts", action="store_true", help="Write full metrics/confusion/prediction files for every threshold.")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.dataset_jsonl))
    if not rows:
        raise RuntimeError(f"No rows found in {args.dataset_jsonl}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = label_names_for_task("three_class")
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    print(f"[INFO] loading model once: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    ds = build_dataset(tokenizer, rows, "three_class", label2id, args.max_length)

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

    print("[INFO] running one forward pass")
    pred = trainer.predict(ds)
    logits = pred.predictions

    thresholds = _parse_floats(args.thresholds)
    summary_rows = []
    for th in thresholds:
        if args.write_artifacts:
            th_out = out_dir / _safe_name(th)
            split = args.split_name
        else:
            th_out = out_dir / "_tmp_metrics"
            split = _safe_name(th)
        metrics = evaluate_three_class_predictions(
            rows=rows,
            logits=logits,
            out_dir=th_out,
            split_name=split,
            id2label=id2label,
            stance_confidence_threshold=th,
            none_confidence_policy=args.none_confidence_policy,
        )
        rep = metrics.get("classification_report", {})
        rates = _confusion_rates(metrics)
        row = {
            "threshold": float(th),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "favor_f1": rep.get("favor", {}).get("f1-score", 0.0),
            "against_f1": rep.get("against", {}).get("f1-score", 0.0),
            "none_f1": rep.get("none", {}).get("f1-score", 0.0),
            "favor_precision": rep.get("favor", {}).get("precision", 0.0),
            "favor_recall": rep.get("favor", {}).get("recall", 0.0),
            "against_precision": rep.get("against", {}).get("precision", 0.0),
            "against_recall": rep.get("against", {}).get("recall", 0.0),
            "none_precision": rep.get("none", {}).get("precision", 0.0),
            "none_recall": rep.get("none", {}).get("recall", 0.0),
            "num_conservative_to_none": metrics.get("num_conservative_to_none", 0),
            "num_low_confidence_none": metrics.get("num_low_confidence_none", 0),
            **rates,
        }
        summary_rows.append(row)
        print(
            f"[OK] threshold={th:.2f} acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
            f"none_to_stance={row['none_to_stance_rate']:.4f} stance_to_none={row['stance_to_none_rate']:.4f} "
            f"converted={row['num_conservative_to_none']}"
        )

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / "threshold_sweep_summary.csv", index=False)
    with (out_dir / "threshold_sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    # Convenience rankings.
    df.sort_values(["macro_f1"], ascending=False).to_csv(out_dir / "rank_by_macro_f1.csv", index=False)
    df.sort_values(["boundary_max", "boundary_avg", "threshold"]).to_csv(out_dir / "rank_by_boundary_max.csv", index=False)
    df.sort_values(["none_to_stance_rate", "macro_f1"], ascending=[True, False]).to_csv(out_dir / "rank_by_none_to_stance.csv", index=False)
    print(f"[OK] wrote sweep results to {out_dir / 'threshold_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
