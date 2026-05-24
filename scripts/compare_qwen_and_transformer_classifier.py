from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.classifiers.stance.qwen_compare import (
    STANCE_LABELS,
    build_qwen_prompt,
    normalize_label,
    parse_label_from_llm,
    read_jsonl,
    stratified_sample_rows,
    write_jsonl,
    write_metrics_bundle,
)


def _load_subset(args) -> List[Dict[str, Any]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_jsonl:
        rows = read_jsonl(Path(args.sample_jsonl))
        print(f"[INFO] loaded existing sample: {args.sample_jsonl} ({len(rows)} rows)")
        return rows

    all_rows = read_jsonl(Path(args.dataset_jsonl))
    rows = stratified_sample_rows(
        all_rows,
        n_per_topic_label=args.n_per_topic_label,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    sample_path = out_dir / "sample.jsonl"
    write_jsonl(sample_path, rows)
    pd.DataFrame(rows).to_csv(out_dir / "sample.csv", index=False)
    print(f"[INFO] sampled {len(rows)} rows -> {sample_path}")
    return rows


def _local_predict(args, rows: List[Dict[str, Any]]) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(args.local_model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"local model dir not found: {model_dir}")
    print(f"[INFO] loading local transformer model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), use_safetensors=not args.no_safetensors)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", None) or {i: lab for i, lab in enumerate(STANCE_LABELS)}
    id2label = {int(k): str(v) for k, v in id2label.items()}

    preds: List[str] = []
    confs: List[float] = []
    prob_maps: List[Dict[str, float]] = []
    texts = [str(r.get("input_text") or r.get("comment_text") or "") for r in rows]
    bs = args.local_batch_size
    for start in tqdm(range(0, len(texts), bs), desc="local predict"):
        batch_texts = texts[start : start + bs]
        enc = tokenizer(batch_texts, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        idx = probs.argmax(axis=1)
        for i, prob in zip(idx, probs):
            lab = normalize_label(id2label.get(int(i), str(i)))
            preds.append(lab)
            confs.append(float(prob[int(i)]))
            prob_maps.append({normalize_label(id2label.get(j, str(j))): float(p) for j, p in enumerate(prob)})
    return preds, confs, prob_maps


def _make_openai_client(args):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is required. Install with: pip install -r requirements_qwen_compare.txt") from e
    api_key = args.qwen_api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set PowerShell variable like: $env:DASHSCOPE_API_KEY='your_key', "
            "or pass --qwen_api_key."
        )
    return OpenAI(api_key=api_key, base_url=args.qwen_base_url)


def _qwen_predict(args, rows: List[Dict[str, Any]], out_dir: Path) -> Tuple[List[str], List[str]]:
    if args.skip_qwen:
        print("[INFO] --skip_qwen set; skipping Qwen calls")
        return [], []

    client = _make_openai_client(args)
    cache_path = out_dir / "qwen_raw_predictions.jsonl"
    done: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists() and not args.force_qwen:
        for r in read_jsonl(cache_path):
            done[str(r.get("row_id"))] = r
        print(f"[INFO] loaded qwen cache: {len(done)} rows")

    all_records: List[Dict[str, Any]] = []
    # Preserve old records first; we will rewrite the file in sample order later.
    for r in rows:
        rid = str(r.get("row_id") or r.get("comment_id") or len(all_records))
        if rid in done:
            all_records.append(done[rid])
            continue

        prompt = build_qwen_prompt(r, max_context_chars=args.max_context_chars, max_comment_chars=args.max_comment_chars)
        messages = [
            {"role": "system", "content": "You are a careful target-conditioned stance classifier. Output JSON only."},
            {"role": "user", "content": prompt},
        ]
        last_error = ""
        raw_text = ""
        parsed = "invalid"
        for attempt in range(args.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=args.qwen_model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                raw_text = resp.choices[0].message.content or ""
                parsed, raw_text = parse_label_from_llm(raw_text)
                last_error = ""
                break
            except Exception as e:
                last_error = repr(e)
                if attempt < args.max_retries:
                    time.sleep(args.retry_sleep * (attempt + 1))
        rec = {
            "row_id": rid,
            "gold_stance": r.get("stance"),
            "topic": r.get("topic"),
            "target_entity": r.get("target_entity"),
            "qwen_model": args.qwen_model,
            "qwen_label": parsed,
            "qwen_raw_response": raw_text,
            "error": last_error,
        }
        all_records.append(rec)
        # Incremental write for resumability.
        write_jsonl(cache_path, all_records)
        time.sleep(args.request_sleep)

    # Ensure final file is in the sampled order.
    write_jsonl(cache_path, all_records)
    labels = [normalize_label(r.get("qwen_label")) for r in all_records]
    raws = [str(r.get("qwen_raw_response") or "") for r in all_records]
    return labels, raws


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare a fine-tuned local transformer stance classifier with Qwen on a small stratified sample.")
    ap.add_argument("--dataset_jsonl", default=r".\outputs\stance_dataset_stage1_depth1\test.jsonl")
    ap.add_argument("--sample_jsonl", default=None, help="Reuse an existing sample JSONL instead of sampling from dataset_jsonl.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_topic_label", type=int, default=5)
    ap.add_argument("--max_samples", type=int, default=0, help="Optional cap after stratified sampling; 0 means no cap.")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--local_model_dir", required=True, help="Path to fine-tuned HF model dir, e.g. outputs/.../model/final")
    ap.add_argument("--local_batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_safetensors", action="store_true")

    ap.add_argument("--qwen_model", default="qwen-plus")
    ap.add_argument("--qwen_base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--qwen_api_key", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=20)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--retry_sleep", type=float, default=2.0)
    ap.add_argument("--request_sleep", type=float, default=0.2)
    ap.add_argument("--force_qwen", action="store_true", help="Ignore cached qwen_raw_predictions.jsonl and call Qwen again.")
    ap.add_argument("--skip_qwen", action="store_true", help="Only run local model; useful for checking local path.")
    ap.add_argument("--max_context_chars", type=int, default=1200)
    ap.add_argument("--max_comment_chars", type=int, default=800)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.max_samples = None if args.max_samples == 0 else args.max_samples

    rows = _load_subset(args)
    if not rows:
        raise RuntimeError("No rows sampled/loaded.")

    local_preds, local_confs, local_probs = _local_predict(args, rows)
    local_metrics = write_metrics_bundle(out_dir, "local", rows, local_preds)

    qwen_preds: List[str] = []
    qwen_raws: List[str] = []
    if not args.skip_qwen:
        qwen_preds, qwen_raws = _qwen_predict(args, rows, out_dir)
        qwen_metrics = write_metrics_bundle(out_dir, "qwen", rows, qwen_preds)
    else:
        qwen_metrics = None

    comp_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        row = {
            "row_id": r.get("row_id"),
            "topic": r.get("topic"),
            "target_entity": r.get("target_entity"),
            "gold_stance": r.get("stance"),
            "local_pred": local_preds[i],
            "local_correct": local_preds[i] == r.get("stance"),
            "local_confidence": local_confs[i],
            "comment_text": r.get("comment_text"),
            "parent_text": r.get("parent_text"),
            "post_title": r.get("post_title"),
        }
        for lab in STANCE_LABELS:
            row[f"local_prob_{lab}"] = local_probs[i].get(lab)
        if qwen_preds:
            row["qwen_pred"] = qwen_preds[i]
            row["qwen_correct"] = qwen_preds[i] == r.get("stance")
            row["qwen_raw_response"] = qwen_raws[i]
            row["local_vs_qwen_agree"] = local_preds[i] == qwen_preds[i]
        comp_rows.append(row)
    pd.DataFrame(comp_rows).to_csv(out_dir / "comparison_predictions.csv", index=False)

    summary = {
        "rows": len(rows),
        "sample_source": args.sample_jsonl or args.dataset_jsonl,
        "local_model_dir": args.local_model_dir,
        "qwen_model": None if args.skip_qwen else args.qwen_model,
        "local": {
            "accuracy": local_metrics["accuracy"],
            "macro_f1": local_metrics["macro_f1"],
            "weighted_f1": local_metrics["weighted_f1"],
            "invalid_predictions": local_metrics["invalid_predictions"],
        },
        "qwen": None if qwen_metrics is None else {
            "accuracy": qwen_metrics["accuracy"],
            "macro_f1": qwen_metrics["macro_f1"],
            "weighted_f1": qwen_metrics["weighted_f1"],
            "invalid_predictions": qwen_metrics["invalid_predictions"],
        },
    }
    with (out_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] comparison finished")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"outputs: {out_dir}")


if __name__ == "__main__":
    main()
