from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.classifiers.emotion.goemotions_mapping import (
    FINE_TO_COARSE,
    balanced_sample,
    convert_split_to_rows,
    get_goemotions_label_names,
    load_goemotions_dataset,
    make_summary,
    write_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a 3-class negative/neutral/positive dataset from GoEmotions.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_name", default="google-research-datasets/go_emotions")
    ap.add_argument("--config_name", default="simplified")
    ap.add_argument("--train_per_class", type=int, default=5000)
    ap.add_argument("--dev_per_class", type=int, default=800)
    ap.add_argument("--test_per_class", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--multi_label_policy", choices=["skip_conflict", "priority_negative", "priority_non_neutral"], default="skip_conflict")
    ap.add_argument("--no_prefer_single_label", action="store_true", help="Allow more mixed/multi-label examples after mapping.")
    ap.add_argument("--allow_less", action="store_true", help="Allow fewer than requested if a class has insufficient examples.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] loading GoEmotions")
    ds = load_goemotions_dataset(args.dataset_name, args.config_name)
    label_names = get_goemotions_label_names(ds["train"])

    split_rows = {}
    split_meta = {}
    for split in ["train", "validation", "test"]:
        rows, meta = convert_split_to_rows(
            ds[split],
            split_name=split,
            label_names=label_names,
            multi_label_policy=args.multi_label_policy,
            prefer_single_label=not args.no_prefer_single_label,
        )
        split_rows[split] = rows
        split_meta[split] = meta
        print(f"[INFO] mapped {split}: {meta}")

    train_rows, train_sample_meta = balanced_sample(split_rows["train"], per_class=args.train_per_class, seed=args.seed, allow_less=args.allow_less)
    dev_rows, dev_sample_meta = balanced_sample(split_rows["validation"], per_class=args.dev_per_class, seed=args.seed + 1, allow_less=True)
    test_rows, test_sample_meta = balanced_sample(split_rows["test"], per_class=args.test_per_class, seed=args.seed + 2, allow_less=True)

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "dev.jsonl", dev_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    summary = {
        "dataset_name": args.dataset_name,
        "config_name": args.config_name,
        "label_names": label_names,
        "fine_to_coarse": FINE_TO_COARSE,
        "mapping_note": "confusion is mapped to negative as requested.",
        "multi_label_policy": args.multi_label_policy,
        "prefer_single_label": not args.no_prefer_single_label,
        "requested": {
            "train_per_class": args.train_per_class,
            "dev_per_class": args.dev_per_class,
            "test_per_class": args.test_per_class,
        },
        "mapped_split_meta": split_meta,
        "sample_meta": {"train": train_sample_meta, "dev": dev_sample_meta, "test": test_sample_meta},
        "splits": {"train": make_summary(train_rows), "dev": make_summary(dev_rows), "test": make_summary(test_rows)},
    }
    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))
    print(f"[OK] wrote dataset to {out_dir}")


if __name__ == "__main__":
    main()
