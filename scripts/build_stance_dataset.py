from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.classifiers.stance.dataset import StanceDatasetConfig, scan_stance_sources, write_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Build train/dev/test JSONL files for supervised stance classification.")
    ap.add_argument("--data_root", required=True, help="Path to MCSD root directory, e.g. ./data/MCSD")
    ap.add_argument("--topics", required=True, help="Comma-separated topics, e.g. biden,Bitcoin,BMW,costco,tesla,trump")
    ap.add_argument("--out_dir", required=True, help="Output directory for stance dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--no_context", action="store_true", help="Use comment text only instead of topic/post/parent context")
    ap.add_argument(
        "--input_template",
        choices=["task", "plain", "stage1", "simple_stage1", "stance_presence"],
        default="task",
        help="Input text template. stage1 is a simplified none-vs-stance template.",
    )
    ap.add_argument("--split_by_topic", action="store_true", help="Split threads independently inside each topic")
    ap.add_argument("--min_text_len", type=int, default=2)
    ap.add_argument("--max_post_chars", type=int, default=600)
    ap.add_argument("--max_parent_chars", type=int, default=360)
    ap.add_argument("--max_ancestor_depth", type=int, default=1, help="Number of ancestor comments to include as context")
    ap.add_argument("--allow_empty", action="store_true", help="Do not fail when zero rows are extracted")
    ap.add_argument("--scan_only", action="store_true", help="Only scan directories/labels and write scan_report.json")
    args = ap.parse_args()

    topics = [x.strip() for x in args.topics.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scan_only:
        report = scan_stance_sources(args.data_root, topics)
        with (out_dir / "scan_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[OK] wrote scan report to {out_dir / 'scan_report.json'}")
        return

    cfg = StanceDatasetConfig(
        data_root=args.data_root,
        topics=topics,
        out_dir=args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        include_context=not args.no_context,
        input_template=args.input_template,
        split_by_topic=args.split_by_topic,
        min_text_len=args.min_text_len,
        max_post_chars=args.max_post_chars,
        max_parent_chars=args.max_parent_chars,
        max_ancestor_depth=args.max_ancestor_depth,
        fail_on_empty=not args.allow_empty,
    )
    summary = write_dataset(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] wrote stance dataset to {Path(args.out_dir)}")


if __name__ == "__main__":
    main()
