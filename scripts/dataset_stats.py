from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.data.loader import DatasetLoadConfig, iter_loaded_examples


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute dataset statistics for Social-ITP step-transition data.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--topics", required=True, help="Comma-separated topic names.")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_examples_per_topic", type=int, default=None)
    args = ap.parse_args()

    topics = [x.strip() for x in args.topics.split(",") if x.strip()]
    cfg = DatasetLoadConfig(
        data_root=args.data_root,
        topics=topics,
        max_examples_per_topic=args.max_examples_per_topic,
    )

    topic_stats = {}
    for loaded in iter_loaded_examples(cfg):
        st = topic_stats.setdefault(loaded.topic, {
            "examples": 0,
            "thread_paths": set(),
            "graph_nodes": [],
            "next_replies": [],
            "target_replies": [],
            "bystander_replies": [],
            "bystanders_in_obs": [],
            "roles": Counter(),
        })
        ex = loaded.example
        st["examples"] += 1
        st["thread_paths"].add(str(loaded.thread_path))
        st["graph_nodes"].append(len(ex.observation.graph_nodes))
        replies = ex.label.next_replies
        st["next_replies"].append(len(replies))
        st["target_replies"].append(sum(1 for r in replies if r.role == "target_user"))
        st["bystander_replies"].append(sum(1 for r in replies if r.role == "bystander"))
        st["bystanders_in_obs"].append(len(ex.observation.bystanders))
        for r in replies:
            st["roles"][r.role] += 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    rows = []
    for topic, st in topic_stats.items():
        roles = st["roles"]
        rows.append({
            "topic": topic,
            "thread_files_used": len(st["thread_paths"]),
            "examples": st["examples"],
            "avg_graph_nodes": mean(st["graph_nodes"]),
            "avg_next_replies": mean(st["next_replies"]),
            "avg_target_replies": mean(st["target_replies"]),
            "avg_bystander_replies": mean(st["bystander_replies"]),
            "avg_bystanders_in_obs": mean(st["bystanders_in_obs"]),
            "label_target_user": roles["target_user"],
            "label_bystander": roles["bystander"],
            "label_action_author": roles["action_author"],
            "label_other": roles["other"],
        })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
