from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="Print compact case records from results.jsonl.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--example_id", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    n = 0
    for r in iter_jsonl(Path(args.results)):
        if args.example_id and r.get("example_id") != args.example_id:
            continue
        if args.policy and r.get("policy") != args.policy:
            continue
        obs = r.get("observation", {})
        dec = r.get("decision", {})
        action = dec.get("action", {})
        pred = r.get("world_model_prediction", {}).get("next_replies", [])
        ev = r.get("evaluation", {})
        print("=" * 100)
        print(f"example_id: {r.get('example_id')} | topic: {r.get('topic')} | policy: {r.get('policy')}")
        print(f"post_title: {obs.get('post_title')}")
        print(f"strategy: {dec.get('strategy') or action.get('strategy')}")
        print(f"action: {action.get('text')}")
        print(f"score: overall={ev.get('overall_score')} safety={ev.get('safety_risk')} target_engagement={ev.get('target_engagement')}")
        print("predicted replies:")
        for x in pred[:5]:
            print(f"  - [{x.get('role')}] {x.get('user_name') or x.get('user_id')}: {x.get('text')}")
        n += 1
        if n >= args.limit:
            break


if __name__ == "__main__":
    main()
