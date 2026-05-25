from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.experiment.summarizer import summarize_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a Social-ITP experiment directory.")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    summarize_experiment(args.output_dir)


if __name__ == "__main__":
    main()
