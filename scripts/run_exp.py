from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from social_itp.experiment.config import load_config
from social_itp.experiment.runner import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a Social-ITP experiment from a YAML/JSON config.")
    ap.add_argument("--config", required=True, help="Path to config YAML/JSON.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
