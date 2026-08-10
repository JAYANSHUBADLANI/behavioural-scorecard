"""Single entrypoint for the pipeline phases."""

import argparse
import json
import sys

from src.config import Config
from src import phase1, phase2, phase3, phase4

PHASES = {"1": phase1.run, "2": phase2.run, "3": phase3.run, "4": phase4.run}


def main() -> int:
    parser = argparse.ArgumentParser(description="Behavioural scorecard pipeline")
    parser.add_argument("phase", choices=sorted(PHASES) + ["all"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true",
                        help="rebuild cached interim data instead of reusing it")
    args = parser.parse_args()

    config = Config.load(args.config) if args.config else Config.load()
    selected = sorted(PHASES) if args.phase == "all" else [args.phase]

    for name in selected:
        print(f"running phase {name}")
        summary = PHASES[name](config, force=args.force)
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
