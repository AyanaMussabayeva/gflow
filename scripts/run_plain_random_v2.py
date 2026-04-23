from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_v2_suite.py"),
            "--suite",
            "plain_random_v2",
            "--profile",
            args.profile,
            "--methods",
            "random",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
