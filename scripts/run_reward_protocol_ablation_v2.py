from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v2_suite import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--suite", "reward_protocol_ablation_v2", *sys.argv[1:]]
    main()
