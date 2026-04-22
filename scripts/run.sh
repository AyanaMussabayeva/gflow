#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${1:-smoke}"
TARGET="${2:-all}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run.sh [smoke|full] [all|main-only|ablations-only]

Examples:
  bash scripts/run.sh
  bash scripts/run.sh smoke all
  bash scripts/run.sh full main-only
  bash scripts/run.sh full ablations-only

Behavior:
  - runs the main v2 suite and/or required ablation suites
  - exports report-ready assets after the runs finish
EOF
}

if [[ "$PROFILE" != "smoke" && "$PROFILE" != "full" ]]; then
  usage
  echo
  echo "Invalid profile: $PROFILE" >&2
  exit 1
fi

if [[ "$TARGET" != "all" && "$TARGET" != "main-only" && "$TARGET" != "ablations-only" ]]; then
  usage
  echo
  echo "Invalid target: $TARGET" >&2
  exit 1
fi

run_main() {
  echo "[run.sh] Running main_v2 ($PROFILE)"
  python scripts/run_v2_suite.py --suite main_v2 --profile "$PROFILE"

  echo "[run.sh] Exporting main_v2 report assets"
  python scripts/export_report_assets_v2.py --spec main_v2 --suite-name main_v2
}

run_ablations() {
  echo "[run.sh] Running reward_protocol_ablation_v2 ($PROFILE)"
  python scripts/run_reward_protocol_ablation_v2.py --profile "$PROFILE"

  echo "[run.sh] Exporting reward_protocol_ablation_v2 assets"
  python scripts/export_reward_protocol_ablation_v2.py

  echo "[run.sh] Running pool_ablation_v2 ($PROFILE)"
  python scripts/run_v2_suite.py --suite pool_ablation_v2 --profile "$PROFILE"

  echo "[run.sh] Running finetune_ablation_v2 ($PROFILE)"
  python scripts/run_v2_suite.py --suite finetune_ablation_v2 --profile "$PROFILE"
}

echo "[run.sh] Root: $ROOT_DIR"
echo "[run.sh] Profile: $PROFILE"
echo "[run.sh] Target: $TARGET"

case "$TARGET" in
  all)
    run_main
    run_ablations
    ;;
  main-only)
    run_main
    ;;
  ablations-only)
    run_ablations
    ;;
esac

echo "[run.sh] Done."
