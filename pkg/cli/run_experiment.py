from __future__ import annotations

import argparse
import json

from pkg.experiments import make_experiment_spec, run_experiment
from pkg.reporting.artifacts import _json_default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["branin", "hartmann6", "ackley10"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--reward-protocol", default="rank", choices=["raw", "softplus_scaled", "zscore", "rank"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    spec = make_experiment_spec(
        benchmark_name=args.benchmark,
        name=args.name,
        profile=args.profile,
        reward_protocol=args.reward_protocol,
    )
    result = run_experiment(spec, verbose=args.verbose)
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
