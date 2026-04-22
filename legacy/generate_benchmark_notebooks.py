from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in dedent(text).strip().splitlines()],
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in dedent(code).strip().splitlines()],
    }


def notebook_for(benchmark_name: str, title: str, output_name: str) -> None:
    cells = [
        markdown_cell(
            f"""
            # {title}

            This notebook compares two BO policies on `{benchmark_name}`:

            - `random`: block-wise dropout masks are sampled uniformly at random.
            - `gfn`: a contextual GFlowNet is trained to generate masks with probability proportional to proxy improvement potential.

            The implementation lives in `gfn_bo_experiments.py`, so the same surrogate, reward, and evaluation code is reused across all benchmarks.
            """
        ),
        markdown_cell(
            """
            ## Notes

            - The GFlowNet is re-trained or fine-tuned at every BO step on the current surrogate, using a dataset-statistics context vector.
            - Proxy rewards are computed from held-out random masks to reduce overfitting to a single overconfident surrogate interpretation.
            - If the full configuration is too slow on your machine, reduce `cfg.seeds`, `cfg.n_iter`, `cfg.gfn_steps`, or `cfg.n_candidates`.
            """
        ),
        code_cell(
            f"""
            from gfn_bo_experiments import (
                default_config,
                plot_proxy_diagnostics,
                plot_regret_comparison,
                print_comparison_table,
                run_benchmark_comparison,
            )

            benchmark_name = "{benchmark_name}"
            cfg = default_config(benchmark_name, seeds=[0, 1, 2, 3, 4])
            cfg
            """
        ),
        code_cell(
            """
            results = run_benchmark_comparison(benchmark_name, cfg)
            print_comparison_table(results)
            """
        ),
        code_cell(
            f"""
            plot_regret_comparison(
                results,
                title="{title}: Random masks vs contextual GFlowNet masks",
            )
            """
        ),
        code_cell(
            """
            random_trial = min(
                results["random"]["trials"],
                key=lambda trial: trial["regrets"][-1],
            )
            gfn_trial = min(
                results["gfn"]["trials"],
                key=lambda trial: trial["regrets"][-1],
            )

            plot_proxy_diagnostics(random_trial, "Best random-mask seed")
            plot_proxy_diagnostics(gfn_trial, "Best GFlowNet seed")
            """
        ),
        code_cell(
            """
            random_final = results["random"]["summary"]["final_regret_mean"]
            gfn_final = results["gfn"]["summary"]["final_regret_mean"]
            improvement = random_final - gfn_final

            print(f"Mean final regret (random): {random_final:.4f}")
            print(f"Mean final regret (gfn)   : {gfn_final:.4f}")
            print(f"Absolute regret gain      : {improvement:.4f}")
            """
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path = ROOT / output_name
    output_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    notebook_for("branin", "Branin (2D) GFlowNet-Guided BO", "branin_gfn_bo.ipynb")
    notebook_for("hartmann6", "Hartmann-6 (6D) GFlowNet-Guided BO", "hartmann6_gfn_bo.ipynb")
    notebook_for("ackley10", "Ackley (10D) GFlowNet-Guided BO", "ackley10_gfn_bo.ipynb")


if __name__ == "__main__":
    main()
