from __future__ import annotations

import json
from textwrap import dedent

from pkg.config import FROZEN_NOTEBOOKS_ROOT, REPO_ROOT


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


COMMON_SETUP = f"""
from pathlib import Path
import json
from IPython.display import Image, Markdown, display

ROOT = Path(r"{REPO_ROOT}")

def latest_dir(path_str: str) -> Path:
    path = ROOT / path_str
    candidates = sorted([p for p in path.iterdir() if p.is_dir()]) if path.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No directories found under {{path}}")
    return candidates[-1]

def show_text(path: Path, max_chars: int | None = None) -> None:
    text = path.read_text(encoding="utf-8")
    if max_chars is not None:
        text = text[:max_chars]
    display(Markdown(f"```\\n{{text}}\\n```"))

def show_image(path: Path) -> None:
    display(Markdown(f"### {{path.name}}"))
    display(Image(filename=str(path)))
"""


def build_main_overview_notebook() -> None:
    notebook = {
        "cells": [
            markdown_cell(
                """
                # Frozen Overview: Main v2

                This notebook is read-only. It does not run experiments.
                It loads the latest `main_v2` artifacts and report assets and displays the most useful summaries.
                """
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
                suite_manifest = latest_dir("artifacts_v2/_suites/main_v2") / "manifest.json"
                report_root = latest_dir("report_assets_v2/main_v2")

                display(Markdown(f"**Suite manifest:** `{suite_manifest}`"))
                display(Markdown(f"**Report assets root:** `{report_root}`"))

                suite_data = json.loads(suite_manifest.read_text(encoding="utf-8"))
                suite_data["runs"]
                """
            ),
            code_cell(
                """
                show_text(report_root / "tables" / "summary_metrics.md")
                show_text(report_root / "snippets" / "report_snippet.md")
                """
            ),
            code_cell(
                """
                for image_name in [
                    "branin_regret_comparison.png",
                    "hartmann6_regret_comparison.png",
                    "ackley10_regret_comparison.png",
                    "gfn_reward_hist.png",
                ]:
                    show_image(report_root / "img" / image_name)
                """
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = FROZEN_NOTEBOOKS_ROOT / "00_main_overview.ipynb"
    path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def build_benchmark_notebook(filename: str, benchmark: str) -> None:
    notebook = {
        "cells": [
            markdown_cell(
                f"""
                # Frozen View: {benchmark}

                This notebook is read-only. It loads the latest saved artifacts for `{benchmark}` from `main_v2`.
                """
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                f"""
                artifact_root = latest_dir("artifacts_v2/main_v2/{benchmark}")
                report_root = latest_dir("report_assets_v2/main_v2")

                display(Markdown(f"**Artifact root:** `{{artifact_root}}`"))
                display(Markdown(f"**Report assets root:** `{{report_root}}`"))

                summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
                list(summary.keys())
                """
            ),
            code_cell(
                """
                show_text(artifact_root / "partial_summary.json")
                """
            ),
            code_cell(
                f"""
                for image_name in [
                    "{benchmark}_regret_comparison.png",
                    "{benchmark}_random_diagnostics.png",
                    "{benchmark}_gfn_diagnostics.png",
                    "{benchmark}_reward_hist.png",
                ]:
                    show_image(report_root / "img" / image_name)

                show_image(report_root / "comparison" / "{benchmark}_legacy_vs_v2_regret.png")
                """
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = FROZEN_NOTEBOOKS_ROOT / filename
    path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def build_ablation_notebook(filename: str, suite_name: str, spec_names: list[str], title: str) -> None:
    spec_names_literal = json.dumps(spec_names)
    notebook = {
        "cells": [
            markdown_cell(
                f"""
                # {title}

                This notebook is read-only. It loads the latest saved report assets for `{suite_name}`.
                """
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                f"""
                suite_root = latest_dir("report_assets_v2/{suite_name}")
                spec_names = {spec_names_literal}

                display(Markdown(f"**Suite root:** `{{suite_root}}`"))
                spec_names
                """
            ),
            code_cell(
                """
                for spec_name in spec_names:
                    display(Markdown(f"## {spec_name}"))
                    show_text(suite_root / spec_name / "tables" / "summary_metrics.md")
                    show_text(suite_root / spec_name / "snippets" / "report_snippet.md")
                """
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = FROZEN_NOTEBOOKS_ROOT / filename
    path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    build_main_overview_notebook()
    build_benchmark_notebook("01_branin_main_v2.ipynb", "branin")
    build_benchmark_notebook("02_hartmann6_main_v2.ipynb", "hartmann6")
    build_benchmark_notebook("03_ackley10_main_v2.ipynb", "ackley10")
    build_ablation_notebook(
        "10_reward_protocol_ablation.ipynb",
        "reward_protocol_ablation_v2",
        [
            "reward_protocol_softplus_scaled_v2",
            "reward_protocol_zscore_v2",
            "reward_protocol_rank_v2",
        ],
        "Frozen View: Reward Protocol Ablation",
    )
    build_ablation_notebook(
        "11_pool_ablation.ipynb",
        "pool_ablation_v2",
        ["pool_shared_v2", "pool_fresh_v2"],
        "Frozen View: Pool Ablation",
    )
    build_ablation_notebook(
        "12_finetune_ablation.ipynb",
        "finetune_ablation_v2",
        ["finetune_continual_v2", "finetune_restart_v2"],
        "Frozen View: Finetune Ablation",
    )


if __name__ == "__main__":
    main()
