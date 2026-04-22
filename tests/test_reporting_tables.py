from pathlib import Path

from pkg.reporting.tables import benchmark_summary_row, write_report_snippet, write_summary_csv, write_summary_markdown


def test_summary_writers_create_files(tmp_path: Path):
    row = benchmark_summary_row(
        "branin",
        {
            "random": {
                "final_regret_mean": 1.0,
                "final_regret_std": 0.1,
                "floor_reward_fraction_mean": 0.0,
                "same_mask_repeat_std_mean": 0.2,
            },
            "gfn": {
                "final_regret_mean": 0.8,
                "final_regret_std": 0.05,
                "floor_reward_fraction_mean": 0.0,
                "same_mask_repeat_std_mean": 0.1,
            },
            "tradeoff": {"regret_gain_of_gfn": 0.2, "slowdown_factor": 1.5},
        },
    )
    csv_path = write_summary_csv(tmp_path / "summary.csv", [row])
    md_path = write_summary_markdown(tmp_path / "summary.md", [row])
    snippet_path = write_report_snippet(tmp_path / "snippet.md", "main_v2", [row])
    assert csv_path.exists()
    assert md_path.exists()
    assert snippet_path.exists()
