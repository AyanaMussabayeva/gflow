# Legacy Pipeline

This directory contains the pre-`pkg` version of the project.

The contents were kept in the original relative layout:

- `gfn_bo_experiments.py`
- benchmark notebooks
- `run_compute_tradeoff_experiment.py`
- `results/`

`project_report/` and PDF files were intentionally left at the repository root and were not moved here.

## Running The Legacy Pipeline

From the repository root:

```bash
python legacy/run_compute_tradeoff_experiment.py --profile smoke
python legacy/run_compute_tradeoff_experiment.py --profile full
```

Or open the notebooks:

- `legacy/branin_gfn_bo.ipynb`
- `legacy/hartmann6_gfn_bo.ipynb`
- `legacy/ackley10_gfn_bo.ipynb`

Legacy artifacts are written to:

- `legacy/results/compute_tradeoff/...`

This keeps the old and new versions side by side without artifact collisions.
