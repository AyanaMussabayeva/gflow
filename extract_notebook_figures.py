from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "project_report" / "img"


NOTEBOOK_SPECS = {
    "branin_gfn_bo.ipynb": {
        4: ["branin_regret_comparison.png"],
        5: ["branin_random_diagnostics.png", "branin_gfn_diagnostics.png"],
    },
    "hartmann6_gfn_bo.ipynb": {
        4: ["hartmann6_regret_comparison.png"],
        5: ["hartmann6_random_diagnostics.png", "hartmann6_gfn_diagnostics.png"],
    },
    "ackley10_gfn_bo.ipynb": {
        4: ["ackley10_regret_comparison.png"],
        5: ["ackley10_random_diagnostics.png", "ackley10_gfn_diagnostics.png"],
    },
}


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for notebook_name, cell_map in NOTEBOOK_SPECS.items():
        notebook = json.loads((ROOT / notebook_name).read_text(encoding="utf-8"))
        for cell_idx, output_names in cell_map.items():
            outputs = notebook["cells"][cell_idx]["outputs"]
            png_outputs = [out["data"]["image/png"] for out in outputs if "data" in out and "image/png" in out["data"]]
            if len(png_outputs) != len(output_names):
                raise ValueError(
                    f"{notebook_name} cell {cell_idx} has {len(png_outputs)} PNG outputs, expected {len(output_names)}."
                )
            for png_data, output_name in zip(png_outputs, output_names):
                output_path = IMG_DIR / output_name
                output_path.write_bytes(base64.b64decode(png_data))


if __name__ == "__main__":
    main()
