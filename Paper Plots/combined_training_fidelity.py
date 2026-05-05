from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_plot_utils import load_results, plot_combined_training_fidelity  # noqa: E402


RESULT_CONFIGS = (
    (
        "Poisson",
        REPO_ROOT / "Data/Poisson/poisson_N16_epochs180_seeds3_20260504_163301.pkl",
    ),
    (
        "Helmholtz",
        REPO_ROOT
        / "Data/Helmholtz/helmholtz_N16_epochs180_seeds3_20260504_163221.pkl",
    ),
    (
        "Variable-coefficient Poisson",
        REPO_ROOT
        / "Data/VarPoisson/varpoisson_N16_epochs180_seeds3_20260504_163028.pkl",
    ),
)
FIG_DIR = SCRIPT_DIR / "paper_figures"


def main(
    *,
    fig_dir: str | Path = FIG_DIR,
) -> None:
    titled_results = [
        (title, load_results(results_path))
        for title, results_path in RESULT_CONFIGS
    ]
    fig, _ = plot_combined_training_fidelity(titled_results, fig_dir=fig_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
