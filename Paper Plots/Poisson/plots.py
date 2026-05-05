from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR.parent

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from paper_plot_utils import (  # noqa: E402
    LoadedResults,
    load_results,
    plot_absolute_spectral_error,
    plot_expressibility,
    plot_gradient_power,
    plot_gradient_variance,
    plot_training_fidelity,
    plot_training_loss,
)


RESULTS_PATH = "Data/Poisson/poisson_N16_epochs180_seeds3_20260504_163301.pkl"
FIG_DIR = SCRIPT_DIR / "paper_figures"


def main(
    *,
    fig_dir: str | Path = FIG_DIR,
    save_figures: bool = True,
    show: bool = True,
) -> LoadedResults:
    results = load_results(RESULTS_PATH)

    plot_absolute_spectral_error(results, fig_dir=fig_dir, save_figures=save_figures, show=show)
    plot_training_loss(results, fig_dir=fig_dir, save_figures=save_figures, show=show)
    plot_gradient_power(results, fig_dir=fig_dir, save_figures=save_figures, show=show)
    plot_gradient_variance(results, fig_dir=fig_dir, save_figures=save_figures, show=show)
    plot_expressibility(results, fig_dir=fig_dir, save_figures=save_figures, show=show)
    plot_training_fidelity(results, fig_dir=fig_dir, save_figures=save_figures, show=show)

    return results


if __name__ == "__main__":
    main()
