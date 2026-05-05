from __future__ import annotations

import sys
from pathlib import Path


PAPER_PLOTS_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from paper_plot_utils import (  # noqa: E402
    load_results,
    plot_absolute_spectral_error,
    plot_expressibility,
    plot_gradient_power,
    plot_gradient_variance,
    plot_training_fidelity,
    plot_training_loss,
    resolve_results_path,
)


results_path = "../../Data/VarPoisson/varpoisson_N16_epochs180_seeds3_20260504_163028.pkl"
# Alternate repo-root-friendly path used in the original script:
# results_path = "Data/VarPoisson/varpoisson_N16_epochs180_seeds3_20260504_163028.pkl"
alternate_results_path = "Data/VarPoisson/varpoisson_N16_epochs180_seeds3_20260504_163028.pkl"
fig_dir = "paper_figures"


def _resolve_varpoisson_results_path() -> Path:
    script_dir = Path(__file__).resolve().parent

    for candidate in (results_path, alternate_results_path):
        resolved = resolve_results_path(candidate)
        if resolved.exists():
            return resolved

        file_relative = (script_dir / candidate).resolve()
        if file_relative.exists():
            return file_relative

    return resolve_results_path(results_path)


def main() -> None:
    results = load_results(_resolve_varpoisson_results_path())

    plot_absolute_spectral_error(results, fig_dir=fig_dir)
    plot_training_loss(results, fig_dir=fig_dir)
    plot_gradient_power(results, fig_dir=fig_dir)
    plot_gradient_variance(results, fig_dir=fig_dir)
    plot_expressibility(results, fig_dir=fig_dir)
    plot_training_fidelity(results, fig_dir=fig_dir)


if __name__ == "__main__":
    main()
