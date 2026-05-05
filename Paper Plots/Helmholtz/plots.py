from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

PAPER_PLOTS_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from paper_plot_utils import (
    LoadedResults,
    load_results as _shared_load_results,
    plot_absolute_spectral_error,
    plot_expressibility,
    plot_gradient_power,
    plot_gradient_variance,
    plot_training_fidelity,
    plot_training_loss,
)


results_path = "../../Data/Helmholtz/helmholtz_N16_epochs180_seeds3_20260504_163221.pkl"


def _resolve_results_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (Path(__file__).resolve().parent / candidate).resolve()


def load_helmholtz_results() -> LoadedResults:
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = _shared_load_results(_resolve_results_path(results_path))

    print("Loaded results from:", results_path)
    print("Equation:", loaded.equation_type)
    print("Models:", loaded.all_model_keys)
    print("N:", loaded.N)
    print("n_epochs:", loaded.n_epochs)
    print("richer_epsilon_list:", loaded.richer_epsilon_list)
    return loaded


def main(*, fig_dir: str | Path = "paper_figures", save_figures: bool = True, show: bool = True) -> None:
    results = load_helmholtz_results()

    plot_absolute_spectral_error(results, fig_dir=fig_dir, save_figures=save_figures, show=show)
    plot_training_loss(results, fig_dir=fig_dir, save_figures=save_figures, show=show)
    plot_gradient_power(results, fig_dir=fig_dir, save_figures=save_figures, show=show)
    plot_gradient_variance(results, fig_dir=fig_dir, save_figures=save_figures, show=show)
    plot_expressibility(results, fig_dir=fig_dir, save_figures=save_figures, show=show)
    plot_training_fidelity(results, fig_dir=fig_dir, save_figures=save_figures, show=show)


if __name__ == "__main__":
    main()
