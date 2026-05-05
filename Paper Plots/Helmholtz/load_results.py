from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

PAPER_PLOTS_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from paper_plot_utils import LoadedResults, load_results as _shared_load_results


results_path = "Data/Helmholtz/helmholtz_N16_epochs180_seeds3_20260504_163221.pkl"


def _resolve_results_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PAPER_PLOTS_DIR.parent / candidate).resolve()


def load_results(path: str | Path = results_path) -> LoadedResults:
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = _shared_load_results(_resolve_results_path(path))

    print("Loaded results from:", path)
    print("Equation:", loaded.equation_type)
    print("Models:", loaded.all_model_keys)
    print("N:", loaded.N)
    print("n_epochs:", loaded.n_epochs)
    print("richer_epsilon_list:", loaded.richer_epsilon_list)
    return loaded


loaded_results = load_results()

all_results = loaded_results.all_results
aggregated = loaded_results.aggregated
all_model_keys = loaded_results.all_model_keys
N = loaded_results.N
n_epochs = loaded_results.n_epochs
richer_epsilon_list = loaded_results.richer_epsilon_list
equation_type = loaded_results.equation_type
model_kind = loaded_results.model_kind
noise_std = loaded_results.noise_std
seed_list = loaded_results.seed_list
resolved_path = loaded_results.resolved_path


__all__ = [
    "LoadedResults",
    "N",
    "aggregated",
    "all_model_keys",
    "all_results",
    "equation_type",
    "load_results",
    "loaded_results",
    "model_kind",
    "n_epochs",
    "noise_std",
    "resolved_path",
    "results_path",
    "richer_epsilon_list",
    "seed_list",
]
