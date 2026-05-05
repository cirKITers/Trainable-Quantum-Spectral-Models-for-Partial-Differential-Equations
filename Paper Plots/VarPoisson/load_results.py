from __future__ import annotations

import sys
from pathlib import Path


PAPER_PLOTS_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from paper_plot_utils import load_results as _load_results  # noqa: E402


results_path = "Data/VarPoisson/varpoisson_N16_epochs180_seeds3_20260504_163028.pkl"
loaded_results = _load_results(results_path)

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
    "results_path",
    "resolved_path",
    "loaded_results",
    "all_results",
    "aggregated",
    "all_model_keys",
    "N",
    "n_epochs",
    "richer_epsilon_list",
    "equation_type",
    "model_kind",
    "noise_std",
    "seed_list",
]
