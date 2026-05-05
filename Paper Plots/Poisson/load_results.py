from __future__ import annotations

import sys
from pathlib import Path


POISSON_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = POISSON_DIR.parent

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from paper_plot_utils import load_results as shared_load_results


results_path = "Data/Poisson/poisson_N16_epochs180_seeds3_20260504_163301.pkl"

loaded_results = shared_load_results(results_path)

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

payload = {
    "all_results": all_results,
    "aggregated": aggregated,
    "all_model_keys": all_model_keys,
    "N": N,
    "n_epochs": n_epochs,
    "richer_epsilon_list": richer_epsilon_list,
    "equation_type": equation_type,
    "model_kind": model_kind,
    "noise_std": noise_std,
    "seed_list": seed_list,
}


__all__ = [
    "results_path",
    "resolved_path",
    "loaded_results",
    "payload",
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
