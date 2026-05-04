import pickle
import os
from pathlib import Path

results_filename = "Data/VarPoisson/varpoisson_N16_epochs180_seeds3_20260504_163028.pkl"
for base_path in [Path.cwd(), *Path.cwd().parents]:
    candidate_path = base_path / results_filename
    if candidate_path.exists():
        results_path = candidate_path
        break
else:
    results_path = Path(results_filename)

if not os.path.exists(results_path):
    raise FileNotFoundError(f"Could not find results file: {results_path}")

with open(results_path, "rb") as f:
    payload = pickle.load(f)

# Restore variables expected by the plotting scripts
all_results = payload["all_results"]
aggregated = payload["aggregated"]
all_model_keys = payload["all_model_keys"]

N = payload["N"]
n_epochs = payload["n_epochs"]
richer_epsilon_list = payload["richer_epsilon_list"]

# Optional metadata restore
equation_type = payload.get("equation_type")
model_kind = payload.get("model_kind")
noise_std = payload.get("noise_std")
seed_list = payload.get("seed_list")

print("Loaded results from:", results_path)
print("Equation:", equation_type)
print("Models:", all_model_keys)
print("N:", N)
print("n_epochs:", n_epochs)
print("richer_epsilon_list:", richer_epsilon_list)
