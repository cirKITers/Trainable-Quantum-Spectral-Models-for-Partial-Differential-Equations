from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = REPO_ROOT / "Paper Plots"

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from paper_plot_utils import (  # noqa: E402
    LoadedResults,
    load_results,
    plot_absolute_spectral_error,
    plot_combined_training_fidelity,
    plot_combined_training_loss,
    plot_expressibility,
    plot_gradient_power,
    plot_gradient_variance,
    plot_training_fidelity,
    plot_training_loss,
)


@dataclass(frozen=True)
class PlotConfig:
    name: str
    results_path: Path
    figures_dir: Path


PLOT_CONFIGS = {
    "helmholtz": PlotConfig(
        name="Helmholtz",
        results_path=REPO_ROOT
        / "Data/Helmholtz/helmholtz_N16_epochs180_seeds3_20260504_163221.pkl",
        figures_dir=PAPER_PLOTS_DIR / "Helmholtz/paper_figures",
    ),
    "poisson": PlotConfig(
        name="Poisson",
        results_path=REPO_ROOT
        / "Data/Poisson/poisson_N16_epochs180_seeds3_20260504_163301.pkl",
        figures_dir=PAPER_PLOTS_DIR / "Poisson/paper_figures",
    ),
    "varpoisson": PlotConfig(
        name="VarPoisson",
        results_path=REPO_ROOT
        / "Data/VarPoisson/varpoisson_N16_epochs180_seeds3_20260504_163028.pkl",
        figures_dir=PAPER_PLOTS_DIR / "VarPoisson/paper_figures",
    ),
}

COMBINED_FIGURES_DIR = PAPER_PLOTS_DIR / "paper_figures"
COMBINED_TRAINING_LOSS_ORDER = ("poisson", "helmholtz", "varpoisson")
COMBINED_TRAINING_LOSS_TITLES = {
    "poisson": "Poisson",
    "helmholtz": "Helmholtz",
    "varpoisson": "Variable Poisson",
}

PLOT_FUNCTIONS = (
    plot_absolute_spectral_error,
    plot_training_loss,
    plot_gradient_power,
    plot_gradient_variance,
    plot_expressibility,
    plot_training_fidelity,
)


def selected_plot_configs(equations: Sequence[str]) -> list[PlotConfig]:
    if "all" in equations:
        return list(PLOT_CONFIGS.values())

    return [PLOT_CONFIGS[equation] for equation in equations]


def selected_combined_training_loss_configs(
    equations: Sequence[str],
) -> list[tuple[str, PlotConfig]]:
    if "all" in equations:
        keys = COMBINED_TRAINING_LOSS_ORDER
    else:
        keys = [
            equation
            for equation in COMBINED_TRAINING_LOSS_ORDER
            if equation in equations
        ]

    return [
        (COMBINED_TRAINING_LOSS_TITLES[equation], PLOT_CONFIGS[equation])
        for equation in keys
    ]


def plot_results(
    config: PlotConfig,
) -> LoadedResults:
    print(f"Generating {config.name} figures in {config.figures_dir}")
    results = load_results(config.results_path)

    for plot_function in PLOT_FUNCTIONS:
        fig, _ = plot_function(
            results,
            fig_dir=config.figures_dir,
        )
        plt.close(fig)

    return results


def run_plots(
    equations: Sequence[str],
) -> list[LoadedResults]:
    return [plot_results(config) for config in selected_plot_configs(equations)]


def run_combined_training_loss_plot(equations: Sequence[str]) -> LoadedResults | None:
    configs = selected_combined_training_loss_configs(equations)
    if not configs:
        return None

    titled_results = [
        (title, load_results(config.results_path))
        for title, config in configs
    ]
    fig, _ = plot_combined_training_loss(
        titled_results,
        fig_dir=COMBINED_FIGURES_DIR,
    )
    plt.close(fig)
    return titled_results[-1][1]


def run_combined_training_modewise_plot(equations: Sequence[str]) -> LoadedResults | None:
    configs = selected_combined_training_loss_configs(equations)
    if not configs:
        return None

    titled_results = [
        (title, load_results(config.results_path))
        for title, config in configs
    ]
    fig, _ = plot_combined_training_loss(
        titled_results,
        fig_dir=COMBINED_FIGURES_DIR,
    )
    plt.close(fig)
    return titled_results[-1][1]


def run_combined_training_fidelity_plot(
    equations: Sequence[str],
) -> LoadedResults | None:
    configs = selected_combined_training_loss_configs(equations)
    if not configs:
        return None

    titled_results = [
        (title, load_results(config.results_path))
        for title, config in configs
    ]
    fig, _ = plot_combined_training_fidelity(
        titled_results,
        fig_dir=COMBINED_FIGURES_DIR,
    )
    plt.close(fig)
    return titled_results[-1][1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run project workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plots_parser = subparsers.add_parser(
        "plots",
        aliases=["plot"],
        help="Regenerate paper figures from existing result pickle files.",
    )
    plots_parser.add_argument(
        "--equations",
        nargs="+",
        choices=["all", *PLOT_CONFIGS.keys()],
        default=["all"],
        help="Equation result sets to plot. Defaults to all.",
    )
    plots_parser.add_argument(
        "--combined-training-loss",
        action="store_true",
        help=(
            "Also generate one shared multi-panel training-loss figure "
            "for the selected equations."
        ),
    )
    plots_parser.add_argument(
        "--combined-training-fidelity",
        action="store_true",
        help=(
            "Also generate one shared multi-panel training-fidelity figure "
            "for the selected equations."
        ),
    )
    plots_parser.add_argument(
        "--combined-training-modewise",
        action="store_true",
        help=(
            "Also generate one shared multi-panel training-modewise figure "
            "for the selected equations."
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"plots", "plot"}:
        run_plots(
            args.equations,
        )
        if args.combined_training_loss:
            run_combined_training_loss_plot(args.equations)
        if args.combined_training_modewise:
            run_combined_training_modewise_plot(args.equations)
        if args.combined_training_fidelity:
            run_combined_training_fidelity_plot(args.equations)


if __name__ == "__main__":
    main()
