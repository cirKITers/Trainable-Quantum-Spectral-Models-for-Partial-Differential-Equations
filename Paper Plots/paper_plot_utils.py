from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


EPS_LOG = 1e-40
DEFAULT_EXCLUDE_MODELS = {"spectral_filter_baseline", "hhl_like_structured"}


@dataclass(frozen=True)
class LoadedResults:
    all_results: list[dict[str, Any]]
    aggregated: Any
    all_model_keys: list[str]
    N: int
    n_epochs: int
    richer_epsilon_list: list[float]
    equation_type: str | None = None
    model_kind: str | None = None
    noise_std: float | None = None
    seed_list: list[int] | None = None
    resolved_path: Path | None = None


@dataclass(frozen=True)
class PlotStyleMaps:
    plot_model_keys: list[str]
    label_map: dict[str, str]
    color_map: dict[str, Any]
    style_map: dict[str, dict[str, Any]]


def resolve_results_path(results_path: str | Path) -> Path:
    candidate = Path(results_path)
    if candidate.is_absolute():
        return candidate

    for base_path in [Path.cwd(), *Path.cwd().parents]:
        resolved_path = base_path / candidate
        if resolved_path.exists():
            return resolved_path

    return candidate


def load_results(results_path: str | Path) -> LoadedResults:
    resolved_path = resolve_results_path(results_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Could not find results file: {resolved_path}")

    with resolved_path.open("rb") as handle:
        payload = pickle.load(handle)

    loaded = LoadedResults(
        all_results=payload["all_results"],
        aggregated=payload["aggregated"],
        all_model_keys=list(payload["all_model_keys"]),
        N=payload["N"],
        n_epochs=payload["n_epochs"],
        richer_epsilon_list=list(payload["richer_epsilon_list"]),
        equation_type=payload.get("equation_type"),
        model_kind=payload.get("model_kind"),
        noise_std=payload.get("noise_std"),
        seed_list=payload.get("seed_list"),
        resolved_path=resolved_path,
    )

    print("Loaded results from:", loaded.resolved_path)
    print("Equation:", loaded.equation_type)
    print("Models:", loaded.all_model_keys)
    print("N:", loaded.N)
    print("n_epochs:", loaded.n_epochs)
    print("richer_epsilon_list:", loaded.richer_epsilon_list)

    return loaded


def apply_paper_style(variant: str = "line", *, legend_fontsize: int = 10) -> None:
    if variant == "line":
        params = {
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "lines.linewidth": 2.4,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    elif variant == "bar":
        params = {
            "font.size": 11,
            "axes.labelsize": 13,
            "ytick.labelsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    else:
        raise ValueError(f"Unsupported paper style variant: {variant}")

    plt.rcParams.update(params)


def build_style_maps(
    all_model_keys: list[str],
    richer_epsilon_list: list[float],
    label_variant: str = "long",
) -> PlotStyleMaps:
    plot_model_keys = [mk for mk in all_model_keys if mk not in DEFAULT_EXCLUDE_MODELS]

    if label_variant == "absolute_error":
        label_map = {
            "diag_phase_free": "Diagonal phase",
            "unitary": "Spectral hardware-efficient",
            "hhl_like_free": "HHL-inspired",
            "hwe": "Hardware-efficient baseline",
        }
        richer_template = r"Richer spectral $\epsilon={eps:.2f}$"
    elif label_variant == "long":
        label_map = {
            "diag_phase_free": "Diagonal phase",
            "unitary": "Spectral hardware-efficient",
            "hhl_like_free": "HHL-inspired Model",
            "hwe": "Hardware-efficient baseline",
        }
        richer_template = r"Richer spectral $\epsilon={eps:.2f}$"
    elif label_variant == "short":
        label_map = {
            "diag_phase_free": "Diagonal phase",
            "unitary": "Spectral HE",
            "hhl_like_free": "HHL-inspired",
            "hwe": "HE baseline",
        }
        richer_template = r"Richer $\epsilon={eps:.2f}$"
    else:
        raise ValueError(f"Unsupported label variant: {label_variant}")

    for eps in richer_epsilon_list:
        label_map[f"richer_spectral_eps_{eps:.2f}"] = richer_template.format(eps=eps)

    color_map = {
        "diag_phase_free": "#009E73",
        "unitary": "#0072B2",
        "hhl_like_free": "#D55E00",
        "hwe": "#222222",
    }

    richer_keys = [
        f"richer_spectral_eps_{eps:.2f}"
        for eps in richer_epsilon_list
        if f"richer_spectral_eps_{eps:.2f}" in plot_model_keys
    ]

    cmap = cm.get_cmap("Purples", len(richer_keys) + 3)
    for index, model_key in enumerate(richer_keys):
        color_map[model_key] = cmap(index + 2)

    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    style_map = {
        model_key: {
            "linestyle": line_styles[index % len(line_styles)],
            "marker": markers[index % len(markers)],
        }
        for index, model_key in enumerate(plot_model_keys)
    }

    return PlotStyleMaps(
        plot_model_keys=plot_model_keys,
        label_map=label_map,
        color_map=color_map,
        style_map=style_map,
    )


def _extract_metric_stack(
    all_results: list[dict[str, Any]],
    model_key: str,
    metric_key: str,
) -> np.ndarray:
    return np.stack([result[model_key][metric_key] for result in all_results], axis=0)


def _log_space_median_q25_q75(values: np.ndarray, *, axis: int | None = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_values = np.log10(values + EPS_LOG)
    med = np.median(log_values, axis=axis)
    q25 = np.percentile(log_values, 25, axis=axis)
    q75 = np.percentile(log_values, 75, axis=axis)
    return med, q25, q75


def _apply_dynamic_log_ticks(
    ax: plt.Axes,
    log_values: np.ndarray,
    *,
    lower_bound: float = -40,
    upper_bound: float = 2,
    n_labels_max: int = 8,
    pad: float = 0.3,
) -> np.ndarray:
    ymin = np.floor(np.nanmin(log_values))
    ymax = np.ceil(np.nanmax(log_values))

    ymin = max(ymin, lower_bound)
    ymax = min(ymax, upper_bound)

    span = ymax - ymin
    step = max(1, int(np.ceil(span / (n_labels_max - 1))))
    if step > 1 and step % 2 != 0:
        step += 1

    tick_vals = np.arange(ymax, ymin - 1, -step)
    ax.set_ylim(ymin - pad, ymax + pad)
    _set_log_tick_labels(ax, tick_vals)
    return tick_vals


def _set_log_tick_labels(ax: plt.Axes, tick_vals: np.ndarray) -> None:
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([rf"$10^{{{int(value)}}}$" for value in tick_vals])


def _save_figure(
    fig: plt.Figure,
    fig_dir: str | Path,
    pdf_basename: str,
    png_basename: str | None = None,
) -> None:
    output_dir = Path(fig_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_basename = pdf_basename if png_basename is None else png_basename
    fig.savefig(output_dir / f"{pdf_basename}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{png_basename}.png", bbox_inches="tight")


def plot_absolute_spectral_error(
    results: LoadedResults,
    *,
    fig_dir: str | Path = "paper_figures",
    save_figures: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    apply_paper_style("line", legend_fontsize=10)
    styles = build_style_maps(results.all_model_keys, results.richer_epsilon_list, label_variant="absolute_error")

    k_plot = np.arange(1, results.N + 1)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    all_log_values: list[np.ndarray] = []

    for model_key in styles.plot_model_keys:
        metric_values = _extract_metric_stack(results.all_results, model_key, "abs_err")
        med, q25, q75 = _log_space_median_q25_q75(metric_values, axis=0)

        all_log_values.extend([q25, q75])

        ax.plot(
            k_plot,
            med,
            label=styles.label_map.get(model_key, model_key),
            color=styles.color_map.get(model_key),
            linestyle=styles.style_map[model_key]["linestyle"],
            marker=styles.style_map[model_key]["marker"],
            markersize=5.5,
        )
        ax.fill_between(
            k_plot,
            q25,
            q75,
            color=styles.color_map.get(model_key),
            alpha=0.12,
        )

    ax.set_xlim(0.5, results.N + 0.5)
    ax.set_xticks(np.arange(1, results.N + 1, 1))
    ax.set_xlabel(r"Spectral mode $k$")
    _apply_dynamic_log_ticks(ax, np.asarray(all_log_values))
    ax.set_ylabel(r"$E_k$", labelpad=8)

    fig.tight_layout(rect=[0, 0.13, 1, 1])
    if save_figures:
        _save_figure(
            fig,
            fig_dir,
            "fig_absolute_spectral_error_paper_3",
            "fig_absolute_spectral_error_paper",
        )

    
    return fig, ax


def plot_training_loss(
    results: LoadedResults,
    *,
    fig_dir: str | Path = "paper_figures",
    save_figures: bool = True,
    
) -> tuple[plt.Figure, plt.Axes]:
    apply_paper_style("line", legend_fontsize=10)
    styles = build_style_maps(results.all_model_keys, results.richer_epsilon_list, label_variant="long")

    epochs = np.arange(results.n_epochs)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    all_log_values: list[np.ndarray] = []

    for model_key in styles.plot_model_keys:
        metric_values = _extract_metric_stack(results.all_results, model_key, "train_loss")
        med, q25, q75 = _log_space_median_q25_q75(metric_values, axis=0)

        all_log_values.extend([q25, q75])

        ax.plot(
            epochs,
            med,
            label=styles.label_map.get(model_key, model_key),
            color=styles.color_map.get(model_key),
            linestyle=styles.style_map[model_key]["linestyle"],
            marker=styles.style_map[model_key]["marker"],
            markersize=5.0,
            markevery=max(1, results.n_epochs // 12),
        )
        ax.fill_between(
            epochs,
            q25,
            q75,
            color=styles.color_map.get(model_key),
            alpha=0.12,
        )

    ax.set_xlim(0, results.n_epochs - 1)
    ax.set_xlabel("Epoch")
    _apply_dynamic_log_ticks(ax, np.asarray(all_log_values))
    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{train}}$", labelpad=8)

    fig.tight_layout(rect=[0, 0.13, 1, 1])
    if save_figures:
        _save_figure(fig, fig_dir, "fig_training_loss_paper_3")

    
    return fig, ax


def plot_gradient_power(
    results: LoadedResults,
    *,
    fig_dir: str | Path = "paper_figures",
    save_figures: bool = True,
    
) -> tuple[plt.Figure, plt.Axes]:
    apply_paper_style("line", legend_fontsize=9)
    styles = build_style_maps(results.all_model_keys, results.richer_epsilon_list, label_variant="long")

    k_plot = np.arange(1, results.N + 1)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    all_log_values: list[np.ndarray] = []

    for model_key in styles.plot_model_keys:
        metric_values = _extract_metric_stack(results.all_results, model_key, "grad_power")
        med, q25, q75 = _log_space_median_q25_q75(metric_values, axis=0)

        all_log_values.extend([q25, q75])

        ax.plot(
            k_plot,
            med,
            label=styles.label_map.get(model_key, model_key),
            color=styles.color_map.get(model_key),
            linestyle=styles.style_map[model_key]["linestyle"],
            marker=styles.style_map[model_key]["marker"],
            markersize=5.5,
        )
        ax.fill_between(
            k_plot,
            q25,
            q75,
            color=styles.color_map.get(model_key),
            alpha=0.12,
        )

    ax.set_xlim(0.5, results.N + 0.5)
    ax.set_xticks(np.arange(1, results.N + 1, 1))
    ax.set_xlabel(r"Spectral mode $k$")
    _apply_dynamic_log_ticks(ax, np.asarray(all_log_values))
    ax.set_ylabel(r"$G_k$", labelpad=8)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        borderaxespad=0.0,
    )

    fig.tight_layout()
    if save_figures:
        _save_figure(fig, fig_dir, "fig_gradient_power_paper")

    
    return fig, ax


def plot_gradient_variance(
    results: LoadedResults,
    *,
    fig_dir: str | Path = "paper_figures",
    save_figures: bool = True,
    
) -> tuple[plt.Figure, plt.Axes]:
    apply_paper_style("bar")
    styles = build_style_maps(results.all_model_keys, results.richer_epsilon_list, label_variant="short")

    vals, q25s, q75s, cols = [], [], [], []
    for model_key in styles.plot_model_keys:
        metric_values = np.array(
            [result[model_key]["grad_var_mean"] for result in results.all_results],
            dtype=float,
        )
        med, q25, q75 = _log_space_median_q25_q75(metric_values, axis=None)
        vals.append(med)
        q25s.append(q25)
        q75s.append(q75)
        cols.append(styles.color_map.get(model_key))

    vals = np.asarray(vals)
    q25s = np.asarray(q25s)
    q75s = np.asarray(q75s)

    ymin = np.floor(np.min(q25s))
    ymax = np.ceil(np.max(q75s))
    base = ymin - 0.5
    heights = vals - base
    x = np.arange(len(styles.plot_model_keys))

    fig, ax = plt.subplots(figsize=(11, 5.0))
    bars = ax.bar(
        x,
        heights,
        bottom=base,
        color=cols,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.88,
    )

    yerr = np.vstack([vals - q25s, q75s - vals])
    ax.errorbar(
        x,
        vals,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
    )

    special_outside = {"diag_phase_free", "richer_spectral_eps_0.00"}
    common_label_y = base + 0.52 * np.max(heights)
    for index, bar in enumerate(bars):
        model_key = styles.plot_model_keys[index]
        text_color = "gray" if model_key in special_outside else "white"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            common_label_y,
            styles.label_map.get(model_key, model_key),
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            color=text_color,
            weight="bold",
        )

    ax.set_xticks([])
    tick_vals = np.arange(ymax, ymin - 1, -2)
    ax.set_ylim(base, ymax + 0.8)
    _set_log_tick_labels(ax, tick_vals)
    ax.set_ylabel(r"$V_g$")

    fig.tight_layout()
    if save_figures:
        _save_figure(fig, fig_dir, "fig_gradient_variance_final_aligned")

    
    return fig, ax


def plot_expressibility(
    results: LoadedResults,
    *,
    fig_dir: str | Path = "paper_figures",
    save_figures: bool = True,
    
) -> tuple[plt.Figure, plt.Axes]:
    apply_paper_style("bar")
    styles = build_style_maps(results.all_model_keys, results.richer_epsilon_list, label_variant="short")

    vals, q25s, q75s, cols = [], [], [], []
    for model_key in styles.plot_model_keys:
        metric_values = np.array(
            [result[model_key]["expressibility"] for result in results.all_results],
            dtype=float,
        )
        vals.append(np.median(metric_values))
        q25s.append(np.percentile(metric_values, 25))
        q75s.append(np.percentile(metric_values, 75))
        cols.append(styles.color_map.get(model_key))

    vals = np.asarray(vals)
    q25s = np.asarray(q25s)
    q75s = np.asarray(q75s)
    x = np.arange(len(styles.plot_model_keys))
    ymax = np.ceil(np.max(q75s) * 10) / 10

    fig, ax = plt.subplots(figsize=(11, 5.0))
    bars = ax.bar(
        x,
        vals,
        color=cols,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.88,
    )

    yerr = np.vstack([vals - q25s, q75s - vals])
    ax.errorbar(
        x,
        vals,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
    )

    label_y = 0.15
    for index, bar in enumerate(bars):
        model_key = styles.plot_model_keys[index]
        text_color = "white" if model_key == "hhl_like_free" else "gray"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            styles.label_map.get(model_key, model_key),
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            color=text_color,
            weight="bold",
        )

    ax.set_xticks([])
    ax.set_ylim(0, ymax + 0.05)
    ax.set_ylabel(r"$D_{\mathrm{KL}}\!\left(p(F)\,\|\,p_{\mathrm{Haar}}(F)\right)$")

    fig.tight_layout()
    if save_figures:
        _save_figure(fig, fig_dir, "fig_expressibility_centered_labels")

    
    return fig, ax


def plot_training_fidelity(
    results: LoadedResults,
    *,
    fig_dir: str | Path = "paper_figures",
    save_figures: bool = True,
    
) -> tuple[plt.Figure, plt.Axes]:
    apply_paper_style("line", legend_fontsize=10)
    styles = build_style_maps(results.all_model_keys, results.richer_epsilon_list, label_variant="long")

    epochs = np.arange(results.n_epochs)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    for model_key in styles.plot_model_keys:
        metric_values = _extract_metric_stack(results.all_results, model_key, "train_fid")
        mean = np.mean(metric_values, axis=0)
        std = np.std(metric_values, axis=0)

        ax.plot(
            epochs,
            mean,
            label=styles.label_map.get(model_key, model_key),
            color=styles.color_map.get(model_key),
            linestyle=styles.style_map[model_key]["linestyle"],
            marker=styles.style_map[model_key]["marker"],
            markersize=5.0,
            markevery=max(1, results.n_epochs // 12),
        )
        ax.fill_between(
            epochs,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=styles.color_map.get(model_key),
            alpha=0.12,
        )

    ax.set_xlim(0, results.n_epochs - 1)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$F_{\mathrm{train}}$")
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )

    fig.tight_layout(rect=[0, 0.13, 1, 1])
    if save_figures:
        _save_figure(fig, fig_dir, "fig_training_fidelity_paper")

    
    return fig, ax


__all__ = [
    "LoadedResults",
    "PlotStyleMaps",
    "apply_paper_style",
    "build_style_maps",
    "load_results",
    "plot_absolute_spectral_error",
    "plot_expressibility",
    "plot_gradient_power",
    "plot_gradient_variance",
    "plot_training_fidelity",
    "plot_training_loss",
    "resolve_results_path",
]
