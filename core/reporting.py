from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIME_LABELS = {
    0.0: "Day 0",
    1.0: "Day 1",
    2.0: "Day 2",
}

COLOR_LABELS = {
    "green": "Sirloin tender steak",
    "red": "New York strip steak",
}

TIME_COLORS = {
    0.0: "#1f4e79",
    1.0: "#2e8b57",
    2.0: "#b85450",
}


def build_paper_style_outputs(
    aggregated_df: pd.DataFrame,
    model_results: dict[str, dict[str, dict[str, pd.DataFrame]]],
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary = _build_metrics_summary(model_results)
    mechanics_summary = _build_mechanics_summary(aggregated_df)

    metrics_summary.to_csv(tables_dir / "paper_metrics_summary.csv", index=False)
    mechanics_summary.to_csv(tables_dir / "paper_mechanical_summary.csv", index=False)
    metrics_summary.to_csv(tables_dir / "variant_metrics_summary.csv", index=False)

    weighted_results = {color: variants["weighted_all"] for color, variants in model_results.items()}
    _plot_results_panels(weighted_results, figures_dir / "paper_style_model_fits.png")
    _plot_time_evolution(mechanics_summary, figures_dir / "paper_style_time_evolution.png")
    _plot_cross_variant_panels(model_results, figures_dir / "cross_variant")


def _build_metrics_summary(model_results: dict[str, dict[str, dict[str, pd.DataFrame]]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for color, variants in model_results.items():
        for variant, result in variants.items():
            metrics = result["metrics"].copy()
            metrics.insert(0, "variant", variant)
            metrics.insert(0, "cut", COLOR_LABELS.get(color, color))
            metrics.insert(0, "color", color)
            metrics["time_label"] = metrics["time_days"].map(TIME_LABELS).fillna("Overall")
            rows.append(metrics)
    summary = pd.concat(rows, ignore_index=True)
    return summary[
        ["color", "cut", "variant", "train_variant", "loading", "split", "time_days", "time_label", "mae_pa", "rmse_pa", "r2", "n_points"]
    ].sort_values(["color", "variant", "split", "loading", "time_days"])


def _build_mechanics_summary(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = aggregated_df.groupby(["color", "cut_name", "mode", "time_days"], sort=True)
    for (color, cut_name, mode, time_days), group in grouped:
        stress = group["stress_pa"].to_numpy(dtype=float)
        deformation = group["deformation"].to_numpy(dtype=float)
        peak_idx = int(np.argmax(np.abs(stress)))
        rows.append(
            {
                "color": color,
                "cut": cut_name,
                "mode": mode,
                "time_days": float(time_days),
                "time_label": TIME_LABELS.get(float(time_days), f"Day {time_days:g}"),
                "mean_stress_pa": float(np.mean(stress)),
                "peak_abs_stress_pa": float(np.max(np.abs(stress))),
                "peak_stress_pa": float(stress[peak_idx]),
                "peak_deformation": float(deformation[peak_idx]),
                "mean_point_std_pa": float(group["stress_std_pa"].fillna(0.0).mean()),
                "mean_samples_used": float(group["n_used"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["color", "mode", "time_days"]).reset_index(drop=True)


def _plot_results_panels(model_results: dict[str, dict[str, pd.DataFrame]], save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False, sharey=False)
    axes = np.asarray(axes)

    for row_idx, color in enumerate(sorted(model_results)):
        predictions = model_results[color]["predictions"]
        for col_idx, mode in enumerate(["uniaxial", "shear"]):
            ax = axes[row_idx, col_idx]
            subset = predictions.loc[predictions["mode"] == mode]
            for time_days, group in subset.groupby("time_days", sort=True):
                color_code = TIME_COLORS.get(float(time_days), "#444444")
                ordered = group.sort_values("deformation")
                ax.scatter(
                    ordered["deformation"],
                    ordered["stress_pa"],
                    s=18,
                    alpha=0.7,
                    color=color_code,
                    label=f"{TIME_LABELS.get(float(time_days), time_days)} exp",
                )
                ax.plot(
                    ordered["deformation"],
                    ordered["predicted_stress_pa"],
                    linewidth=2.2,
                    color=color_code,
                    label=f"{TIME_LABELS.get(float(time_days), time_days)} CANN",
                )
            ax.set_title(f"{COLOR_LABELS.get(color, color)} | {mode}")
            ax.set_xlabel("Stretch" if mode == "uniaxial" else "Shear strain")
            ax.set_ylabel("Stress (Pa)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def _plot_cross_variant_panels(
    model_results: dict[str, dict[str, dict[str, pd.DataFrame]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for color, variants in model_results.items():
        for variant, result in variants.items():
            predictions = result["predictions"]
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            for ax, loading in zip(axes, ["compression", "tension", "shear"]):
                subset = predictions.loc[predictions["loading"] == loading]
                for time_days, group in subset.groupby("time_days", sort=True):
                    ordered = group.sort_values("deformation")
                    color_code = TIME_COLORS.get(float(time_days), "#444444")
                    ax.scatter(
                        ordered["deformation"],
                        ordered["stress_pa"],
                        s=10,
                        alpha=0.45,
                        color=color_code,
                    )
                    ax.plot(
                        ordered["deformation"],
                        ordered["predicted_stress_pa"],
                        linewidth=2.0,
                        color=color_code,
                        label=TIME_LABELS.get(float(time_days), f"Day {time_days:g}"),
                    )
                ax.set_title(loading)
                ax.set_xlabel("Deformation")
                ax.set_ylabel("Stress (Pa)")
                ax.grid(alpha=0.2)
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"{COLOR_LABELS.get(color, color)} | trained on {variant}")
            fig.tight_layout()
            fig.savefig(output_dir / f"{color}_{variant}_cross_predictions.png", dpi=240)
            plt.close(fig)


def _plot_time_evolution(mechanics_summary: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True)

    for ax, mode in zip(axes, ["uniaxial", "shear"]):
        subset = mechanics_summary.loc[mechanics_summary["mode"] == mode]
        for color, group in subset.groupby("color", sort=True):
            ordered = group.sort_values("time_days")
            ax.plot(
                ordered["time_days"],
                ordered["peak_abs_stress_pa"],
                marker="o",
                linewidth=2.4,
                markersize=7,
                label=COLOR_LABELS.get(color, color),
            )
        ax.set_title(f"{mode.capitalize()} peak stress evolution")
        ax.set_xlabel("Storage time (days)")
        ax.set_ylabel("Peak |stress| (Pa)")
        ax.set_xticks([0.0, 1.0, 2.0], ["0", "1", "2"])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=240)
    plt.close(fig)
