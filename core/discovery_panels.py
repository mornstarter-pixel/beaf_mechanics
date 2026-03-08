from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from core.data_pipeline import (
    build_full_training_payload_from_curves,
    load_all_workbooks,
    preprocess_robust_curves,
)
from core.runtime import RuntimeConfig, configure_tensorflow_runtime, runtime_summary_text
from core.settings import ProjectPaths
from core.time_cann import TrainingConfig, fit_cann


TRAIN_VARIANTS = [
    ("compression_only", "train compression"),
    ("unloading_only", "train unloading"),
    ("shear_only", "train shear"),
    ("weighted_all", "train all"),
]

ROW_LOADINGS = [
    ("compression", "compression"),
    ("unloading", "unloading"),
    ("shear", "shear"),
]

TERM_LABELS = [
    "[I1-3]",
    "exp([I1-3])-1",
    "ln(1-[I1-3])",
    "[I1-3]^2",
    "exp([I1-3]^2)-1",
    "ln(1-[I1-3]^2)",
    "[I2-3]",
    "exp([I2-3])-1",
    "ln(1-[I2-3])",
    "[I2-3]^2",
    "exp([I2-3]^2)-1",
    "ln(1-[I2-3]^2)",
]

TIME_LABELS = {
    0.0: "Day 0",
    1.0: "Day 1",
    2.0: "Day 2",
}

COLOR_LABELS = {
    "green": "Sirloin steak",
    "red": "New York strip steak",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model-discovery panels with term contributions.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--epochs", type=int, default=4000, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=500, help="Early stopping patience")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--l1-penalty", type=float, default=0.0, help="L1 penalty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--decay-model", choices=["stretched", "double_exp"], default="stretched")
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--xla", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = RuntimeConfig(
        device_preference=args.device,
        enable_mixed_precision=args.mixed_precision,
        enable_xla=args.xla,
    )
    status = configure_tensorflow_runtime(runtime)
    print(runtime_summary_text(runtime, status))
    paths = ProjectPaths(root=args.root.resolve())
    raw_df = load_all_workbooks(
        paths.data_dir,
        axial_stress_column=args.axial_stress_column,
        compression_sign=args.compression_sign,
    )
    aggregated_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)

    config = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        l1_penalty=args.l1_penalty,
        random_seed=args.seed,
        decay_model=args.decay_model,
    )

    out_dir = paths.output_dir / "discovery_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_df.to_csv(out_dir / "robust_summary_curves.csv", index=False)
    interpolated_df.to_csv(out_dir / "interpolated_individual_curves.csv", index=False)
    diagnostics_df.to_csv(out_dir / "curve_diagnostics.csv", index=False)

    all_rows: list[dict] = []
    colors = sorted(raw_df["color"].unique())
    color_progress = tqdm(colors, desc="discovery colors", leave=True, unit="group")
    for color in color_progress:
        color_progress.set_postfix_str(color)
        payload = build_full_training_payload_from_curves(aggregated_df, color=color)
        trained_models = {}
        variant_progress = tqdm(TRAIN_VARIANTS, desc=f"{color} variants", leave=False, unit="model")
        for variant, _ in variant_progress:
            variant_progress.set_postfix_str(variant)
            model, history_df = fit_cann(payload, config=config, variant=variant)
            trained_models[variant] = model
            run_dir = out_dir / color / variant
            run_dir.mkdir(parents=True, exist_ok=True)
            history_df.to_csv(run_dir / "training_history.csv", index=False)
        variant_progress.close()

        time_values = sorted(aggregated_df.loc[aggregated_df["color"] == color, "time_days"].unique())
        time_progress = tqdm(time_values, desc=f"{color} figures", leave=False, unit="day")
        for time_days in time_progress:
            time_progress.set_postfix_str(f"day {int(time_days)}")
            figure_path = out_dir / color / f"discovery_{color}_day_{int(time_days)}.png"
            rows = _plot_time_panel(
                color=color,
                time_days=float(time_days),
                aggregated_df=aggregated_df,
                trained_models=trained_models,
                save_path=figure_path,
            )
            all_rows.extend(rows)
        time_progress.close()
    color_progress.close()

    pd.DataFrame(all_rows).to_csv(out_dir / "discovery_r2_summary.csv", index=False)


def _plot_time_panel(
    color: str,
    time_days: float,
    aggregated_df: pd.DataFrame,
    trained_models: dict[str, object],
    save_path: Path,
) -> list[dict]:
    color_map = matplotlib.colormaps["jet_r"].resampled(len(TERM_LABELS))
    cmaplist = [color_map(i) for i in range(color_map.N)]
    subset = aggregated_df.loc[(aggregated_df["color"] == color) & (aggregated_df["time_days"] == time_days)].copy()

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    summary_rows: list[dict] = []

    for row_idx, (loading, loading_title) in enumerate(ROW_LOADINGS):
        display_df = subset.loc[subset["loading"] == loading].sort_values("deformation")
        x = display_df["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1)
        t = np.full_like(x, fill_value=time_days, dtype=np.float32)
        y_true = display_df["stress_pa"].to_numpy(dtype=float)

        for col_idx, (variant, variant_title) in enumerate(TRAIN_VARIANTS):
            ax = axes[row_idx, col_idx]
            model = trained_models[variant]
            if loading == "shear":
                contrib = model.predict_shear_contributions(x, t)
                y_pred = contrib.sum(axis=1)
                xlabel = "shear strain [-]"
            else:
                contrib = model.predict_uniaxial_contributions(x, t)
                y_pred = contrib.sum(axis=1)
                xlabel = "stretch [-]"

            # For visualization only: flip panels with predominantly negative stress
            # so all contribution maps stack upward from a zero baseline.
            display_sign = -1.0 if np.nanmean(y_true) < 0.0 else 1.0
            y_true_plot = display_sign * y_true
            y_pred_plot = display_sign * y_pred
            contrib_plot = display_sign * contrib

            lower = np.zeros_like(y_pred_plot)
            for term_idx in range(contrib_plot.shape[1]):
                upper = lower + contrib_plot[:, term_idx]
                ax.fill_between(
                    x.reshape(-1),
                    lower,
                    upper,
                    color=cmaplist[term_idx],
                    linewidth=0.3,
                    edgecolor="k",
                )
                lower = upper
            ax.plot(x.reshape(-1), y_pred_plot, color="k", linewidth=1.0)
            ax.scatter(x.reshape(-1), y_true_plot, s=32, facecolors="white", edgecolors="gray", linewidth=0.8, zorder=5)

            score = r2_score(y_true, y_pred)
            summary_rows.append(
                {
                    "color": color,
                    "time_days": time_days,
                    "loading": loading,
                    "variant": variant,
                    "r2": score,
                }
            )

            ax.set_title(f"{COLOR_LABELS[color]} | {variant_title}", fontsize=11)
            ax.text(0.03, 0.90, f"R$^2$ = {score:.4f}", transform=ax.transAxes, fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("stress [Pa]")
            ymax = max(float(np.max(y_true_plot)), float(np.max(y_pred_plot)), 1.0)
            ax.set_ylim(0.0, ymax * 1.08)
            if loading == "compression":
                ax.invert_xaxis()
            ax.grid(alpha=0.15)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="gray", label="data")
    ]
    handles.extend(
        [plt.Rectangle((0, 0), 1, 1, facecolor=cmaplist[i], edgecolor="none", label=TERM_LABELS[i]) for i in range(len(TERM_LABELS))]
    )
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"Model discovery | {COLOR_LABELS[color]} | {TIME_LABELS.get(time_days, time_days)}", fontsize=18)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(save_path, dpi=240)
    plt.close(fig)
    return summary_rows


if __name__ == "__main__":
    main()
