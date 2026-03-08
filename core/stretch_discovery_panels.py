from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from core.data_pipeline import load_all_workbooks, preprocess_robust_curves
from core.settings import ProjectPaths
from core.stretch_time_cann import StretchTimeTrainingConfig, fit_stretch_time_cann


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

TIME_LABELS = {0.0: "Day 0", 1.0: "Day 1", 2.0: "Day 2"}
COLOR_LABELS = {"green": "Sirloin steak", "red": "New York strip steak"}
TOP_K_TERMS = 6


def _term_labels(model: object) -> list[str]:
    labels = [f"lambda^{p}" for p in model.powers]
    labels.extend([f"gamma^{degree}" for degree in model.shear_poly_degrees])
    return labels


def _figure_term_order(trained_models: dict[str, object]) -> list[str]:
    ordered: list[str] = []
    for variant, _ in TRAIN_VARIANTS:
        labels = _term_labels(trained_models[variant])
        for label in labels:
            if label not in ordered:
                ordered.append(label)
    return ordered


def _compress_contributions(contrib: np.ndarray, labels: list[str], top_k: int = TOP_K_TERMS) -> tuple[np.ndarray, list[str]]:
    if contrib.shape[1] <= top_k:
        return contrib, labels
    importance = np.mean(np.abs(contrib), axis=0)
    keep_idx = np.argsort(importance)[::-1][:top_k]
    keep_idx = np.sort(keep_idx)
    kept = contrib[:, keep_idx]
    kept_labels = [labels[idx] for idx in keep_idx]
    dropped_idx = [idx for idx in range(contrib.shape[1]) if idx not in set(keep_idx.tolist())]
    others = contrib[:, dropped_idx].sum(axis=1, keepdims=True)
    return np.concatenate([kept, others], axis=1), kept_labels + ["others"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stretch-time CANN discovery panels.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(root=args.root.resolve())
    raw_df = load_all_workbooks(
        paths.data_dir,
        axial_stress_column=args.axial_stress_column,
        compression_sign=args.compression_sign,
    )
    aggregated_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)

    config = StretchTimeTrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    out_dir = paths.output_dir / "stretch_discovery_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_df.to_csv(out_dir / "robust_summary_curves.csv", index=False)
    interpolated_df.to_csv(out_dir / "interpolated_individual_curves.csv", index=False)
    diagnostics_df.to_csv(out_dir / "curve_diagnostics.csv", index=False)

    summary_rows: list[dict] = []
    color_progress = tqdm(sorted(aggregated_df["color"].unique()), desc="stretch discovery colors", unit="group")
    for color in color_progress:
        color_progress.set_postfix_str(color)
        color_df = aggregated_df.loc[aggregated_df["color"] == color].copy()
        trained_models = {}

        variant_progress = tqdm(TRAIN_VARIANTS, desc=f"{color} variants", leave=False, unit="model")
        for variant, _ in variant_progress:
            variant_progress.set_postfix_str(variant)
            model, history_df = fit_stretch_time_cann(color_df, variant=variant, config=config)
            trained_models[variant] = model
            run_dir = out_dir / color / variant
            run_dir.mkdir(parents=True, exist_ok=True)
            history_df.to_csv(run_dir / "training_history.csv", index=False)
            (run_dir / "weights.json").write_text(json.dumps(model.export_weights(), indent=2), encoding="utf-8")
        variant_progress.close()

        for time_days in sorted(color_df["time_days"].unique()):
            figure_path = out_dir / color / f"stretch_discovery_{color}_day_{int(time_days)}.png"
            rows = _plot_time_panel(
                color=color,
                time_days=float(time_days),
                aggregated_df=color_df,
                trained_models=trained_models,
                save_path=figure_path,
            )
            summary_rows.extend(rows)
    color_progress.close()

    pd.DataFrame(summary_rows).to_csv(out_dir / "stretch_discovery_r2_summary.csv", index=False)


def _plot_time_panel(
    color: str,
    time_days: float,
    aggregated_df: pd.DataFrame,
    trained_models: dict[str, object],
    save_path: Path,
) -> list[dict]:
    subset = aggregated_df.loc[aggregated_df["time_days"] == time_days].copy()
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    rows: list[dict] = []
    ordered_labels = _figure_term_order(trained_models)
    ordered_labels.append("others")
    cmap = plt.cm.turbo(np.linspace(0.03, 0.97, len(ordered_labels)))
    label_to_color = {label: cmap[idx] for idx, label in enumerate(ordered_labels)}

    for row_idx, (loading, _) in enumerate(ROW_LOADINGS):
        display_df = subset.loc[subset["loading"] == loading].sort_values("deformation")
        x = display_df["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1)
        t = np.full_like(x, fill_value=time_days, dtype=np.float32)
        y_true = display_df["stress_pa"].to_numpy(dtype=float)

        for col_idx, (variant, variant_title) in enumerate(TRAIN_VARIANTS):
            ax = axes[row_idx, col_idx]
            model = trained_models[variant]
            term_labels = _term_labels(model)
            if loading == "shear":
                contrib = model.predict_shear_contributions(x, t) * 1000.0
                y_pred = contrib.sum(axis=1)
                xlabel = "shear strain [-]"
            else:
                contrib = model.predict_uniaxial_contributions(x, t) * 1000.0
                y_pred = contrib.sum(axis=1)
                xlabel = "stretch [-]"
            contrib, term_labels = _compress_contributions(contrib, term_labels)

            display_sign = -1.0 if np.nanmean(y_true) < 0.0 else 1.0
            y_true_plot = display_sign * y_true
            y_pred_plot = display_sign * y_pred
            contrib_plot = display_sign * contrib

            pos_lower = np.zeros_like(y_pred_plot)
            neg_lower = np.zeros_like(y_pred_plot)
            for term_idx in range(contrib_plot.shape[1]):
                term = contrib_plot[:, term_idx]
                term_color = label_to_color[term_labels[term_idx]]
                pos_term = np.clip(term, 0.0, None)
                neg_term = np.clip(term, None, 0.0)
                if np.any(pos_term > 0):
                    pos_upper = pos_lower + pos_term
                    ax.fill_between(x.reshape(-1), pos_lower, pos_upper, color=term_color, linewidth=0.0, alpha=0.9)
                    pos_lower = pos_upper
                if np.any(neg_term < 0):
                    neg_upper = neg_lower + neg_term
                    ax.fill_between(x.reshape(-1), neg_lower, neg_upper, color=term_color, linewidth=0.0, alpha=0.9)
                    neg_lower = neg_upper
            ax.plot(x.reshape(-1), y_pred_plot, color="k", linewidth=1.0)
            ax.scatter(x.reshape(-1), y_true_plot, s=28, facecolors="white", edgecolors="gray", linewidth=0.8, zorder=5)
            ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)

            score = r2_score(y_true, y_pred)
            rows.append(
                {
                    "color": color,
                    "time_days": time_days,
                    "loading": loading,
                    "variant": variant,
                    "r2": float(score),
                }
            )

            ax.set_title(f"{COLOR_LABELS[color]} | {variant_title}", fontsize=11)
            ax.text(0.03, 0.90, f"R$^2$ = {score:.4f}", transform=ax.transAxes, fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("stress [Pa]")
            ymax = max(float(np.max(y_true_plot)), float(np.max(y_pred_plot)), float(np.max(pos_lower)), 1.0)
            ymin = min(float(np.min(y_true_plot)), float(np.min(y_pred_plot)), float(np.min(neg_lower)), 0.0)
            pad = 0.08 * max(ymax - ymin, 1.0)
            ax.set_ylim(ymin - pad, ymax + pad)
            if loading == "compression":
                ax.invert_xaxis()
            ax.grid(alpha=0.15)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="gray", label="data")
    ]
    legend_handles.extend(
        [mpatches.Patch(facecolor=label_to_color[label], edgecolor="none", label=label) for label in ordered_labels]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(8, max(4, len(legend_handles))),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(f"Stretch-time discovery | {COLOR_LABELS[color]} | {TIME_LABELS.get(time_days, time_days)}", fontsize=18)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(save_path, dpi=240)
    plt.close(fig)
    return rows


if __name__ == "__main__":
    main()
