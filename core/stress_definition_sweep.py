from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from core.data_pipeline import build_training_payload, load_all_workbooks, preprocess_robust_curves
from core.runtime import RuntimeConfig, configure_tensorflow_runtime, runtime_summary_text
from core.settings import ProjectPaths
from core.time_cann import TrainingConfig, evaluate_cann, fit_cann, save_training_artifacts


STRESS_VARIANTS = [
    {
        "name": "axial_stress",
        "axial_stress_column": "stress",
        "compression_sign": 1.0,
        "description": "Use axial Stress column for compression and tension.",
    },
    {
        "name": "axial_normal_stress",
        "axial_stress_column": "normal_stress",
        "compression_sign": 1.0,
        "description": "Use axial Normal stress column for compression and tension.",
    },
    {
        "name": "axial_stress_flip_compression",
        "axial_stress_column": "stress",
        "compression_sign": -1.0,
        "description": "Use axial Stress column and flip compression sign.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stress definitions for the beef CANN pipeline.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum training epochs per variant")
    parser.add_argument("--patience", type=int, default=200, help="Early stopping patience")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Adam learning rate")
    parser.add_argument("--l1-penalty", type=float, default=0.0, help="L1 penalty coefficient")
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--lr-reduce-patience", type=int, default=150)
    parser.add_argument("--min-learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-sample-num", type=int, default=5, help="Held-out sample number within each group")
    parser.add_argument(
        "--decay-model",
        choices=["stretched", "double_exp"],
        default="stretched",
        help="Time degradation model",
    )
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
    base_dir = paths.output_dir / "stress_definition_sweep"
    base_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        l1_penalty=args.l1_penalty,
        random_seed=args.seed,
        decay_model=args.decay_model,
        lr_reduce_factor=args.lr_reduce_factor,
        lr_reduce_patience=args.lr_reduce_patience,
        min_learning_rate=args.min_learning_rate,
    )

    all_metrics: list[pd.DataFrame] = []
    sweep_progress = tqdm(STRESS_VARIANTS, desc="stress variants", leave=True, unit="variant")
    for variant in sweep_progress:
        sweep_progress.set_postfix_str(variant["name"])
        raw_df = load_all_workbooks(
            paths.data_dir,
            axial_stress_column=variant["axial_stress_column"],
            compression_sign=variant["compression_sign"],
        )
        representative_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)

        variant_dir = base_dir / variant["name"]
        tables_dir = variant_dir / "tables"
        figures_dir = variant_dir / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        representative_df.to_csv(tables_dir / "robust_summary_curves.csv", index=False)
        interpolated_df.to_csv(tables_dir / "interpolated_individual_curves.csv", index=False)
        diagnostics_df.to_csv(tables_dir / "curve_diagnostics.csv", index=False)

        variant_metrics: list[pd.DataFrame] = []
        color_progress = tqdm(sorted(raw_df["color"].unique()), desc=f"{variant['name']} colors", leave=False, unit="group")
        for color in color_progress:
            color_progress.set_postfix_str(color)
            payload = build_training_payload(
                interpolated_df,
                color=color,
                random_state=args.seed,
                preferred_test_sample_num=args.test_sample_num,
            )
            model, history_df = fit_cann(payload, config=config, variant="weighted_all")
            predictions_df, metrics_df = evaluate_cann(model, payload, variant="weighted_all")

            run_dir = variant_dir / "models" / color
            save_training_artifacts(
                model,
                history_df,
                predictions_df,
                metrics_df,
                config,
                run_dir,
                split_info={
                    "train_files": payload["train_files"],
                    "test_files": payload["test_files"],
                    "stress_variant": variant["name"],
                },
            )

            metrics_out = metrics_df.copy()
            metrics_out.insert(0, "color", color)
            metrics_out.insert(0, "stress_variant", variant["name"])
            variant_metrics.append(metrics_out)

            _plot_color_predictions(
                predictions_df,
                figures_dir / f"{color}_weighted_all_predictions.png",
                title=f"{color} | {variant['name']}",
            )
        color_progress.close()

        metrics_table = pd.concat(variant_metrics, ignore_index=True)
        metrics_table.to_csv(tables_dir / "metrics_summary.csv", index=False)
        all_metrics.append(metrics_table)
    sweep_progress.close()

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(base_dir / "stress_definition_comparison.csv", index=False)
    (base_dir / "stress_definition_variants.json").write_text(
        json.dumps(STRESS_VARIANTS, indent=2),
        encoding="utf-8",
    )
    _plot_variant_comparison(combined, base_dir / "stress_definition_comparison.png")


def _plot_color_predictions(predictions_df: pd.DataFrame, save_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, loading in zip(axes, ["compression", "unloading", "shear"]):
        subset = predictions_df.loc[predictions_df["loading"] == loading]
        for time_days, group in subset.groupby("time_days", sort=True):
            ordered = group.sort_values("deformation")
            ax.scatter(ordered["deformation"], ordered["stress_pa"], s=10, alpha=0.45)
            ax.plot(ordered["deformation"], ordered["predicted_stress_pa"], linewidth=2.0, label=f"t={time_days:g}d")
        ax.set_title(loading)
        ax.set_xlabel("Deformation")
        ax.set_ylabel("Stress (Pa)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def _plot_variant_comparison(metrics_df: pd.DataFrame, save_path: Path) -> None:
    subset = metrics_df.loc[(metrics_df["loading"] == "overall") & (metrics_df["split"] == "test")].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x_labels = [f"{row.stress_variant}\n{row.color}" for row in subset.itertuples()]
    ax.bar(x_labels, subset["rmse_pa"])
    ax.set_ylabel("Test RMSE (Pa)")
    ax.set_title("Stress-definition comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
