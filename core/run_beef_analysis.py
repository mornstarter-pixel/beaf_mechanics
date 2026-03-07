from __future__ import annotations

import argparse
from pathlib import Path

from core.data_pipeline import (
    aggregate_group_curves,
    build_training_payload,
    load_all_workbooks,
    save_preprocessed_outputs,
)
from core.reporting import build_paper_style_outputs
from core.settings import ProjectPaths
from core.time_cann import TrainingConfig, evaluate_cann, fit_cann, save_training_artifacts

TRAINING_VARIANTS = [
    "compression_only",
    "tension_only",
    "shear_only",
    "weighted_all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-dependent CANN analysis for beef mechanics.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--epochs", type=int, default=4000, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=500, help="Early stopping patience")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--l1-penalty", type=float, default=0.0, help="L1 penalty coefficient")
    parser.add_argument(
        "--decay-model",
        choices=["stretched", "double_exp"],
        default="stretched",
        help="Time degradation model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--test-sample-num",
        type=int,
        default=5,
        help="Per freshness group, reserve this sample number for testing when available",
    )
    return parser.parse_args()


def execute_analysis(args: argparse.Namespace) -> None:
    paths = ProjectPaths(root=args.root.resolve())
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_all_workbooks(paths.data_dir)
    aggregated_df, outlier_df = aggregate_group_curves(raw_df)
    save_preprocessed_outputs(raw_df, aggregated_df, outlier_df, paths.tables_dir)

    config = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        l1_penalty=args.l1_penalty,
        random_seed=args.seed,
        decay_model=args.decay_model,
    )

    model_results: dict[str, dict[str, dict[str, object]]] = {}
    for color in sorted(aggregated_df["color"].unique()):
        payload = build_training_payload(
            outlier_df,
            color=color,
            random_state=args.seed,
            preferred_test_sample_num=args.test_sample_num,
        )
        model_results[color] = {}
        for variant in TRAINING_VARIANTS:
            model, history_df = fit_cann(payload, config=config, variant=variant)
            predictions_df, metrics_df = evaluate_cann(model, payload, variant=variant)
            run_dir = paths.models_dir / color / args.decay_model / variant
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
                    "train_variant": variant,
                },
            )
            model_results[color][variant] = {
                "predictions": predictions_df,
                "metrics": metrics_df,
            }

    build_paper_style_outputs(
        aggregated_df=aggregated_df,
        model_results=model_results,
        figures_dir=paths.figures_dir,
        tables_dir=paths.tables_dir,
    )


def main() -> None:
    args = parse_args()
    execute_analysis(args)


if __name__ == "__main__":
    main()
