from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from core.data_pipeline import (
    build_training_payload,
    load_all_workbooks,
    preprocess_robust_curves,
    save_preprocessed_outputs,
)
from core.runtime import RuntimeConfig, configure_tensorflow_runtime, runtime_summary_text
from core.reporting import build_paper_style_outputs
from core.settings import ProjectPaths
from core.time_cann import TrainingConfig, evaluate_cann, fit_cann, save_training_artifacts

TRAINING_VARIANTS = [
    "compression_only",
    "unloading_only",
    "shear_only",
    "weighted_all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-dependent CANN analysis for beef mechanics.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--epochs", type=int, default=4000, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=500, help="Early stopping patience")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Adam learning rate")
    parser.add_argument("--l1-penalty", type=float, default=0.0, help="L1 penalty coefficient")
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5, help="Factor applied when reducing learning rate")
    parser.add_argument("--lr-reduce-patience", type=int, default=250, help="Epochs without validation improvement before lowering learning rate")
    parser.add_argument("--min-learning-rate", type=float, default=1e-4, help="Lower bound for automatic learning-rate decay")
    parser.add_argument(
        "--decay-model",
        choices=["stretched", "double_exp"],
        default="stretched",
        help="Time degradation model",
    )
    parser.add_argument(
        "--axial-stress-column",
        choices=["stress", "normal_stress"],
        default="stress",
        help="Axial stress definition for compression/tension sheets",
    )
    parser.add_argument(
        "--compression-sign",
        type=float,
        default=-1.0,
        help="Sign multiplier applied to compression stress",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--test-sample-num",
        type=int,
        default=5,
        help="Per freshness group, reserve this sample number for testing when available",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="TensorFlow device preference")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision when a GPU is available",
    )
    parser.add_argument("--xla", action="store_true", help="Enable TensorFlow XLA JIT compilation")
    return parser.parse_args()


def execute_analysis(args: argparse.Namespace) -> None:
    paths = ProjectPaths(root=args.root.resolve())
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_all_workbooks(
        paths.data_dir,
        axial_stress_column=args.axial_stress_column,
        compression_sign=args.compression_sign,
    )
    representative_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)
    save_preprocessed_outputs(raw_df, representative_df, diagnostics_df, paths.tables_dir)

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

    model_results: dict[str, dict[str, dict[str, object]]] = {}
    colors = sorted(representative_df["color"].unique())
    color_progress = tqdm(colors, desc="colors", leave=True, unit="group")
    for color in color_progress:
        color_progress.set_postfix_str(color)
        payload = build_training_payload(
            interpolated_df,
            color=color,
            random_state=args.seed,
            preferred_test_sample_num=args.test_sample_num,
        )
        model_results[color] = {}
        variant_progress = tqdm(TRAINING_VARIANTS, desc=f"{color} variants", leave=False, unit="model")
        for variant in variant_progress:
            variant_progress.set_postfix_str(variant)
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
        variant_progress.close()
    color_progress.close()

    build_paper_style_outputs(
        aggregated_df=representative_df,
        model_results=model_results,
        figures_dir=paths.figures_dir,
        tables_dir=paths.tables_dir,
    )


def main() -> None:
    args = parse_args()
    runtime = RuntimeConfig(
        device_preference=args.device,
        enable_mixed_precision=args.mixed_precision,
        enable_xla=args.xla,
    )
    status = configure_tensorflow_runtime(runtime)
    print(runtime_summary_text(runtime, status))
    execute_analysis(args)


if __name__ == "__main__":
    main()
