from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

from core.data_pipeline import load_all_workbooks, preprocess_robust_curves
from core.settings import ProjectPaths


COLOR_LABELS = {
    "green": "Sirloin steak",
    "red": "New York strip steak",
}

TIME_LABELS = {
    0.0: "Day 0",
    1.0: "Day 1",
    2.0: "Day 2",
}

LOADING_LABELS = {
    "compression": "compression",
    "unloading": "unloading",
    "shear": "shear",
}

TERM_LABELS = [
    "I1b",
    "I1b^2",
    "I1b^3",
    "I1b^4",
    "exp(I1b)-1",
    "log(1+I1b)",
    "I2b",
    "I2b^2",
    "I2b^3",
    "I2b^4",
    "exp(I2b)-1",
    "log(1+I2b)",
    "I1b*I2b",
    "I1b^2*I2b",
    "I1b*I2b^2",
]


@dataclass
class InvariantBaselineConfig:
    epochs: int = 4000
    patience: int = 600
    learning_rate: float = 3e-3
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 250
    min_learning_rate: float = 1e-4
    l1_penalty: float = 1e-4
    seed: int = 42
    grad_clip_norm: float = 5.0
    test_sample_num: int = 5


class PolynomialInvariantCANN(tf.Module):
    def __init__(self, seed: int = 42, name: str | None = None) -> None:
        super().__init__(name=name or "poly_invariant_cann")
        init = tf.keras.initializers.GlorotNormal(seed=seed)
        self.coeffs = tf.Variable(init(shape=(len(TERM_LABELS), 1), dtype=tf.float32), trainable=True, name="coeffs")

    def features(self, i1: tf.Tensor, i2: tf.Tensor) -> tf.Tensor:
        i1b = i1 - 3.0
        i2b = i2 - 3.0
        return tf.concat(
            [
                i1b,
                tf.square(i1b),
                tf.pow(i1b, 3.0),
                tf.pow(i1b, 4.0),
                tf.math.exp(tf.clip_by_value(i1b, 0.0, 3.0)) - 1.0,
                tf.math.log1p(tf.clip_by_value(i1b, 0.0, 1e6)),
                i2b,
                tf.square(i2b),
                tf.pow(i2b, 3.0),
                tf.pow(i2b, 4.0),
                tf.math.exp(tf.clip_by_value(i2b, 0.0, 3.0)) - 1.0,
                tf.math.log1p(tf.clip_by_value(i2b, 0.0, 1e6)),
                i1b * i2b,
                tf.square(i1b) * i2b,
                i1b * tf.square(i2b),
            ],
            axis=1,
        )

    def energy(self, i1: tf.Tensor, i2: tf.Tensor) -> tf.Tensor:
        return tf.matmul(self.features(i1, i2), self.coeffs)

    def predict_uniaxial(self, stretch: tf.Tensor) -> tf.Tensor:
        stretch = tf.convert_to_tensor(stretch, dtype=tf.float32)
        i1 = tf.square(stretch) + 2.0 / stretch
        i2 = 2.0 * stretch + 1.0 / tf.square(stretch)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([i1, i2])
            psi = self.energy(i1, i2)
        dpsi_di1 = tape.gradient(psi, i1)
        dpsi_di2 = tape.gradient(psi, i2)
        del tape
        minus = 2.0 * (dpsi_di1 / tf.square(stretch) + dpsi_di2 / tf.pow(stretch, 3.0))
        return 2.0 * (dpsi_di1 * stretch + dpsi_di2) - minus

    def predict_shear(self, gamma: tf.Tensor) -> tf.Tensor:
        gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        i1 = tf.square(gamma) + 3.0
        i2 = tf.square(gamma) + 3.0
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([i1, i2])
            psi = self.energy(i1, i2)
        dpsi_di1 = tape.gradient(psi, i1)
        dpsi_di2 = tape.gradient(psi, i2)
        del tape
        return 2.0 * gamma * (dpsi_di1 + dpsi_di2)

    def predict_uniaxial_contributions(self, stretch: np.ndarray) -> np.ndarray:
        return self._predict_contributions(stretch, loading="uniaxial")

    def predict_shear_contributions(self, gamma: np.ndarray) -> np.ndarray:
        return self._predict_contributions(gamma, loading="shear")

    def _predict_contributions(self, deformation: np.ndarray, loading: str) -> np.ndarray:
        x = tf.convert_to_tensor(deformation.reshape(-1, 1), dtype=tf.float32)
        outputs = []
        for idx in range(len(TERM_LABELS)):
            mask = np.zeros((len(TERM_LABELS), 1), dtype=np.float32)
            mask[idx, 0] = 1.0
            coeffs = tf.convert_to_tensor(mask, dtype=tf.float32) * self.coeffs
            with tf.GradientTape() as tape:
                tape.watch(x)
                if loading == "shear":
                    i1 = tf.square(x) + 3.0
                    i2 = tf.square(x) + 3.0
                    psi = tf.matmul(self.features(i1, i2), coeffs)
                else:
                    i1 = tf.square(x) + 2.0 / x
                    i2 = 2.0 * x + 1.0 / tf.square(x)
                    psi = tf.matmul(self.features(i1, i2), coeffs)
            if loading == "shear":
                grad = tape.gradient(psi, x)
            else:
                grad = tape.gradient(psi, x)
            outputs.append(grad.numpy().reshape(-1))
        return np.stack(outputs, axis=1)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return [self.coeffs]

    def export_weights(self) -> dict[str, object]:
        return {
            "terms": TERM_LABELS,
            "coeffs": self.coeffs.numpy().reshape(-1).tolist(),
        }

    def set_coefficients(self, coeffs: np.ndarray) -> None:
        self.coeffs.assign(np.asarray(coeffs, dtype=np.float32).reshape(-1, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invariant CANN single-loading baseline.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--l1-penalty", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-sample-num", type=int, default=5)
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    return parser.parse_args()


def _split_files(color_df: pd.DataFrame, test_sample_num: int) -> tuple[set[str], set[str]]:
    unique_files = color_df[["file_name", "freshness_index", "sample_num"]].drop_duplicates()
    test_files: set[str] = set()
    train_files: set[str] = set()
    for freshness_index, group in unique_files.groupby("freshness_index", sort=True):
        candidates = group.sort_values("sample_num")
        preferred = candidates.loc[candidates["sample_num"] == test_sample_num, "file_name"].tolist()
        test_file = preferred[0] if preferred else candidates.iloc[-1]["file_name"]
        test_files.add(test_file)
        train_files.update(set(group["file_name"].tolist()) - {test_file})
    return train_files, test_files


def _prepare_loading_frame(interpolated_df: pd.DataFrame, color: str, loading: str) -> tuple[pd.DataFrame, set[str], set[str]]:
    color_df = interpolated_df.loc[(interpolated_df["color"] == color) & (~interpolated_df["is_curve_outlier"])].copy()
    if color_df.empty:
        raise ValueError(f"No sample-level curves after filtering for color={color}")
    train_files, test_files = _split_files(color_df, test_sample_num=5)
    frame = color_df.loc[color_df["loading"] == loading].copy()
    if frame.empty:
        raise ValueError(f"No rows for color={color}, loading={loading}")
    frame["split"] = np.where(frame["file_name"].isin(list(test_files)), "test", "train")
    return frame, train_files, test_files


def _design_matrix(model: PolynomialInvariantCANN, x: np.ndarray, loading: str) -> np.ndarray:
    basis_columns = []
    for idx in range(len(TERM_LABELS)):
        coeffs = np.zeros((len(TERM_LABELS), 1), dtype=np.float32)
        coeffs[idx, 0] = 1.0
        model.set_coefficients(coeffs)
        if loading == "shear":
            pred = model.predict_shear(x).numpy().reshape(-1)
        else:
            pred = model.predict_uniaxial(x).numpy().reshape(-1)
        basis_columns.append(pred)
    return np.stack(basis_columns, axis=1)


def _weighted_ridge(X: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    sqrt_w = np.sqrt(np.clip(weights, 1e-8, None))[:, None]
    Xw = X * sqrt_w
    yw = y * sqrt_w[:, 0]
    gram = Xw.T @ Xw + alpha * np.eye(X.shape[1], dtype=float)
    rhs = Xw.T @ yw
    return np.linalg.solve(gram, rhs)


def fit_day_model(
    summary_train_df: pd.DataFrame,
    loading: str,
    config: InvariantBaselineConfig,
) -> tuple[PolynomialInvariantCANN, pd.DataFrame]:
    model = PolynomialInvariantCANN(seed=config.seed)
    x = summary_train_df["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y = summary_train_df["stress_pa"].to_numpy(dtype=float)
    X = _design_matrix(model, x, loading)
    coeffs = _weighted_ridge(X, y, np.ones(len(y), dtype=float), alpha=max(config.l1_penalty, 1e-6))
    model.set_coefficients(coeffs)
    pred = X @ coeffs
    history_df = pd.DataFrame(
        [
            {
                "epoch": 1,
                "loss": float(np.mean((y - pred) ** 2)),
                "lr": 0.0,
            }
        ]
    )
    return model, history_df


def evaluate_day_model(
    model: PolynomialInvariantCANN,
    summary_frame: pd.DataFrame,
    sample_frame: pd.DataFrame,
    loading: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    ordered = sample_frame.sort_values(["split", "file_name", "point_index"]).copy()
    x = ordered["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1)
    if loading == "shear":
        pred = model.predict_shear(x).numpy().reshape(-1)
    else:
        pred = model.predict_uniaxial(x).numpy().reshape(-1)
    ordered["predicted_stress_pa"] = pred
    summary_ordered = summary_frame.sort_values("deformation").copy()
    xs = summary_ordered["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1)
    if loading == "shear":
        summary_pred = model.predict_shear(xs).numpy().reshape(-1)
    else:
        summary_pred = model.predict_uniaxial(xs).numpy().reshape(-1)
    summary_ordered["predicted_stress_pa"] = summary_pred

    test_df = ordered.loc[ordered["split"] == "test"]
    train_df = ordered.loc[ordered["split"] == "train"]
    test_r2 = float(r2_score(test_df["stress_pa"], test_df["predicted_stress_pa"])) if not test_df.empty else float("nan")
    test_rmse = (
        float(np.sqrt(mean_squared_error(test_df["stress_pa"], test_df["predicted_stress_pa"]))) if not test_df.empty else float("nan")
    )
    metrics = {
        "summary_r2": float(r2_score(summary_ordered["stress_pa"], summary_ordered["predicted_stress_pa"])),
        "train_r2": float(r2_score(train_df["stress_pa"], train_df["predicted_stress_pa"])) if not train_df.empty else float("nan"),
        "test_r2": test_r2,
        "summary_rmse_pa": float(np.sqrt(mean_squared_error(summary_ordered["stress_pa"], summary_ordered["predicted_stress_pa"]))),
        "train_rmse_pa": float(np.sqrt(mean_squared_error(train_df["stress_pa"], train_df["predicted_stress_pa"])))
        if not train_df.empty
        else float("nan"),
        "test_rmse_pa": test_rmse,
    }
    return ordered, metrics


def _plot_day_fit(frame: pd.DataFrame, loading: str, color: str, time_days: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for split, group in frame.groupby("split", sort=False):
        for _, sample in group.groupby("file_name", sort=False):
            ordered = sample.sort_values("deformation")
            ax.scatter(
                ordered["deformation"],
                ordered["stress_pa"],
                s=16 if split == "train" else 28,
                alpha=0.55 if split == "train" else 1.0,
                facecolors="none" if split == "train" else "white",
                edgecolors="#888888" if split == "train" else "#222222",
                linewidths=0.7,
            )
    ordered = frame.sort_values("deformation")
    ax.plot(ordered["deformation"], ordered["predicted_stress_pa"], color="#0b5ed7", linewidth=2.2, label="prediction")
    if loading == "compression":
        ax.invert_xaxis()
    ax.set_title(f"{COLOR_LABELS[color]} | {LOADING_LABELS[loading]} | {TIME_LABELS[time_days]}")
    ax.set_xlabel("stretch [-]" if loading != "shear" else "shear strain [-]")
    ax.set_ylabel("stress [Pa]")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def run_baseline(config: InvariantBaselineConfig, root: Path, axial_stress_column: str, compression_sign: float) -> pd.DataFrame:
    paths = ProjectPaths(root=root)
    raw_df = load_all_workbooks(
        paths.data_dir,
        axial_stress_column=axial_stress_column,
        compression_sign=compression_sign,
    )
    summary_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)

    out_dir = paths.output_dir / "invariant_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "robust_summary_curves.csv", index=False)
    interpolated_df.to_csv(out_dir / "interpolated_individual_curves.csv", index=False)
    diagnostics_df.to_csv(out_dir / "curve_diagnostics.csv", index=False)

    all_metrics: list[dict] = []
    for color in sorted(interpolated_df["color"].unique()):
        for loading in ["compression", "shear", "unloading"]:
            frame, train_files, test_files = _prepare_loading_frame(interpolated_df, color, loading)
            for time_days in sorted(frame["time_days"].unique()):
                day_frame = frame.loc[frame["time_days"] == time_days].copy()
                summary_day = summary_df.loc[
                    (summary_df["color"] == color) & (summary_df["loading"] == loading) & (summary_df["time_days"] == time_days)
                ].copy()
                if summary_day.empty:
                    continue
                model, history_df = fit_day_model(summary_day, loading, config)
                pred_df, metrics = evaluate_day_model(model, summary_day, day_frame, loading)
                run_dir = out_dir / color / loading / f"day_{int(time_days)}"
                run_dir.mkdir(parents=True, exist_ok=True)
                history_df.to_csv(run_dir / "training_history.csv", index=False)
                pred_df.to_csv(run_dir / "predictions.csv", index=False)
                (run_dir / "weights.json").write_text(json.dumps(model.export_weights(), indent=2), encoding="utf-8")
                (run_dir / "data_split.json").write_text(
                    json.dumps({"train_files": sorted(train_files), "test_files": sorted(test_files)}, indent=2),
                    encoding="utf-8",
                )
                (run_dir / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
                _plot_day_fit(pred_df, loading, color, float(time_days), run_dir / "fit.png")
                all_metrics.append(
                    {
                        "color": color,
                        "loading": loading,
                        "time_days": float(time_days),
                        **metrics,
                    }
                )

    metrics_df = pd.DataFrame(all_metrics).sort_values(["color", "loading", "time_days"]).reset_index(drop=True)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    return metrics_df


def main() -> None:
    args = parse_args()
    config = InvariantBaselineConfig(
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        l1_penalty=args.l1_penalty,
        seed=args.seed,
        test_sample_num=args.test_sample_num,
    )
    metrics_df = run_baseline(
        config=config,
        root=args.root.resolve(),
        axial_stress_column=args.axial_stress_column,
        compression_sign=args.compression_sign,
    )
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
