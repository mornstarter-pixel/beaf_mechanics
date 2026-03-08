from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from core.data_pipeline import load_all_workbooks, preprocess_robust_curves
from core.settings import ProjectPaths

import tensorflow as tf


COLOR_LABELS = {
    "green": "Sirloin steak",
    "red": "New York strip steak",
}

TIME_LABELS = {
    0.0: "Day 0",
    1.0: "Day 1",
    2.0: "Day 2",
}


@dataclass
class FitConfig:
    epochs: int = 5000
    patience: int = 500
    learning_rate: float = 1e-2
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 250
    min_learning_rate: float = 1e-4
    seed: int = 42
    grad_clip_norm: float = 5.0


class StretchCANN(tf.Module):
    def __init__(
        self,
        powers: list[int],
        seed: int = 42,
        shear_poly_degrees: list[int] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "stretch_cann")
        self.powers = [int(p) for p in powers if p != 0]
        self.shear_poly_degrees = [int(d) for d in (shear_poly_degrees or []) if d > 0]
        init = tf.keras.initializers.GlorotNormal(seed=seed)
        self.coeffs = tf.Variable(
            init(shape=(len(self.powers) + len(self.shear_poly_degrees), 1), dtype=tf.float32),
            trainable=True,
            name="coeffs",
        )

    def _basis_uniaxial(self, stretch: tf.Tensor) -> tf.Tensor:
        terms = []
        for p in self.powers:
            p_float = tf.constant(float(p), dtype=tf.float32)
            term = tf.pow(stretch, p_float) + 2.0 * tf.pow(stretch, -0.5 * p_float) - 3.0
            terms.append(term)
        return tf.concat(terms, axis=1)

    def _principal_stretches_from_shear(self, gamma: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Closed-form principal stretches for simple shear are more stable than
        # recovering them from invariants.
        half_gamma = 0.5 * gamma
        root = tf.sqrt(1.0 + tf.square(half_gamma))
        l1 = root + half_gamma
        l2 = root - half_gamma
        l3 = tf.ones_like(gamma)
        return l1, l2, l3

    def _basis_shear(self, gamma: tf.Tensor) -> tf.Tensor:
        l1, l2, l3 = self._principal_stretches_from_shear(gamma)
        terms = []
        for p in self.powers:
            p_float = tf.constant(float(p), dtype=tf.float32)
            term = tf.pow(l1, p_float) + tf.pow(l2, p_float) + tf.pow(l3, p_float) - 3.0
            terms.append(term)
        for degree in self.shear_poly_degrees:
            degree_float = tf.constant(float(degree), dtype=tf.float32)
            terms.append(tf.pow(gamma, degree_float))
        return tf.concat(terms, axis=1)

    def energy_uniaxial(self, stretch: tf.Tensor) -> tf.Tensor:
        return tf.matmul(self._basis_uniaxial(stretch), self.coeffs)

    def energy_shear(self, gamma: tf.Tensor) -> tf.Tensor:
        return tf.matmul(self._basis_shear(gamma), self.coeffs)

    def predict_uniaxial(self, stretch: tf.Tensor) -> tf.Tensor:
        stretch = tf.convert_to_tensor(stretch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(stretch)
            psi = self.energy_uniaxial(stretch)
        return tape.gradient(psi, stretch)

    def predict_shear(self, gamma: tf.Tensor) -> tf.Tensor:
        gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(gamma)
            psi = self.energy_shear(gamma)
        return tape.gradient(psi, gamma)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return [self.coeffs]

    def export_weights(self) -> dict[str, object]:
        return {
            "powers": self.powers,
            "shear_poly_degrees": self.shear_poly_degrees,
            "coeffs": self.coeffs.numpy().reshape(-1).tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stretch-based CANN fit.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--color", choices=["green", "red"], default="red")
    parser.add_argument("--loading", choices=["compression", "unloading", "shear"], default="compression")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    parser.add_argument(
        "--power-mode",
        choices=["auto", "even", "all"],
        default="auto",
        help="Basis exponent set. 'auto' uses loading-specific defaults.",
    )
    return parser.parse_args()


def select_powers(loading: str, power_mode: str) -> list[int]:
    if power_mode == "even":
        return list(range(-10, 12, 2))
    if power_mode == "all":
        return list(range(-10, 11))
    if loading == "shear":
        return list(range(-10, 11))
    return list(range(-10, 12, 2))


def select_shear_poly_degrees(loading: str) -> list[int]:
    if loading != "shear":
        return []
    return [1, 2, 3, 4, 5, 6]


def main() -> None:
    args = parse_args()
    cfg = FitConfig(
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    paths = ProjectPaths(root=args.root.resolve())
    raw_df = load_all_workbooks(
        paths.data_dir,
        axial_stress_column=args.axial_stress_column,
        compression_sign=args.compression_sign,
    )
    summary_df, _, _ = preprocess_robust_curves(raw_df)
    subset = summary_df.loc[(summary_df["color"] == args.color) & (summary_df["loading"] == args.loading)].copy()
    if subset.empty:
        raise ValueError("No matching data for stretch CANN fit.")

    out_dir = paths.output_dir / "stretch_cann_fit" / args.color / args.loading
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics: list[dict] = []

    powers = select_powers(args.loading, args.power_mode)
    for ax, time_days in zip(axes, sorted(subset["time_days"].unique())):
        group = subset.loc[subset["time_days"] == time_days].sort_values("deformation").copy()
        x = group["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1)
        y_pa = group["stress_pa"].to_numpy(dtype=np.float32).reshape(-1, 1)
        y = y_pa / 1000.0

        model = StretchCANN(
            powers=powers,
            seed=cfg.seed + int(time_days * 10),
            shear_poly_degrees=select_shear_poly_degrees(args.loading),
        )
        history_df = fit_single_curve(model, x, y, args.loading, cfg)
        y_pred = predict_loading(model, x, args.loading).numpy().reshape(-1) * 1000.0
        score = r2_score(y_pa.reshape(-1), y_pred)
        rmse = np.sqrt(mean_squared_error(y_pa.reshape(-1), y_pred))
        metrics.append(
            {
                "color": args.color,
                "loading": args.loading,
                "time_days": float(time_days),
                "r2": float(score),
                "rmse_pa": float(rmse),
                "n_points": int(len(group)),
            }
        )
        history_df.to_csv(out_dir / f"{args.color}_{args.loading}_day_{int(time_days)}_history.csv", index=False)
        (out_dir / f"{args.color}_{args.loading}_day_{int(time_days)}_weights.json").write_text(
            json.dumps(model.export_weights(), indent=2),
            encoding="utf-8",
        )

        ax.scatter(x.reshape(-1), y_pa.reshape(-1), s=30, facecolors="white", edgecolors="gray", linewidth=0.9, label="data")
        ax.plot(x.reshape(-1), y_pred, color="#0b5ed7", linewidth=2.2, label="stretch CANN")
        if args.loading == "compression":
            ax.invert_xaxis()
        ax.set_title(TIME_LABELS.get(float(time_days), time_days))
        ax.text(0.03, 0.90, f"R$^2$ = {score:.4f}", transform=ax.transAxes, fontsize=10)
        ax.set_xlabel("stretch [-]" if args.loading != "shear" else "shear strain [-]")
        ax.set_ylabel("stress [Pa]")
        ax.grid(alpha=0.2)

    axes[0].legend(frameon=False)
    fig.suptitle(f"Stretch CANN | {COLOR_LABELS[args.color]} | {args.loading}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"{args.color}_{args.loading}_stretch_cann.png", dpi=240)
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(out_dir / f"{args.color}_{args.loading}_stretch_cann_metrics.csv", index=False)


def fit_single_curve(model: StretchCANN, x: np.ndarray, y: np.ndarray, loading: str, cfg: FitConfig) -> pd.DataFrame:
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    best_loss = np.inf
    best_weights = [var.numpy().copy() for var in model.trainable_variables]
    wait = 0
    lr_wait = 0
    history: list[dict[str, float]] = []

    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    weight_tensor = build_loss_weights(x_tensor, loading)

    for epoch in range(cfg.epochs):
        with tf.GradientTape() as tape:
            pred = predict_loading(model, x_tensor, loading)
            residual_sq = tf.square(y_tensor - pred)
            loss = tf.reduce_mean(weight_tensor * residual_sq)
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(grad, cfg.grad_clip_norm) for grad in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss = float(loss.numpy())
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "lr": float(optimizer.learning_rate.numpy())})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = [var.numpy().copy() for var in model.trainable_variables]
            wait = 0
            lr_wait = 0
        else:
            wait += 1
            lr_wait += 1
            if lr_wait >= cfg.lr_reduce_patience:
                current_lr = float(optimizer.learning_rate.numpy())
                new_lr = max(current_lr * cfg.lr_reduce_factor, cfg.min_learning_rate)
                if new_lr < current_lr:
                    optimizer.learning_rate.assign(new_lr)
                lr_wait = 0
            if wait >= cfg.patience:
                break

    for var, best in zip(model.trainable_variables, best_weights):
        var.assign(best)
    return pd.DataFrame(history)


def predict_loading(model: StretchCANN, x: tf.Tensor, loading: str) -> tf.Tensor:
    if loading == "shear":
        return model.predict_shear(x)
    return model.predict_uniaxial(x)


def build_loss_weights(x: tf.Tensor, loading: str) -> tf.Tensor:
    if loading != "shear":
        return tf.ones_like(x, dtype=tf.float32)
    gamma = tf.maximum(x, tf.constant(0.0, dtype=tf.float32))
    # Shear curves are most informative near the initial rise and first bend.
    return 1.0 + 3.0 * tf.exp(-gamma / 0.04)


if __name__ == "__main__":
    main()
