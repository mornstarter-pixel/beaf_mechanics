from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow is required for the CANN scripts. Install dependencies from requirements.txt."
    ) from exc


@dataclass
class TrainingConfig:
    epochs: int = 4000
    learning_rate: float = 1e-3
    l1_penalty: float = 0.0
    patience: int = 500
    random_seed: int = 42
    decay_model: str = "stretched"
    grad_clip_norm: float = 5.0


VARIANT_LABELS = {
    "compression_only": "compression",
    "unloading_only": "unloading",
    "shear_only": "shear",
    "weighted_all": "weighted_all",
}


def _softplus_var(shape: tuple[int, ...], seed: int, name: str) -> tf.Variable:
    init = tf.keras.initializers.GlorotNormal(seed=seed)
    return tf.Variable(init(shape=shape, dtype=tf.float32), name=name, trainable=True)


class InvariantEnergy(tf.Module):
    def __init__(self, seed: int = 42, name: str | None = None) -> None:
        super().__init__(name=name)
        self.w1_i1 = _softplus_var((6, 1), seed + 1, "w1_i1_raw")
        self.w1_i2 = _softplus_var((6, 1), seed + 2, "w1_i2_raw")
        self.w2 = _softplus_var((12, 1), seed + 3, "w2_raw")

    @staticmethod
    def _activation_exp(x: tf.Tensor) -> tf.Tensor:
        return tf.math.exp(x) - 1.0

    @staticmethod
    def _activation_log(x: tf.Tensor) -> tf.Tensor:
        x = tf.clip_by_value(x, -1e6, 0.95)
        return -tf.math.log(1.0 - x)

    def _single_invariant_features(self, inv_ref: tf.Tensor, weights_raw: tf.Variable) -> tf.Tensor:
        weights = tf.nn.softplus(weights_raw)
        linear = weights[0] * inv_ref
        exp_linear = self._activation_exp(weights[1] * inv_ref)
        log_linear = self._activation_log(weights[2] * inv_ref)
        sq = tf.square(inv_ref)
        linear_sq = weights[3] * sq
        exp_sq = self._activation_exp(weights[4] * sq)
        log_sq = self._activation_log(weights[5] * sq)
        return tf.concat([linear, exp_linear, log_linear, linear_sq, exp_sq, log_sq], axis=1)

    def features(self, i1: tf.Tensor, i2: tf.Tensor) -> tf.Tensor:
        i1_ref = i1 - 3.0
        i2_ref = i2 - 3.0
        return tf.concat(
            [
                self._single_invariant_features(i1_ref, self.w1_i1),
                self._single_invariant_features(i2_ref, self.w1_i2),
            ],
            axis=1,
        )

    def energy(self, i1: tf.Tensor, i2: tf.Tensor) -> tf.Tensor:
        outer = tf.nn.softplus(self.w2)
        return tf.matmul(self.features(i1, i2), outer)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return [self.w1_i1, self.w1_i2, self.w2]


class TimeDecay(tf.Module):
    def __init__(self, decay_model: str = "stretched", seed: int = 42, name: str | None = None) -> None:
        super().__init__(name=name)
        if decay_model not in {"stretched", "double_exp"}:
            raise ValueError("decay_model must be 'stretched' or 'double_exp'")
        self.decay_model = decay_model
        self.rate_raw = _softplus_var((1, 1), seed + 10, "rate_raw")
        self.beta_raw = tf.Variable(tf.zeros((1, 1), dtype=tf.float32), name="beta_raw", trainable=True)
        self.rate_fast_raw = _softplus_var((1, 1), seed + 11, "rate_fast_raw")
        self.rate_slow_raw = _softplus_var((1, 1), seed + 12, "rate_slow_raw")
        self.mix_logits = tf.Variable(tf.zeros((1, 2), dtype=tf.float32), name="mix_logits", trainable=True)

    def phi(self, time_days: tf.Tensor) -> tf.Tensor:
        t = tf.clip_by_value(time_days, 0.0, 1e6)
        if self.decay_model == "stretched":
            rate = tf.nn.softplus(self.rate_raw)
            beta = tf.clip_by_value(tf.nn.sigmoid(self.beta_raw), 1e-3, 1.0)
            # Avoid undefined gradients at t=0 for the stretching exponent term.
            safe_scaled_time = tf.maximum(rate * t, 1e-8)
            return tf.math.exp(-tf.pow(safe_scaled_time, beta))

        rate_fast = tf.nn.softplus(self.rate_fast_raw)
        rate_slow = tf.nn.softplus(self.rate_slow_raw)
        amps = tf.nn.softmax(self.mix_logits, axis=1)
        return amps[:, :1] * tf.math.exp(-rate_fast * t) + amps[:, 1:] * tf.math.exp(-rate_slow * t)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        if self.decay_model == "stretched":
            return [self.rate_raw, self.beta_raw]
        return [self.rate_fast_raw, self.rate_slow_raw, self.mix_logits]

    def params_dict(self) -> dict[str, float]:
        if self.decay_model == "stretched":
            return {
                "rate": float(tf.nn.softplus(self.rate_raw).numpy().squeeze()),
                "beta": float(tf.clip_by_value(tf.nn.sigmoid(self.beta_raw), 1e-3, 1.0).numpy().squeeze()),
            }
        amps = tf.nn.softmax(self.mix_logits, axis=1).numpy().reshape(-1)
        return {
            "amp_fast": float(amps[0]),
            "amp_slow": float(amps[1]),
            "rate_fast": float(tf.nn.softplus(self.rate_fast_raw).numpy().squeeze()),
            "rate_slow": float(tf.nn.softplus(self.rate_slow_raw).numpy().squeeze()),
        }


class TimeDependentCANN(tf.Module):
    def __init__(self, decay_model: str = "stretched", seed: int = 42, energy_scale_init: float = 1.0) -> None:
        super().__init__(name="time_dependent_cann")
        tf.random.set_seed(seed)
        self.energy_net = InvariantEnergy(seed=seed)
        self.time_decay = TimeDecay(decay_model=decay_model, seed=seed)
        init_value = max(float(energy_scale_init), 1.0)
        self.energy_scale_raw = tf.Variable(
            tf.math.log(tf.math.expm1(tf.constant(init_value, dtype=tf.float32))),
            name="energy_scale_raw",
            trainable=True,
        )

    def _energy_scale(self) -> tf.Tensor:
        return tf.clip_by_value(tf.nn.softplus(self.energy_scale_raw), 1.0, 500.0)

    def total_energy(self, i1: tf.Tensor, i2: tf.Tensor, time_days: tf.Tensor) -> tf.Tensor:
        return self._energy_scale() * self.time_decay.phi(time_days) * self.energy_net.energy(i1, i2)

    def _total_energy_masked(
        self,
        i1: tf.Tensor,
        i2: tf.Tensor,
        time_days: tf.Tensor,
        term_mask: tf.Tensor | None = None,
    ) -> tf.Tensor:
        features = self.energy_net.features(i1, i2)
        outer = tf.nn.softplus(self.energy_net.w2)
        if term_mask is not None:
            outer = outer * term_mask
        psi = tf.matmul(features, outer)
        return self._energy_scale() * self.time_decay.phi(time_days) * psi

    def predict_uniaxial(self, stretch: tf.Tensor, time_days: tf.Tensor) -> tf.Tensor:
        stretch = tf.convert_to_tensor(stretch, dtype=tf.float32)
        time_days = tf.convert_to_tensor(time_days, dtype=tf.float32)
        i1 = tf.square(stretch) + 2.0 / stretch
        i2 = 2.0 * stretch + 1.0 / tf.square(stretch)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([i1, i2])
            psi = self.total_energy(i1, i2, time_days)
        dpsi_di1 = tape.gradient(psi, i1)
        dpsi_di2 = tape.gradient(psi, i2)
        del tape
        minus = 2.0 * (dpsi_di1 / tf.square(stretch) + dpsi_di2 / tf.pow(stretch, 3.0))
        return 2.0 * (dpsi_di1 * stretch + dpsi_di2) - minus

    def predict_shear(self, gamma: tf.Tensor, time_days: tf.Tensor) -> tf.Tensor:
        gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        time_days = tf.convert_to_tensor(time_days, dtype=tf.float32)
        i1 = tf.square(gamma) + 3.0
        i2 = tf.square(gamma) + 3.0
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([i1, i2])
            psi = self.total_energy(i1, i2, time_days)
        dpsi_di1 = tape.gradient(psi, i1)
        dpsi_di2 = tape.gradient(psi, i2)
        del tape
        return 2.0 * gamma * (dpsi_di1 + dpsi_di2)

    def predict_uniaxial_contributions(self, stretch: tf.Tensor, time_days: tf.Tensor) -> np.ndarray:
        term_count = int(self.energy_net.w2.shape[0])
        outputs = []
        for i in range(term_count):
            mask = np.zeros((term_count, 1), dtype=np.float32)
            mask[i, 0] = 1.0
            outputs.append(self._predict_uniaxial_masked(stretch, time_days, mask).numpy().reshape(-1))
        return np.stack(outputs, axis=1)

    def predict_shear_contributions(self, gamma: tf.Tensor, time_days: tf.Tensor) -> np.ndarray:
        term_count = int(self.energy_net.w2.shape[0])
        outputs = []
        for i in range(term_count):
            mask = np.zeros((term_count, 1), dtype=np.float32)
            mask[i, 0] = 1.0
            outputs.append(self._predict_shear_masked(gamma, time_days, mask).numpy().reshape(-1))
        return np.stack(outputs, axis=1)

    def _predict_uniaxial_masked(self, stretch: tf.Tensor, time_days: tf.Tensor, term_mask: np.ndarray) -> tf.Tensor:
        stretch = tf.convert_to_tensor(stretch, dtype=tf.float32)
        time_days = tf.convert_to_tensor(time_days, dtype=tf.float32)
        mask = tf.convert_to_tensor(term_mask, dtype=tf.float32)
        i1 = tf.square(stretch) + 2.0 / stretch
        i2 = 2.0 * stretch + 1.0 / tf.square(stretch)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([i1, i2])
            psi = self._total_energy_masked(i1, i2, time_days, mask)
        dpsi_di1 = tape.gradient(psi, i1)
        dpsi_di2 = tape.gradient(psi, i2)
        del tape
        minus = 2.0 * (dpsi_di1 / tf.square(stretch) + dpsi_di2 / tf.pow(stretch, 3.0))
        return 2.0 * (dpsi_di1 * stretch + dpsi_di2) - minus

    def _predict_shear_masked(self, gamma: tf.Tensor, time_days: tf.Tensor, term_mask: np.ndarray) -> tf.Tensor:
        gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        time_days = tf.convert_to_tensor(time_days, dtype=tf.float32)
        mask = tf.convert_to_tensor(term_mask, dtype=tf.float32)
        i1 = tf.square(gamma) + 3.0
        i2 = tf.square(gamma) + 3.0
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([i1, i2])
            psi = self._total_energy_masked(i1, i2, time_days, mask)
        dpsi_di1 = tape.gradient(psi, i1)
        dpsi_di2 = tape.gradient(psi, i2)
        del tape
        return 2.0 * gamma * (dpsi_di1 + dpsi_di2)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return self.energy_net.trainable_variables + self.time_decay.trainable_variables + [self.energy_scale_raw]

    def export_weights(self) -> dict[str, list[float] | dict[str, float]]:
        return {
            "w1_i1_raw": self.energy_net.w1_i1.numpy().reshape(-1).tolist(),
            "w1_i2_raw": self.energy_net.w1_i2.numpy().reshape(-1).tolist(),
            "w2_raw": self.energy_net.w2.numpy().reshape(-1).tolist(),
            "time_decay": self.time_decay.params_dict(),
            "energy_scale": float(self._energy_scale().numpy()),
        }


def fit_cann(payload: dict, config: TrainingConfig, variant: str) -> tuple[TimeDependentCANN, pd.DataFrame]:
    train_blocks, test_blocks, energy_scale_init = _prepare_blocks(payload, variant)
    model = TimeDependentCANN(
        decay_model=config.decay_model,
        seed=config.random_seed,
        energy_scale_init=energy_scale_init,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    history: list[dict] = []
    best_loss = np.inf
    best_weights = [var.numpy().copy() for var in model.trainable_variables]
    wait = 0

    progress = tqdm(
        range(config.epochs),
        desc=f"train {variant}",
        leave=False,
        unit="epoch",
    )
    for epoch in progress:
        with tf.GradientTape() as tape:
            loss = _loss_over_blocks(model, train_blocks)
            if config.l1_penalty > 0.0:
                l1 = tf.add_n([tf.reduce_sum(tf.abs(var)) for var in model.trainable_variables])
                loss += config.l1_penalty * l1

        grads = tape.gradient(loss, model.trainable_variables)
        if any(grad is None for grad in grads):
            raise RuntimeError("Encountered disconnected gradients during training.")
        grads = [tf.clip_by_norm(grad, config.grad_clip_norm) for grad in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss = float(loss.numpy())
        val_loss = float(_loss_over_blocks(model, test_blocks).numpy())
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            progress.set_postfix_str("non-finite loss")
            break
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        progress.set_postfix(
            train=f"{train_loss:.4g}",
            val=f"{val_loss:.4g}",
            best=f"{best_loss:.4g}" if np.isfinite(best_loss) else "inf",
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = [var.numpy().copy() for var in model.trainable_variables]
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                progress.set_postfix_str(f"early stop @ {epoch + 1}")
                break
    progress.close()

    for var, best in zip(model.trainable_variables, best_weights):
        var.assign(best)

    return model, pd.DataFrame(history)


def evaluate_cann(model: TimeDependentCANN, payload: dict, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    train_files = set(payload["train_files"])
    test_files = set(payload["test_files"])

    full_uni = payload["full_uniaxial_table"].copy()
    full_shear = payload["full_shear_table"].copy()

    uni_pred = model.predict_uniaxial(
        full_uni["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1),
        full_uni["time_days"].to_numpy(dtype=np.float32).reshape(-1, 1),
    ).numpy().reshape(-1)
    shear_pred = model.predict_shear(
        full_shear["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1),
        full_shear["time_days"].to_numpy(dtype=np.float32).reshape(-1, 1),
    ).numpy().reshape(-1)

    full_uni["predicted_stress_pa"] = uni_pred
    full_shear["predicted_stress_pa"] = shear_pred
    full_uni["split"] = np.where(full_uni["file_name"].isin(list(test_files)), "test", "train")
    full_shear["split"] = np.where(full_shear["file_name"].isin(list(test_files)), "test", "train")
    full_uni["train_variant"] = variant
    full_shear["train_variant"] = variant
    prediction_rows.extend(full_uni.to_dict("records"))
    prediction_rows.extend(full_shear.to_dict("records"))
    predictions = pd.DataFrame(prediction_rows)

    for (loading, split, time_days), group in predictions.groupby(["loading", "split", "time_days"], sort=True):
        y_true = group["stress_pa"].to_numpy(dtype=float)
        y_pred = group["predicted_stress_pa"].to_numpy(dtype=float)
        metric_rows.append(
            {
                "train_variant": variant,
                "loading": loading,
                "split": split,
                "time_days": float(time_days),
                "mae_pa": mean_absolute_error(y_true, y_pred),
                "rmse_pa": np.sqrt(mean_squared_error(y_true, y_pred)),
                "r2": r2_score(y_true, y_pred),
                "n_points": int(len(group)),
            }
        )

    for split, group in predictions.groupby("split", sort=True):
        total_true = group["stress_pa"].to_numpy(dtype=float)
        total_pred = group["predicted_stress_pa"].to_numpy(dtype=float)
        metric_rows.append(
            {
                "train_variant": variant,
                "loading": "overall",
                "split": split,
                "time_days": -1.0,
                "mae_pa": mean_absolute_error(total_true, total_pred),
                "rmse_pa": np.sqrt(mean_squared_error(total_true, total_pred)),
                "r2": r2_score(total_true, total_pred),
                "n_points": int(len(group)),
            }
        )

    return predictions, pd.DataFrame(metric_rows)


def save_training_artifacts(
    model: TimeDependentCANN,
    history_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    config: TrainingConfig,
    run_dir: Path,
    split_info: dict[str, list[str]] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    predictions_df.to_csv(run_dir / "predictions.csv", index=False)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "model_weights.json").write_text(json.dumps(model.export_weights(), indent=2), encoding="utf-8")
    (run_dir / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    if split_info is not None:
        (run_dir / "data_split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    _plot_losses(history_df, run_dir / "loss_curve.png")
    _plot_predictions(predictions_df, run_dir / "predicted_curves.png")


def _plot_losses(history_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted loss")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_predictions(predictions_df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, mode in zip(axes, ["uniaxial", "shear"]):
        subset = predictions_df.loc[predictions_df["mode"] == mode]
        for time_days, group in subset.groupby("time_days", sort=True):
            order = group.sort_values("deformation")
            ax.plot(order["deformation"], order["stress_pa"], linestyle="--", label=f"exp t={time_days:g}d")
            ax.plot(order["deformation"], order["predicted_stress_pa"], label=f"pred t={time_days:g}d")
        ax.set_title(mode)
        ax.set_xlabel("stretch" if mode == "uniaxial" else "gamma")
        ax.set_ylabel("stress (Pa)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor, weight: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(weight * tf.square(y_true - y_pred))


def _tensor_block(x: np.ndarray, t: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, tf.Tensor]:
    return {
        "x": tf.convert_to_tensor(x, dtype=tf.float32),
        "t": tf.convert_to_tensor(t, dtype=tf.float32),
        "y": tf.convert_to_tensor(y, dtype=tf.float32),
        "w": tf.convert_to_tensor(w, dtype=tf.float32),
    }


def _frame_to_block(frame: pd.DataFrame, feature_name: str, scale: float, loss_weight: float) -> dict[str, tf.Tensor | str]:
    return {
        "kind": "shear" if feature_name == "deformation" and frame["mode"].iloc[0] == "shear" else "uniaxial",
        "x": tf.convert_to_tensor(frame[feature_name].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=tf.float32),
        "t": tf.convert_to_tensor(frame["time_days"].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=tf.float32),
        "y": tf.convert_to_tensor(frame["stress_pa"].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=tf.float32),
        "w": tf.ones((len(frame), 1), dtype=tf.float32),
        "scale": tf.constant(float(scale), dtype=tf.float32),
        "loss_weight": tf.constant(float(loss_weight), dtype=tf.float32),
    }


def _prepare_blocks(payload: dict, variant: str) -> tuple[list[dict], list[dict], float]:
    if variant not in VARIANT_LABELS:
        raise ValueError(f"Unsupported training variant: {variant}")

    full_uni = payload["full_uniaxial_table"].copy()
    full_shear = payload["full_shear_table"].copy()
    train_files = set(payload["train_files"])
    test_files = set(payload["test_files"])

    train_uni = full_uni.loc[full_uni["file_name"].isin(list(train_files))].copy()
    test_uni = full_uni.loc[full_uni["file_name"].isin(list(test_files))].copy()
    train_shear = full_shear.loc[full_shear["file_name"].isin(list(train_files))].copy()
    test_shear = full_shear.loc[full_shear["file_name"].isin(list(test_files))].copy()
    if test_uni.empty:
        test_uni = train_uni.copy()
    if test_shear.empty:
        test_shear = train_shear.copy()

    train_blocks: list[dict] = []
    test_blocks: list[dict] = []

    if variant == "compression_only":
        comp_train = train_uni.loc[train_uni["loading"] == "compression"].copy()
        comp_test = test_uni.loc[test_uni["loading"] == "compression"].copy()
        scale = max(float(comp_train["stress_pa"].abs().max()), 1.0)
        train_blocks.append(_frame_to_block(comp_train, "deformation", scale, 1.0))
        test_blocks.append(_frame_to_block(comp_test, "deformation", scale, 1.0))
        energy_scale_init = scale / 50.0
    elif variant == "unloading_only":
        unloading_train = train_uni.loc[train_uni["loading"] == "unloading"].copy()
        unloading_test = test_uni.loc[test_uni["loading"] == "unloading"].copy()
        scale = max(float(unloading_train["stress_pa"].abs().max()), 1.0)
        train_blocks.append(_frame_to_block(unloading_train, "deformation", scale, 1.0))
        test_blocks.append(_frame_to_block(unloading_test, "deformation", scale, 1.0))
        energy_scale_init = scale / 50.0
    elif variant == "shear_only":
        scale = max(float(train_shear["stress_pa"].abs().max()), 1.0)
        train_blocks.append(_frame_to_block(train_shear, "deformation", scale, 1.0))
        test_blocks.append(_frame_to_block(test_shear, "deformation", scale, 1.0))
        energy_scale_init = scale / 50.0
    else:
        comp_train = train_uni.loc[train_uni["loading"] == "compression"].copy()
        comp_test = test_uni.loc[test_uni["loading"] == "compression"].copy()
        unloading_train = train_uni.loc[train_uni["loading"] == "unloading"].copy()
        unloading_test = test_uni.loc[test_uni["loading"] == "unloading"].copy()
        shear_scale = max(float(train_shear["stress_pa"].abs().max()), 1.0)
        comp_scale = max(float(comp_train["stress_pa"].abs().max()), 1.0)
        unloading_scale = max(float(unloading_train["stress_pa"].abs().max()), 1.0)
        train_blocks.extend(
            [
                _frame_to_block(comp_train, "deformation", comp_scale, 1.0),
                _frame_to_block(unloading_train, "deformation", unloading_scale, 1.0),
                _frame_to_block(train_shear, "deformation", shear_scale, 1.0),
            ]
        )
        test_blocks.extend(
            [
                _frame_to_block(comp_test, "deformation", comp_scale, 1.0),
                _frame_to_block(unloading_test, "deformation", unloading_scale, 1.0),
                _frame_to_block(test_shear, "deformation", shear_scale, 1.0),
            ]
        )
        energy_scale_init = max(comp_scale, unloading_scale, shear_scale) / 50.0

    return train_blocks, test_blocks, energy_scale_init


def _loss_over_blocks(model: TimeDependentCANN, blocks: list[dict]) -> tf.Tensor:
    losses = []
    for block in blocks:
        if block["kind"] == "shear":
            pred = model.predict_shear(block["x"], block["t"])
        else:
            pred = model.predict_uniaxial(block["x"], block["t"])
        loss = _weighted_mse(block["y"] / block["scale"], pred / block["scale"], block["w"])
        losses.append(block["loss_weight"] * loss)
    return tf.add_n(losses)
