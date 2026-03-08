from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf


TRAIN_VARIANTS = {
    "compression_only": ["compression"],
    "unloading_only": ["unloading"],
    "shear_only": ["shear"],
    "weighted_all": ["compression", "unloading", "shear"],
}


@dataclass
class StretchTimeTrainingConfig:
    epochs: int = 4000
    patience: int = 500
    learning_rate: float = 3e-3
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 250
    min_learning_rate: float = 1e-4
    seed: int = 42
    grad_clip_norm: float = 5.0


class StretchTimeCANN(tf.Module):
    def __init__(
        self,
        powers: list[int],
        shear_poly_degrees: list[int] | None = None,
        seed: int = 42,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "stretch_time_cann")
        self.powers = [int(p) for p in powers if p != 0]
        self.shear_poly_degrees = [int(d) for d in (shear_poly_degrees or []) if d > 0]
        total_terms = len(self.powers) + len(self.shear_poly_degrees)
        init = tf.keras.initializers.GlorotNormal(seed=seed)
        zeros = tf.keras.initializers.Zeros()
        intercept = init(shape=(total_terms, 1), dtype=tf.float32)
        deltas = zeros(shape=(total_terms, 3), dtype=tf.float32)
        deltas = tf.tensor_scatter_nd_update(deltas, indices=[[0, 0]], updates=[0.0])
        self.coeff_bank = tf.Variable(
            tf.concat([intercept, intercept, intercept], axis=1) + deltas,
            trainable=True,
            name="coeff_bank",
        )

    def _time_one_hot(self, time_days: tf.Tensor) -> tf.Tensor:
        t = tf.convert_to_tensor(time_days, dtype=tf.float32)
        idx = tf.cast(tf.round(tf.clip_by_value(t, 0.0, 2.0)), tf.int32)
        idx = tf.reshape(idx, [-1])
        return tf.one_hot(idx, depth=3, dtype=tf.float32)

    def _basis_uniaxial(self, stretch: tf.Tensor) -> tf.Tensor:
        terms = []
        for p in self.powers:
            p_float = tf.constant(float(p), dtype=tf.float32)
            term = tf.pow(stretch, p_float) + 2.0 * tf.pow(stretch, -0.5 * p_float) - 3.0
            terms.append(term)
        for _ in self.shear_poly_degrees:
            terms.append(tf.zeros_like(stretch))
        return tf.concat(terms, axis=1)

    def _principal_stretches_from_shear(self, gamma: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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

    def _energy_from_basis(self, basis: tf.Tensor, time_days: tf.Tensor) -> tf.Tensor:
        time_one_hot = self._time_one_hot(time_days)
        coeffs_t = tf.matmul(time_one_hot, self.coeff_bank, transpose_b=True)
        return tf.reduce_sum(basis * coeffs_t, axis=1, keepdims=True)

    def predict_uniaxial(self, stretch: tf.Tensor, time_days: tf.Tensor) -> tf.Tensor:
        stretch = tf.convert_to_tensor(stretch, dtype=tf.float32)
        time_days = tf.convert_to_tensor(time_days, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(stretch)
            psi = self._energy_from_basis(self._basis_uniaxial(stretch), time_days)
        return tape.gradient(psi, stretch)

    def predict_shear(self, gamma: tf.Tensor, time_days: tf.Tensor) -> tf.Tensor:
        gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        time_days = tf.convert_to_tensor(time_days, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(gamma)
            psi = self._energy_from_basis(self._basis_shear(gamma), time_days)
        return tape.gradient(psi, gamma)

    def predict_uniaxial_contributions(self, stretch: np.ndarray, time_days: np.ndarray) -> np.ndarray:
        return self._predict_contributions(stretch, time_days, loading="uniaxial")

    def predict_shear_contributions(self, gamma: np.ndarray, time_days: np.ndarray) -> np.ndarray:
        return self._predict_contributions(gamma, time_days, loading="shear")

    def _predict_contributions(self, deformation: np.ndarray, time_days: np.ndarray, loading: str) -> np.ndarray:
        x = tf.convert_to_tensor(deformation.reshape(-1, 1), dtype=tf.float32)
        t = tf.convert_to_tensor(time_days.reshape(-1, 1), dtype=tf.float32)
        outputs = []
        n_terms = len(self.powers) + len(self.shear_poly_degrees)
        for idx in range(n_terms):
            with tf.GradientTape() as tape:
                tape.watch(x)
                basis = self._basis_uniaxial(x) if loading == "uniaxial" else self._basis_shear(x)
                time_one_hot = self._time_one_hot(t)
                coeff_t = tf.matmul(time_one_hot, self.coeff_bank[idx : idx + 1], transpose_b=True)
                term_energy = basis[:, idx : idx + 1] * coeff_t
            grad = tape.gradient(term_energy, x)
            outputs.append(tf.zeros_like(x).numpy().reshape(-1) if grad is None else grad.numpy().reshape(-1))
        return np.stack(outputs, axis=1)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return [self.coeff_bank]

    def export_weights(self) -> dict[str, object]:
        return {
            "powers": self.powers,
            "shear_poly_degrees": self.shear_poly_degrees,
            "coeff_bank": self.coeff_bank.numpy().tolist(),
        }


def variant_architecture(variant: str) -> tuple[list[int], list[int]]:
    if variant == "shear_only":
        return list(range(-10, 11)), [1, 2, 3, 4, 5, 6]
    if variant == "weighted_all":
        return list(range(-10, 12, 2)), [1, 2, 3, 4, 5, 6]
    return list(range(-10, 12, 2)), []


def build_training_blocks(frame: pd.DataFrame) -> dict[str, dict[str, tf.Tensor]]:
    blocks: dict[str, dict[str, tf.Tensor]] = {}
    for loading, group in frame.groupby("loading", sort=False):
        ordered = group.sort_values(["time_days", "deformation"])
        x = ordered["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1)
        t = ordered["time_days"].to_numpy(dtype=np.float32).reshape(-1, 1)
        y = (ordered["stress_pa"].to_numpy(dtype=np.float32) / 1000.0).reshape(-1, 1)
        weights = build_loss_weights(x, loading)
        blocks[loading] = {
            "x": tf.convert_to_tensor(x, dtype=tf.float32),
            "t": tf.convert_to_tensor(t, dtype=tf.float32),
            "y": tf.convert_to_tensor(y, dtype=tf.float32),
            "w": tf.convert_to_tensor(weights, dtype=tf.float32),
        }
    return blocks


def build_loss_weights(x: np.ndarray, loading: str) -> np.ndarray:
    if loading != "shear":
        return np.ones_like(x, dtype=np.float32)
    gamma = np.maximum(x.astype(np.float32), 0.0)
    return 1.0 + 3.0 * np.exp(-gamma / 0.04)


def predict_block(model: StretchTimeCANN, block: dict[str, tf.Tensor], loading: str) -> tf.Tensor:
    if loading == "shear":
        return model.predict_shear(block["x"], block["t"])
    return model.predict_uniaxial(block["x"], block["t"])


def fit_stretch_time_cann(
    frame: pd.DataFrame,
    variant: str,
    config: StretchTimeTrainingConfig,
) -> tuple[StretchTimeCANN, pd.DataFrame]:
    powers, shear_poly_degrees = variant_architecture(variant)
    model = StretchTimeCANN(
        powers=powers,
        shear_poly_degrees=shear_poly_degrees,
        seed=config.seed,
        name=f"stretch_time_{variant}",
    )
    blocks = build_training_blocks(frame)
    target_loadings = [loading for loading in TRAIN_VARIANTS[variant] if loading in blocks]
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    best_loss = np.inf
    best_weights = [var.numpy().copy() for var in model.trainable_variables]
    wait = 0
    lr_wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        with tf.GradientTape() as tape:
            losses = {}
            for loading in target_loadings:
                block = blocks[loading]
                pred = predict_block(model, block, loading)
                residual_sq = tf.square(block["y"] - pred)
                losses[loading] = tf.reduce_mean(block["w"] * residual_sq)
            loss = tf.add_n(list(losses.values())) / float(len(losses))

        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(grad, config.grad_clip_norm) for grad in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss = float(loss.numpy())
        record = {"epoch": epoch + 1, "loss": epoch_loss, "lr": float(optimizer.learning_rate.numpy())}
        for loading in ["compression", "unloading", "shear"]:
            record[f"{loading}_loss"] = float(losses[loading].numpy()) if loading in losses else np.nan
        history.append(record)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = [var.numpy().copy() for var in model.trainable_variables]
            wait = 0
            lr_wait = 0
        else:
            wait += 1
            lr_wait += 1
            if lr_wait >= config.lr_reduce_patience:
                current_lr = float(optimizer.learning_rate.numpy())
                new_lr = max(current_lr * config.lr_reduce_factor, config.min_learning_rate)
                if new_lr < current_lr:
                    optimizer.learning_rate.assign(new_lr)
                lr_wait = 0
            if wait >= config.patience:
                break

    for var, best in zip(model.trainable_variables, best_weights):
        var.assign(best)
    return model, pd.DataFrame(history)
