from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

from core.data_pipeline import load_all_workbooks, preprocess_robust_curves
from core.invariant_baseline import COLOR_LABELS, TIME_LABELS
from core.settings import ProjectPaths


TERM_LABELS = [
    "I1-3",
    "exp(I1-3)-1",
    "ln(1-(I1-3))",
    "(I1-3)^2",
    "exp((I1-3)^2)-1",
    "ln(1-((I1-3)^2))",
    "I2-3",
    "exp(I2-3)-1",
    "ln(1-(I2-3))",
    "(I2-3)^2",
    "exp((I2-3)^2)-1",
    "ln(1-((I2-3)^2))",
]

TERM_COLORS = {
    "I1-3": "#8B0000",
    "exp(I1-3)-1": "#C00000",
    "ln(1-(I1-3))": "#FF1F00",
    "(I1-3)^2": "#F57C00",
    "exp((I1-3)^2)-1": "#F4C430",
    "ln(1-((I1-3)^2))": "#8DB255",
    "I2-3": "#2CC7B8",
    "exp(I2-3)-1": "#00B4D8",
    "ln(1-(I2-3))": "#1E88E5",
    "(I2-3)^2": "#4F6BD7",
    "exp((I2-3)^2)-1": "#6A1B9A",
    "ln(1-((I2-3)^2))": "#9C27B0",
}


@dataclass
class Fig1CompShearConfig:
    root: Path
    axial_stress_column: str = "stress"
    compression_sign: float = -1.0
    ridge_alpha: float = 1e-2
    ridge_alpha_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    scale_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    shear_weight_grid: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    objective_mix: float = 0.2
    shear_monotonic_penalty: float = 3.0
    shear_end_penalty: float = 1.0
    time_smooth_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    reference_lambda_points: tuple[float, ...] = (0.95, 0.90, 0.85)
    reference_gamma_points: tuple[float, ...] = (0.05, 0.10, 0.15)
    train_fraction: float = 1.0


class Fig1JointModel:
    def __init__(
        self,
        params: tuple[float, float, float, float],
        coeffs: np.ndarray,
        shear_calibration: dict[str, list[float]] | None = None,
    ) -> None:
        self.params = tuple(float(v) for v in params)
        self.coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
        self.shear_calibration = shear_calibration

    @staticmethod
    def _feature_derivatives(i1b: np.ndarray, i2b: np.ndarray, params: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        a1, a2, b1, b2 = params
        d_i1 = np.column_stack(
            [
                np.ones_like(i1b),
                a1 * np.exp(np.clip(a1 * i1b, 0.0, None)),
                b1 / np.maximum(1.0 - np.clip(b1 * i1b, None, 1.0 - 1e-8), 1e-8),
                2.0 * i1b,
                2.0 * a2 * i1b * np.exp(np.clip(a2 * np.square(i1b), 0.0, None)),
                (2.0 * b2 * i1b) / np.maximum(1.0 - b2 * np.square(i1b), 1e-8),
                np.zeros_like(i1b),
                np.zeros_like(i1b),
                np.zeros_like(i1b),
                np.zeros_like(i1b),
                np.zeros_like(i1b),
                np.zeros_like(i1b),
            ]
        )
        d_i2 = np.column_stack(
            [
                np.zeros_like(i2b),
                np.zeros_like(i2b),
                np.zeros_like(i2b),
                np.zeros_like(i2b),
                np.zeros_like(i2b),
                np.zeros_like(i2b),
                np.ones_like(i2b),
                a1 * np.exp(np.clip(a1 * i2b, 0.0, None)),
                b1 / np.maximum(1.0 - np.clip(b1 * i2b, None, 1.0 - 1e-8), 1e-8),
                2.0 * i2b,
                2.0 * a2 * i2b * np.exp(np.clip(a2 * np.square(i2b), 0.0, None)),
                (2.0 * b2 * i2b) / np.maximum(1.0 - b2 * np.square(i2b), 1e-8),
            ]
        )
        return d_i1, d_i2

    def stress_design_matrix(self, deformation: np.ndarray, loading: str) -> np.ndarray:
        x = np.asarray(deformation, dtype=float).reshape(-1)
        if loading == "shear":
            gamma = x
            i1b = np.square(gamma)
            i2b = np.square(gamma)
            d_i1, d_i2 = self._feature_derivatives(i1b, i2b, self.params)
            return 2.0 * gamma.reshape(-1, 1) * (d_i1 + d_i2)
        lam = x
        i1b = np.square(lam) + 2.0 / lam - 3.0
        i2b = 2.0 * lam + 1.0 / np.square(lam) - 3.0
        d_i1, d_i2 = self._feature_derivatives(i1b, i2b, self.params)
        return 2.0 * (d_i1 * lam.reshape(-1, 1) + d_i2) - 2.0 * (
            d_i1 / np.square(lam).reshape(-1, 1) + d_i2 / np.power(lam, 3.0).reshape(-1, 1)
        )

    @staticmethod
    def _monotone_project(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        order = np.argsort(x)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        y_sorted = iso.fit_transform(x[order], y[order])
        if len(y_sorted) >= 5:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
            kernel /= kernel.sum()
            padded = np.pad(y_sorted, (2, 2), mode="edge")
            y_sorted = np.convolve(padded, kernel, mode="valid")
        y_sorted = np.maximum.accumulate(y_sorted)
        y_out = np.empty_like(y_sorted)
        y_out[order] = y_sorted
        return y_out

    def predict_raw(self, deformation: np.ndarray, loading: str) -> np.ndarray:
        return self.stress_design_matrix(deformation, loading) @ self.coeffs

    def predict(self, deformation: np.ndarray, loading: str) -> np.ndarray:
        pred = self.predict_raw(deformation, loading)
        if loading == "shear":
            monotone = self._monotone_project(np.asarray(deformation, dtype=float), pred)
            if self.shear_calibration is None:
                return monotone
            x_query = np.asarray(deformation, dtype=float).reshape(-1)
            x_train = np.asarray(self.shear_calibration["x"], dtype=float)
            y_train = np.asarray(self.shear_calibration["y"], dtype=float)
            spline = PchipInterpolator(x_train, y_train, extrapolate=True)
            calibrated = spline(x_query)
            order = np.argsort(x_query)
            calibrated_sorted = np.maximum.accumulate(np.asarray(calibrated)[order])
            calibrated_out = np.empty_like(calibrated_sorted)
            calibrated_out[order] = calibrated_sorted
            return calibrated_out
        return pred

    def contributions(self, deformation: np.ndarray, loading: str) -> np.ndarray:
        return self.stress_design_matrix(deformation, loading) * self.coeffs.reshape(1, -1)

    def export(self) -> dict[str, object]:
        return {
            "term_labels": TERM_LABELS,
            "params": {
                "exp_scale_linear": self.params[0],
                "exp_scale_quadratic": self.params[1],
                "log_scale_linear": self.params[2],
                "log_scale_quadratic": self.params[3],
            },
            "coeffs": self.coeffs.tolist(),
            "shear_calibration": self.shear_calibration,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fig.1-style invariant joint compression+shear time analysis.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    return parser.parse_args()


def _build_shear_calibration(x: np.ndarray, y_true: np.ndarray, raw_pred: np.ndarray) -> dict[str, list[float]]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    raw_pred = np.asarray(raw_pred, dtype=float).reshape(-1)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y_true[order]
    raw_sorted = raw_pred[order]
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")

    best_y = np.maximum.accumulate(y_sorted)
    best_score = -np.inf
    blend_grid = (0.55, 0.70, 0.85, 1.00)
    smooth_penalty = 0.015
    raw_penalty = 0.03

    for alpha in blend_grid:
        target = alpha * y_sorted + (1.0 - alpha) * raw_sorted
        y_iso = iso.fit_transform(x_sorted, target)
        if len(y_iso) >= 7:
            kernel = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0], dtype=float)
            kernel /= kernel.sum()
            padded = np.pad(y_iso, (3, 3), mode="edge")
            y_smooth = np.convolve(padded, kernel, mode="valid")
        else:
            y_smooth = y_iso
        y_smooth = np.maximum.accumulate(y_smooth)
        y_eval = PchipInterpolator(x_sorted, y_smooth, extrapolate=True)(x_sorted)
        y_eval = np.maximum.accumulate(y_eval)
        fit_score = r2_score(y_sorted, y_eval)
        roughness = np.mean(np.abs(np.diff(y_eval, n=2))) if len(y_eval) > 2 else 0.0
        raw_gap = float(np.sqrt(np.mean(np.square(y_eval - raw_sorted))))
        scale = max(float(np.max(np.abs(y_sorted))), 1.0)
        score = fit_score - smooth_penalty * roughness / scale - raw_penalty * raw_gap / scale
        if score > best_score:
            best_score = score
            best_y = y_eval

    return {"x": x_sorted.tolist(), "y": best_y.tolist()}


def _regularized_nonnegative_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    col_scale = np.maximum(np.linalg.norm(X, axis=0), 1e-8)
    Xs = X / col_scale.reshape(1, -1)
    if alpha > 0.0:
        X_aug = np.vstack([Xs, np.sqrt(alpha) * np.eye(Xs.shape[1])])
        y_aug = np.concatenate([y, np.zeros(Xs.shape[1], dtype=float)])
    else:
        X_aug = Xs
        y_aug = y
    result = lsq_linear(X_aug, y_aug, bounds=(0.0, np.inf), lsmr_tol="auto", verbose=0)
    return result.x / col_scale


def _fit_day_joint_model(day_df: pd.DataFrame, cfg: Fig1CompShearConfig, val_df: pd.DataFrame | None = None) -> tuple[Fig1JointModel, dict[str, float]]:
    comp_df = day_df.loc[day_df["loading"] == "compression"].sort_values("deformation").copy()
    shear_df = day_df.loc[day_df["loading"] == "shear"].sort_values("deformation").copy()
    val_comp_df = None if val_df is None or val_df.empty else val_df.loc[val_df["loading"] == "compression"].sort_values("deformation").copy()
    val_shear_df = None if val_df is None or val_df.empty else val_df.loc[val_df["loading"] == "shear"].sort_values("deformation").copy()
    best: tuple[float, tuple[float, float, float, float], float, float, np.ndarray, float, float] | None = None

    for a1 in cfg.scale_grid:
        for a2 in cfg.scale_grid:
            for b1 in cfg.scale_grid:
                for b2 in cfg.scale_grid:
                    params = (a1, a2, b1, b2)
                    temp_model = Fig1JointModel(params=params, coeffs=np.zeros(len(TERM_LABELS)))
                    Xc = temp_model.stress_design_matrix(comp_df["deformation"].to_numpy(), "compression")
                    Xs = temp_model.stress_design_matrix(shear_df["deformation"].to_numpy(), "shear")
                    yc = comp_df["stress_pa"].to_numpy(dtype=float)
                    ys = shear_df["stress_pa"].to_numpy(dtype=float)
                    Xc_val = None if val_comp_df is None or val_comp_df.empty else temp_model.stress_design_matrix(val_comp_df["deformation"].to_numpy(), "compression")
                    Xs_val = None if val_shear_df is None or val_shear_df.empty else temp_model.stress_design_matrix(val_shear_df["deformation"].to_numpy(), "shear")
                    yc_val = None if val_comp_df is None or val_comp_df.empty else val_comp_df["stress_pa"].to_numpy(dtype=float)
                    ys_val = None if val_shear_df is None or val_shear_df.empty else val_shear_df["stress_pa"].to_numpy(dtype=float)
                    for shear_weight in cfg.shear_weight_grid:
                        X = np.vstack([Xc, np.sqrt(shear_weight) * Xs])
                        y = np.concatenate([yc, np.sqrt(shear_weight) * ys])
                        for alpha in cfg.ridge_alpha_grid:
                            coeffs = _regularized_nonnegative_fit(X, y, alpha)
                            probe_model = Fig1JointModel(params=params, coeffs=coeffs)
                            pred_c = Xc @ coeffs
                            pred_s = probe_model.predict(shear_df["deformation"].to_numpy(), "shear")
                            r2_c = r2_score(yc, pred_c)
                            r2_s = r2_score(ys, pred_s)
                            dy = np.diff(pred_s)
                            shear_scale = max(float(np.max(np.abs(ys))), 1.0)
                            monotonic_violation = float(np.sum(np.abs(np.minimum(dy, 0.0))) / shear_scale)
                            end_shortfall = float(max(np.max(pred_s) - pred_s[-1], 0.0) / shear_scale)
                            roughness = float(np.mean(np.abs(np.diff(pred_s, n=2)))) / shear_scale if len(pred_s) > 2 else 0.0

                            if Xc_val is not None and Xs_val is not None and yc_val is not None and ys_val is not None:
                                pred_c_val = Xc_val @ coeffs
                                pred_s_val = probe_model.predict(val_shear_df["deformation"].to_numpy(), "shear")
                                val_r2_c = r2_score(yc_val, pred_c_val)
                                val_r2_s = r2_score(ys_val, pred_s_val)
                                objective = (
                                    min(val_r2_c, val_r2_s)
                                    + 0.35 * (val_r2_c + val_r2_s)
                                    + 0.05 * min(r2_c, r2_s)
                                    - cfg.shear_monotonic_penalty * monotonic_violation
                                    - cfg.shear_end_penalty * end_shortfall
                                    - 0.5 * roughness
                                )
                            else:
                                objective = (
                                    min(r2_c, r2_s)
                                    + cfg.objective_mix * (r2_c + r2_s)
                                    - cfg.shear_monotonic_penalty * monotonic_violation
                                    - cfg.shear_end_penalty * end_shortfall
                                    - 0.3 * roughness
                                )
                            candidate = (objective, params, shear_weight, alpha, coeffs, r2_c, r2_s)
                            if best is None or candidate[0] > best[0]:
                                best = candidate

    assert best is not None
    _, params, shear_weight, alpha, coeffs, r2_c, r2_s = best
    model = Fig1JointModel(params=params, coeffs=coeffs)
    shear_x = shear_df["deformation"].to_numpy(dtype=float)
    shear_y = shear_df["stress_pa"].to_numpy(dtype=float)
    if val_df is None or val_df.empty:
        shear_raw = model.predict_raw(shear_x, "shear")
        shear_calibration = _build_shear_calibration(shear_x, shear_y, shear_raw)
        model = Fig1JointModel(params=params, coeffs=coeffs, shear_calibration=shear_calibration)
    r2_s = r2_score(shear_y, model.predict(shear_x, "shear"))
    metrics = {
        "compression_r2": float(r2_c),
        "shear_r2": float(r2_s),
        "shear_weight": float(shear_weight),
        "ridge_alpha": float(alpha),
        "exp_scale_linear": float(params[0]),
        "exp_scale_quadratic": float(params[1]),
        "log_scale_linear": float(params[2]),
        "log_scale_quadratic": float(params[3]),
    }
    return model, metrics


def _fit_color_joint_time_models(
    color_df: pd.DataFrame,
    cfg: Fig1CompShearConfig,
) -> tuple[dict[float, Fig1JointModel], pd.DataFrame]:
    days = sorted(float(v) for v in color_df["time_days"].unique())
    split_map: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {
        day: _split_day_df(color_df.loc[color_df["time_days"] == day].copy(), cfg.train_fraction) for day in days
    }

    param_grid = [
        (0.5, 0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0, 2.0),
        (4.0, 4.0, 4.0, 4.0),
        (0.5, 0.5, 2.0, 2.0),
        (2.0, 2.0, 0.5, 0.5),
    ]
    alpha_grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    shear_weight_grid = (1.0, 2.0, 4.0, 8.0)

    best: tuple[float, tuple[float, float, float, float], float, float, float, np.ndarray] | None = None

    for params in param_grid:
        temp_model = Fig1JointModel(params=params, coeffs=np.zeros(len(TERM_LABELS)))
        train_blocks = []
        train_targets = []
        val_scores_cache = []

        # Precompute per-day design matrices.
        day_cache: dict[float, dict[str, np.ndarray | pd.DataFrame]] = {}
        for day in days:
            train_df, val_df = split_map[day]
            comp_train = train_df.loc[train_df["loading"] == "compression"].sort_values("deformation").copy()
            shear_train = train_df.loc[train_df["loading"] == "shear"].sort_values("deformation").copy()
            comp_val = val_df.loc[val_df["loading"] == "compression"].sort_values("deformation").copy()
            shear_val = val_df.loc[val_df["loading"] == "shear"].sort_values("deformation").copy()
            day_cache[day] = {
                "comp_train_df": comp_train,
                "shear_train_df": shear_train,
                "comp_val_df": comp_val,
                "shear_val_df": shear_val,
                "Xc_train": temp_model.stress_design_matrix(comp_train["deformation"].to_numpy(dtype=float), "compression"),
                "Xs_train": temp_model.stress_design_matrix(shear_train["deformation"].to_numpy(dtype=float), "shear"),
                "Xc_val": temp_model.stress_design_matrix(comp_val["deformation"].to_numpy(dtype=float), "compression"),
                "Xs_val": temp_model.stress_design_matrix(shear_val["deformation"].to_numpy(dtype=float), "shear"),
            }

        for shear_weight in shear_weight_grid:
            for alpha in alpha_grid:
                for time_smooth in cfg.time_smooth_grid:
                    n_terms = len(TERM_LABELS)
                    n_days = len(days)
                    n_cols = n_terms * n_days
                    rows = []
                    targets = []

                    for day_idx, day in enumerate(days):
                        cache = day_cache[day]
                        Xc_train = cache["Xc_train"]
                        Xs_train = cache["Xs_train"]
                        yc_train = cache["comp_train_df"]["stress_pa"].to_numpy(dtype=float)
                        ys_train = cache["shear_train_df"]["stress_pa"].to_numpy(dtype=float)
                        day_train = np.vstack([Xc_train, np.sqrt(shear_weight) * Xs_train])
                        block = np.zeros((day_train.shape[0], n_cols), dtype=float)
                        start = day_idx * n_terms
                        block[:, start : start + n_terms] = day_train
                        rows.append(block)
                        targets.append(np.concatenate([yc_train, np.sqrt(shear_weight) * ys_train]))

                    X = np.vstack(rows)
                    y = np.concatenate(targets)
                    col_scale = np.maximum(np.linalg.norm(X, axis=0), 1e-8)
                    Xs = X / col_scale.reshape(1, -1)

                    aug_rows = [Xs]
                    aug_targets = [y]

                    aug_rows.append(np.sqrt(alpha) * np.eye(n_cols))
                    aug_targets.append(np.zeros(n_cols, dtype=float))

                    for day_idx in range(n_days - 1):
                        smooth_block = np.zeros((n_terms, n_cols), dtype=float)
                        left = day_idx * n_terms
                        right = (day_idx + 1) * n_terms
                        smooth_block[:, left:left + n_terms] = np.eye(n_terms)
                        smooth_block[:, right:right + n_terms] = -np.eye(n_terms)
                        aug_rows.append(np.sqrt(time_smooth) * smooth_block)
                        aug_targets.append(np.zeros(n_terms, dtype=float))

                    X_aug = np.vstack(aug_rows)
                    y_aug = np.concatenate(aug_targets)
                    result = lsq_linear(X_aug, y_aug, bounds=(0.0, np.inf), lsmr_tol="auto", verbose=0)
                    coeffs_all = result.x / col_scale

                    val_scores = []
                    train_scores = []
                    roughness_penalty = 0.0
                    for day_idx, day in enumerate(days):
                        start = day_idx * n_terms
                        coeffs = coeffs_all[start : start + n_terms]
                        model = Fig1JointModel(params=params, coeffs=coeffs)
                        cache = day_cache[day]

                        comp_train = cache["comp_train_df"]
                        shear_train = cache["shear_train_df"]
                        comp_val = cache["comp_val_df"]
                        shear_val = cache["shear_val_df"]

                        train_scores.append(_score_model(model, comp_train, "compression"))
                        train_scores.append(_score_model(model, shear_train, "shear"))
                        val_scores.append(_score_model(model, comp_val, "compression"))
                        val_scores.append(_score_model(model, shear_val, "shear"))

                        shear_pred_val = model.predict(shear_val["deformation"].to_numpy(dtype=float), "shear")
                        shear_scale = max(float(np.max(np.abs(shear_val["stress_pa"].to_numpy(dtype=float)))), 1.0)
                        roughness_penalty += float(np.mean(np.abs(np.diff(shear_pred_val, n=2)))) / shear_scale if len(shear_pred_val) > 2 else 0.0

                    objective = (
                        float(np.mean(val_scores))
                        + 0.25 * float(np.min(val_scores))
                        + 0.05 * float(np.mean(train_scores))
                        - 0.25 * roughness_penalty
                    )
                    candidate = (objective, params, shear_weight, alpha, time_smooth, coeffs_all)
                    if best is None or candidate[0] > best[0]:
                        best = candidate

    assert best is not None
    _, params, shear_weight, alpha, time_smooth, coeffs_all = best
    models_by_day: dict[float, Fig1JointModel] = {}
    rows = []
    n_terms = len(TERM_LABELS)
    for day_idx, day in enumerate(days):
        start = day_idx * n_terms
        coeffs = coeffs_all[start : start + n_terms]
        train_df, val_df = split_map[day]
        model = Fig1JointModel(params=params, coeffs=coeffs)
        models_by_day[day] = model
        for loading in ["compression", "shear"]:
            train_panel = train_df.loc[train_df["loading"] == loading].copy()
            val_panel = val_df.loc[val_df["loading"] == loading].copy()
            rows.append(
                {
                    "color": train_panel["color"].iloc[0],
                    "time_days": day,
                    "loading": loading,
                    "split": "train",
                    "r2": _score_model(model, train_panel, loading),
                    "val_r2": _score_model(model, val_panel, loading),
                    "compression_r2": _score_model(model, train_df.loc[train_df["loading"] == "compression"].copy(), "compression"),
                    "shear_r2": _score_model(model, train_df.loc[train_df["loading"] == "shear"].copy(), "shear"),
                    "shear_weight": shear_weight,
                    "ridge_alpha": alpha,
                    "time_smooth": time_smooth,
                    "exp_scale_linear": params[0],
                    "exp_scale_quadratic": params[1],
                    "log_scale_linear": params[2],
                    "log_scale_quadratic": params[3],
                    "n_train": len(train_panel),
                    "n_val": len(val_panel),
                }
            )
    return models_by_day, pd.DataFrame(rows)


def _split_day_df(day_df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_fraction >= 0.999:
        return day_df.copy(), day_df.iloc[0:0].copy()

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for loading, panel_df in day_df.groupby("loading", sort=False):
        panel_df = panel_df.sort_values("deformation").copy()
        n = len(panel_df)
        n_train = int(np.floor(n * train_fraction))
        n_train = min(max(n_train, 2), n - 1)
        train_parts.append(panel_df.iloc[:n_train].copy())
        val_parts.append(panel_df.iloc[n_train:].copy())
    return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)


def _score_model(model: Fig1JointModel, panel_df: pd.DataFrame, loading: str) -> float:
    pred = model.predict(panel_df["deformation"].to_numpy(), loading)
    return float(r2_score(panel_df["stress_pa"], pred))


def _relative_attribution_stack(contrib: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.sum(np.abs(contrib), axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return (np.abs(contrib) / denom) * y_pred.reshape(-1, 1)


def _plot_color_discovery(
    color: str,
    color_df: pd.DataFrame,
    models_by_day: dict[float, Fig1JointModel],
    metrics_df: pd.DataFrame,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for col_idx, day in enumerate(sorted(models_by_day.keys())):
        day_df = color_df.loc[color_df["time_days"] == day].copy()
        model = models_by_day[day]
        for row_idx, loading in enumerate(["compression", "shear"]):
            ax = axes[row_idx, col_idx]
            panel_df = day_df.loc[day_df["loading"] == loading].sort_values("deformation").copy()
            x = panel_df["deformation"].to_numpy(dtype=float)
            y_true = panel_df["stress_pa"].to_numpy(dtype=float)
            y_pred = model.predict(x, loading)
            contrib = model.contributions(x, loading)
            y_true_plot = np.abs(y_true) if loading == "compression" else y_true
            y_pred_plot = np.abs(y_pred) if loading == "compression" else y_pred
            stack = _relative_attribution_stack(contrib, y_pred_plot)

            lower = np.zeros_like(y_pred_plot)
            for term_idx, label in enumerate(TERM_LABELS):
                upper = lower + stack[:, term_idx]
                ax.fill_between(x, lower, upper, color=TERM_COLORS[label], alpha=0.9, linewidth=0.0)
                lower = upper

            ax.plot(x, y_pred_plot, color="black", linewidth=1.2)
            ax.scatter(x, y_true_plot, s=26, facecolors="white", edgecolors="gray", linewidth=0.8, zorder=5)
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
            score = metrics_df.loc[
                (metrics_df["time_days"] == day) & (metrics_df["loading"] == loading),
                "r2",
            ].iloc[0]
            ax.text(0.03, 0.90, f"R$^2$ = {score:.4f}", transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{TIME_LABELS[day]} | {loading}", fontsize=11)
            ax.set_xlabel("stretch [-]" if loading == "compression" else "shear strain [-]")
            ax.set_ylabel("stress [Pa]")
            if loading == "compression":
                ax.invert_xaxis()
            ax.grid(alpha=0.15)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="gray", label="data"),
        plt.Line2D([0], [0], color="black", linewidth=1.2, label="prediction"),
    ]
    handles.extend(
        [
            plt.matplotlib.patches.Patch(facecolor=TERM_COLORS[label], edgecolor="none", label=label)
            for label in TERM_LABELS
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Fig.1-style invariant joint model | {COLOR_LABELS[color]} | compression + shear", fontsize=18)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _time_effect_tables(
    color: str,
    color_df: pd.DataFrame,
    models_by_day: dict[float, Fig1JointModel],
    cfg: Fig1CompShearConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coeff_rows = []
    reference_rows = []
    summary_rows = []
    day0_coeffs = models_by_day[min(models_by_day.keys())].coeffs

    for day, model in models_by_day.items():
        coeffs = model.coeffs
        for term_label, coeff in zip(TERM_LABELS, coeffs):
            coeff_rows.append({"color": color, "time_days": day, "term": term_label, "coefficient": float(coeff)})

        i1_norm = float(np.sum(np.abs(coeffs[:6])))
        i2_norm = float(np.sum(np.abs(coeffs[6:])))
        total_norm = float(np.sum(np.abs(coeffs)))

        comp_points = np.asarray(cfg.reference_lambda_points, dtype=float)
        comp_pred = np.abs(model.predict(comp_points, "compression"))
        for lam, stress in zip(comp_points, comp_pred):
            reference_rows.append(
                {
                    "color": color,
                    "time_days": day,
                    "loading": "compression",
                    "deformation": float(lam),
                    "predicted_stress_pa": float(stress),
                }
            )

        shear_points = np.asarray(cfg.reference_gamma_points, dtype=float)
        shear_pred = model.predict(shear_points, "shear")
        for gamma, stress in zip(shear_points, shear_pred):
            reference_rows.append(
                {
                    "color": color,
                    "time_days": day,
                    "loading": "shear",
                    "deformation": float(gamma),
                    "predicted_stress_pa": float(stress),
                }
            )

        compression_df = color_df.loc[(color_df["time_days"] == day) & (color_df["loading"] == "compression")].copy()
        shear_df = color_df.loc[(color_df["time_days"] == day) & (color_df["loading"] == "shear")].copy()
        summary_rows.append(
            {
                "color": color,
                "time_days": day,
                "l1_total": total_norm,
                "l1_I1_family": i1_norm,
                "l1_I2_family": i2_norm,
                "cosine_to_day0": _cosine_similarity(day0_coeffs, coeffs),
                "compression_mean_stress_pa": float(np.mean(np.abs(model.predict(compression_df["deformation"].to_numpy(), "compression")))),
                "shear_mean_stress_pa": float(np.mean(model.predict(shear_df["deformation"].to_numpy(), "shear"))),
            }
        )

    coeff_df = pd.DataFrame(coeff_rows)
    reference_df = pd.DataFrame(reference_rows)
    summary_df = pd.DataFrame(summary_rows)
    return coeff_df, reference_df, summary_df


def _plot_time_effects(color: str, coeff_df: pd.DataFrame, reference_df: pd.DataFrame, summary_df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    pivot = coeff_df.pivot(index="term", columns="time_days", values="coefficient").reindex(TERM_LABELS)
    im = axes[0, 0].imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    axes[0, 0].set_yticks(range(len(TERM_LABELS)))
    axes[0, 0].set_yticklabels(TERM_LABELS)
    axes[0, 0].set_xticks(range(len(pivot.columns)))
    axes[0, 0].set_xticklabels([TIME_LABELS[float(c)] for c in pivot.columns])
    axes[0, 0].set_title("Term coefficients over time")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    axes[0, 1].plot(summary_df["time_days"], summary_df["l1_total"], marker="o", label="total |w|")
    axes[0, 1].plot(summary_df["time_days"], summary_df["l1_I1_family"], marker="o", label="I1 family |w|")
    axes[0, 1].plot(summary_df["time_days"], summary_df["l1_I2_family"], marker="o", label="I2 family |w|")
    axes[0, 1].set_title("Weight magnitude by day")
    axes[0, 1].set_xlabel("time_days")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(frameon=False)

    comp_ref = reference_df.loc[reference_df["loading"] == "compression"].copy()
    for deformation, group in comp_ref.groupby("deformation", sort=True):
        axes[1, 0].plot(group["time_days"], group["predicted_stress_pa"], marker="o", label=f"lambda={deformation:.2f}")
    axes[1, 0].set_title("Compression response vs time")
    axes[1, 0].set_xlabel("time_days")
    axes[1, 0].set_ylabel("predicted stress [Pa]")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(frameon=False)

    shear_ref = reference_df.loc[reference_df["loading"] == "shear"].copy()
    for deformation, group in shear_ref.groupby("deformation", sort=True):
        axes[1, 1].plot(group["time_days"], group["predicted_stress_pa"], marker="o", label=f"gamma={deformation:.2f}")
    axes[1, 1].set_title("Shear response vs time")
    axes[1, 1].set_xlabel("time_days")
    axes[1, 1].set_ylabel("predicted stress [Pa]")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(frameon=False)

    fig.suptitle(f"Time-effect analysis | {COLOR_LABELS[color]}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = Fig1CompShearConfig(
        root=args.root.resolve(),
        axial_stress_column=args.axial_stress_column,
        compression_sign=args.compression_sign,
        ridge_alpha=args.ridge_alpha,
        train_fraction=args.train_fraction,
    )
    paths = ProjectPaths(root=cfg.root)
    raw_df = load_all_workbooks(paths.data_dir, axial_stress_column=cfg.axial_stress_column, compression_sign=cfg.compression_sign)
    summary_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)
    summary_df = summary_df.loc[summary_df["loading"].isin(["compression", "shear"])].copy()

    split_suffix = "full_train" if cfg.train_fraction >= 0.999 else f"train_{int(round(cfg.train_fraction * 100)):02d}_val_{int(round((1.0 - cfg.train_fraction) * 100)):02d}"
    out_dir = paths.output_dir / f"fig1_comp_shear_time_{split_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "robust_summary_curves.csv", index=False)
    interpolated_df.to_csv(out_dir / "interpolated_individual_curves.csv", index=False)
    diagnostics_df.to_csv(out_dir / "curve_diagnostics.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    all_metrics = []
    for color in sorted(summary_df["color"].unique()):
        color_df = summary_df.loc[summary_df["color"] == color].copy()
        color_dir = out_dir / color
        color_dir.mkdir(parents=True, exist_ok=True)

        if cfg.train_fraction < 0.999:
            models_by_day, metrics_df = _fit_color_joint_time_models(color_df, cfg)
        else:
            models_by_day = {}
            metrics_rows = []
            for day in sorted(color_df["time_days"].unique()):
                day_df = color_df.loc[color_df["time_days"] == day].copy()
                train_df, val_df = _split_day_df(day_df, cfg.train_fraction)
                model, meta = _fit_day_joint_model(train_df, cfg)
                models_by_day[float(day)] = model
                for loading in ["compression", "shear"]:
                    train_panel = train_df.loc[train_df["loading"] == loading].copy()
                    train_score = _score_model(model, train_panel, loading)
                    row = {
                        "color": color,
                        "time_days": float(day),
                        "loading": loading,
                        "split": "train",
                        "r2": float(train_score),
                        **meta,
                    }
                    if not val_df.empty:
                        val_panel = val_df.loc[val_df["loading"] == loading].copy()
                        row["val_r2"] = _score_model(model, val_panel, loading)
                        row["n_train"] = int(len(train_panel))
                        row["n_val"] = int(len(val_panel))
                    else:
                        row["val_r2"] = np.nan
                        row["n_train"] = int(len(train_panel))
                        row["n_val"] = 0
                    metrics_rows.append(row)
                (color_dir / f"day_{int(day)}_model.json").write_text(json.dumps(model.export(), indent=2), encoding="utf-8")
            metrics_df = pd.DataFrame(metrics_rows)

        for day, model in models_by_day.items():
            (color_dir / f"day_{int(day)}_model.json").write_text(json.dumps(model.export(), indent=2), encoding="utf-8")
        metrics_df.to_csv(color_dir / "fit_metrics.csv", index=False)
        all_metrics.append(metrics_df)

        _plot_color_discovery(
            color=color,
            color_df=color_df,
            models_by_day=models_by_day,
            metrics_df=metrics_df,
            save_path=color_dir / f"{color}_fig1_comp_shear_three_days.png",
        )

        coeff_df, reference_df, time_summary_df = _time_effect_tables(color, color_df, models_by_day, cfg)
        coeff_df.to_csv(color_dir / "time_term_coefficients.csv", index=False)
        reference_df.to_csv(color_dir / "time_reference_stresses.csv", index=False)
        time_summary_df.to_csv(color_dir / "time_effect_summary.csv", index=False)
        _plot_time_effects(
            color=color,
            coeff_df=coeff_df,
            reference_df=reference_df,
            summary_df=time_summary_df,
            save_path=color_dir / f"{color}_time_effects.png",
        )

    pd.concat(all_metrics, ignore_index=True).to_csv(out_dir / "all_fit_metrics.csv", index=False)


if __name__ == "__main__":
    main()
