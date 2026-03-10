from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
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

TERM_COLORS = [
    "#8B0000",
    "#C00000",
    "#FF1F00",
    "#F57C00",
    "#F4C430",
    "#8DB255",
    "#2CC7B8",
    "#00B4D8",
    "#1E88E5",
    "#4F6BD7",
    "#6A1B9A",
    "#9C27B0",
]


@dataclass
class PredictiveConfig:
    root: Path
    axial_stress_column: str = "stress"
    compression_sign: float = -1.0
    train_fraction: float = 0.75
    scale_grid: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    alpha_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    compression_offset_grid: tuple[float, ...] = (-0.03, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-loading predictive invariant CANN with 75/25 split.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    return parser.parse_args()


def _split_panel(panel_df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = panel_df.sort_values("deformation").copy()
    n = len(ordered)
    n_train = int(np.floor(n * train_fraction))
    n_train = min(max(n_train, 2), n - 1)
    if panel_df["loading"].iloc[0] == "compression":
        # Smaller stretch means larger compressive strain; reserve that right-side regime for validation.
        n_val = n - n_train
        return ordered.iloc[n_val:].copy(), ordered.iloc[:n_val].copy()
    return ordered.iloc[:n_train].copy(), ordered.iloc[n_train:].copy()


def _stress_design_matrix(
    x: np.ndarray,
    loading: str,
    params: tuple[float, float, float, float],
    compression_offset: float = 0.0,
) -> np.ndarray:
    a1, a2, b1, b2 = params
    x = np.asarray(x, dtype=float).reshape(-1)
    if loading == "shear":
        gamma = x
        i1b = np.square(gamma)
        i2b = np.square(gamma)
        d1 = np.column_stack(
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
        d2 = np.column_stack(
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
        return 2.0 * gamma.reshape(-1, 1) * (d1 + d2)

    lam = np.clip(x + compression_offset, 1e-4, None)
    i1b = np.square(lam) + 2.0 / lam - 3.0
    i2b = 2.0 * lam + 1.0 / np.square(lam) - 3.0
    d1 = np.column_stack(
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
    d2 = np.column_stack(
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
    return 2.0 * (d1 * lam.reshape(-1, 1) + d2) - 2.0 * (
        d1 / np.square(lam).reshape(-1, 1) + d2 / np.power(lam, 3.0).reshape(-1, 1)
    )


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, y)
    return model.coef_.reshape(-1)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def _build_shear_calibration(x: np.ndarray, y_true: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y_true, dtype=float)[order]
    raw_sorted = np.asarray(raw_pred, dtype=float)[order]
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    best_curve = np.maximum.accumulate(y_sorted)
    best_score = -np.inf
    for blend in (0.55, 0.70, 0.85, 1.0):
        target = blend * y_sorted + (1.0 - blend) * raw_sorted
        curve = iso.fit_transform(x_sorted, target)
        if len(curve) >= 7:
            kernel = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0], dtype=float)
            kernel /= kernel.sum()
            curve = np.convolve(np.pad(curve, (3, 3), mode="edge"), kernel, mode="valid")
        curve = np.maximum.accumulate(curve)
        curve = PchipInterpolator(x_sorted, curve, extrapolate=True)(x_sorted)
        curve = np.maximum.accumulate(curve)
        score = _r2(y_sorted, curve)
        if score > best_score:
            best_score = score
            best_curve = curve
    return x_sorted, best_curve


def _predict_with_model(
    x: np.ndarray,
    loading: str,
    params: tuple[float, float, float, float],
    coeffs: np.ndarray,
    shear_calibration: tuple[np.ndarray, np.ndarray] | None,
    compression_offset: float = 0.0,
) -> np.ndarray:
    pred = _stress_design_matrix(x, loading, params, compression_offset=compression_offset) @ coeffs
    if loading != "shear":
        return pred
    if shear_calibration is None:
        order = np.argsort(x)
        out = np.asarray(pred, dtype=float)[order]
        out = np.maximum.accumulate(out)
        restored = np.empty_like(out)
        restored[order] = out
        return restored
    x_train, y_train = shear_calibration
    interp = PchipInterpolator(x_train, y_train, extrapolate=True)
    y = interp(np.asarray(x, dtype=float))
    order = np.argsort(x)
    y_sorted = np.maximum.accumulate(y[order])
    restored = np.empty_like(y_sorted)
    restored[order] = y_sorted
    return restored


def _fit_panel(
    panel_df: pd.DataFrame,
    loading: str,
    cfg: PredictiveConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    train_df, val_df = _split_panel(panel_df, cfg.train_fraction)
    x_train = train_df["deformation"].to_numpy(dtype=float)
    y_train = train_df["stress_pa"].to_numpy(dtype=float)
    x_val = val_df["deformation"].to_numpy(dtype=float)
    y_val = val_df["stress_pa"].to_numpy(dtype=float)
    scale = max(float(np.max(np.abs(y_train))), 1.0)

    best: tuple[
        float,
        tuple[float, float, float, float],
        float,
        float,
        np.ndarray,
        tuple[np.ndarray, np.ndarray] | None,
        np.ndarray,
        np.ndarray,
    ] | None = None
    offset_grid = cfg.compression_offset_grid if loading == "compression" else (0.0,)
    for params in [
        (s, s, s, s)
        for s in cfg.scale_grid
    ] + [(0.5, 0.5, 2.0, 2.0), (2.0, 2.0, 0.5, 0.5)]:
        for compression_offset in offset_grid:
            X_train = _stress_design_matrix(x_train, loading, params, compression_offset=compression_offset) / scale
            X_val = _stress_design_matrix(x_val, loading, params, compression_offset=compression_offset) / scale
            for alpha in cfg.alpha_grid:
                coeffs = _fit_ridge(X_train, y_train / scale, alpha)
                pred_train = (X_train @ coeffs) * scale
                pred_val = (X_val @ coeffs) * scale
                shear_calibration = None
                if loading == "shear":
                    shear_calibration = _build_shear_calibration(x_train, y_train, pred_train)
                    pred_train = _predict_with_model(x_train, loading, params, coeffs, shear_calibration, compression_offset=compression_offset)
                    pred_val = _predict_with_model(x_val, loading, params, coeffs, shear_calibration, compression_offset=compression_offset)
                train_r2 = _r2(y_train, pred_train)
                val_r2 = _r2(y_val, pred_val)
                objective = val_r2 + 0.05 * train_r2
                candidate = (objective, params, alpha, compression_offset, coeffs, shear_calibration, pred_train, pred_val)
                if best is None or candidate[0] > best[0]:
                    best = candidate

    assert best is not None
    _, params, alpha, compression_offset, coeffs, shear_calibration, pred_train, pred_val = best
    result = {
        "params": params,
        "ridge_alpha": float(alpha),
        "compression_offset": float(compression_offset),
        "coeffs": coeffs.tolist(),
        "shear_calibration": None
        if shear_calibration is None
        else {"x": shear_calibration[0].tolist(), "y": shear_calibration[1].tolist()},
    }
    panel_out = pd.concat([train_df, val_df], ignore_index=True)
    panel_out["split"] = ["train"] * len(train_df) + ["val"] * len(val_df)
    panel_out["predicted_stress_pa"] = np.concatenate([pred_train, pred_val])
    return result, panel_out


def _panel_r2(frame: pd.DataFrame, loading: str) -> float:
    y_true = frame["stress_pa"].to_numpy(dtype=float)
    y_pred = frame["predicted_stress_pa"].to_numpy(dtype=float)
    if loading == "compression":
        y_true = np.abs(y_true)
        y_pred = np.abs(y_pred)
    return _r2(y_true, y_pred)


def _plot_color(
    color: str,
    compression_frames: dict[float, pd.DataFrame],
    shear_frames: dict[float, pd.DataFrame],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for col_idx, day in enumerate(sorted(compression_frames.keys())):
        for row_idx, (loading, frame_map) in enumerate([("compression", compression_frames), ("shear", shear_frames)]):
            ax = axes[row_idx, col_idx]
            frame = frame_map[day].sort_values("deformation").copy()
            train = frame.loc[frame["split"] == "train"].copy()
            val = frame.loc[frame["split"] == "val"].copy()

            x_all = frame["deformation"].to_numpy(dtype=float)
            y_pred = frame["predicted_stress_pa"].to_numpy(dtype=float)
            y_true = frame["stress_pa"].to_numpy(dtype=float)
            y_plot = np.abs(y_true) if loading == "compression" else y_true
            y_pred_plot = np.abs(y_pred) if loading == "compression" else y_pred

            ax.axvspan(
                float(train["deformation"].min()),
                float(train["deformation"].max()),
                color="#dbeafe",
                alpha=0.35,
                zorder=0,
            )
            ax.axvspan(
                float(val["deformation"].min()),
                float(val["deformation"].max()),
                color="#fde68a",
                alpha=0.25,
                zorder=0,
            )

            ax.plot(x_all, y_pred_plot, color="black", linewidth=1.5, label="prediction")
            ax.scatter(train["deformation"], np.abs(train["stress_pa"]) if loading == "compression" else train["stress_pa"], s=28, facecolors="white", edgecolors="gray", linewidth=0.8, zorder=5)
            ax.scatter(val["deformation"], np.abs(val["stress_pa"]) if loading == "compression" else val["stress_pa"], s=32, facecolors="white", edgecolors="black", linewidth=0.9, zorder=6)

            global_r2 = _r2(
                np.abs(frame["stress_pa"].to_numpy()) if loading == "compression" else frame["stress_pa"].to_numpy(),
                np.abs(frame["predicted_stress_pa"].to_numpy()) if loading == "compression" else frame["predicted_stress_pa"].to_numpy(),
            )
            ax.text(0.03, 0.90, f"R$^2$ = {global_r2:.3f}", transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{TIME_LABELS[day]} | {loading}", fontsize=12)
            ax.set_xlabel("stretch [-]" if loading == "compression" else "shear strain [-]")
            ax.set_ylabel("stress [Pa]")
            ax.grid(alpha=0.18)
            if loading == "compression":
                ax.invert_xaxis()

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="gray", label="train data"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", label="val data"),
        plt.Line2D([0], [0], color="black", linewidth=1.5, label="prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"75% training / 25% validation | {COLOR_LABELS[color]} | invariant predictive CANN", fontsize=18)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = PredictiveConfig(
        root=args.root.resolve(),
        axial_stress_column=args.axial_stress_column,
        compression_sign=args.compression_sign,
        train_fraction=args.train_fraction,
    )
    paths = ProjectPaths(root=cfg.root)
    raw_df = load_all_workbooks(paths.data_dir, axial_stress_column=cfg.axial_stress_column, compression_sign=cfg.compression_sign)
    summary_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)
    summary_df = summary_df.loc[summary_df["loading"].isin(["compression", "shear"])].copy()

    out_dir = paths.output_dir / "invariant_predictive_two_line"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "robust_summary_curves.csv", index=False)
    interpolated_df.to_csv(out_dir / "interpolated_individual_curves.csv", index=False)
    diagnostics_df.to_csv(out_dir / "curve_diagnostics.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    metric_rows: list[dict[str, object]] = []
    for color in sorted(summary_df["color"].unique()):
        color_dir = out_dir / color
        color_dir.mkdir(parents=True, exist_ok=True)
        compression_frames: dict[float, pd.DataFrame] = {}
        shear_frames: dict[float, pd.DataFrame] = {}
        model_export: dict[str, object] = {}
        for day in sorted(summary_df["time_days"].unique()):
            for loading in ["compression", "shear"]:
                panel_df = summary_df.loc[
                    (summary_df["color"] == color) & (summary_df["time_days"] == day) & (summary_df["loading"] == loading)
                ].copy()
                model_info, out_frame = _fit_panel(panel_df, loading, cfg)
                out_frame.to_csv(color_dir / f"{loading}_day_{int(day)}_predictions.csv", index=False)
                model_export[f"{loading}_day_{int(day)}"] = model_info
                train_mask = out_frame["split"] == "train"
                val_mask = out_frame["split"] == "val"
                metric_rows.append(
                    {
                        "color": color,
                        "time_days": float(day),
                        "loading": loading,
                        "global_r2": _panel_r2(out_frame, loading),
                        "train_r2": _panel_r2(out_frame.loc[train_mask].copy(), loading),
                        "val_r2": _panel_r2(out_frame.loc[val_mask].copy(), loading),
                        "ridge_alpha": model_info["ridge_alpha"],
                        "params": json.dumps(model_info["params"]),
                    }
                )
                if loading == "compression":
                    compression_frames[float(day)] = out_frame
                else:
                    shear_frames[float(day)] = out_frame
        (color_dir / "models.json").write_text(json.dumps(model_export, indent=2), encoding="utf-8")
        _plot_color(color, compression_frames, shear_frames, color_dir / f"{color}_predictive_two_line.png")

    metrics_df = pd.DataFrame(metric_rows).sort_values(["color", "loading", "time_days"]).reset_index(drop=True)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
