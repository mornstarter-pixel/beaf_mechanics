from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.metrics import r2_score

from core.data_pipeline import load_all_workbooks, preprocess_robust_curves
from core.invariant_baseline import (
    COLOR_LABELS,
    TIME_LABELS,
)
from core.settings import ProjectPaths


DISPLAY_LABELS = {
    "I1b": "I1-3",
    "exp(I1b)-1": "exp(I1-3)-1",
    "log(1-I1b)": "ln(1-(I1-3))",
    "I1b^2": "(I1-3)^2",
    "exp(I1b^2)-1": "exp((I1-3)^2)-1",
    "log(1-I1b^2)": "ln(1-((I1-3)^2))",
    "I2b": "I2-3",
    "exp(I2b)-1": "exp(I2-3)-1",
    "log(1-I2b)": "ln(1-(I2-3))",
    "I2b^2": "(I2-3)^2",
    "exp(I2b^2)-1": "exp((I2-3)^2)-1",
    "log(1-I2b^2)": "ln(1-((I2-3)^2))",
}

ARCH_TERM_COLORS = {
    "I1b": "#8B0000",
    "exp(I1b)-1": "#D40000",
    "log(1-I1b)": "#FF0D00",
    "I1b^2": "#F07A00",
    "exp(I1b^2)-1": "#F4B000",
    "log(1-I1b^2)": "#7FAA61",
    "I2b": "#49B3A8",
    "exp(I2b)-1": "#19AFCF",
    "log(1-I2b)": "#2D8BCB",
    "I2b^2": "#4E63C3",
    "exp(I2b^2)-1": "#6D35BF",
    "log(1-I2b^2)": "#9A06B9",
}

ARCH_TERM_ORDER = [
    "I1b",
    "exp(I1b)-1",
    "log(1-I1b)",
    "I1b^2",
    "exp(I1b^2)-1",
    "log(1-I1b^2)",
    "I2b",
    "exp(I2b)-1",
    "log(1-I2b)",
    "I2b^2",
    "exp(I2b^2)-1",
    "log(1-I2b^2)",
]


PARAM_GRID = [
    (0.1, 0.1, 0.1, 0.1),
    (0.25, 0.25, 0.25, 0.25),
    (0.5, 0.5, 0.5, 0.5),
    (1.0, 1.0, 1.0, 1.0),
    (2.0, 2.0, 2.0, 2.0),
    (4.0, 4.0, 4.0, 4.0),
    (8.0, 8.0, 8.0, 8.0),
    (0.25, 0.25, 1.0, 1.0),
    (0.5, 0.5, 2.0, 2.0),
    (1.0, 1.0, 0.25, 0.25),
    (2.0, 2.0, 0.5, 0.5),
    (4.0, 4.0, 1.0, 1.0),
]
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot robust-summary compression fits and term contributions.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    return parser.parse_args()


def _relative_stack(contrib: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    total = np.sum(np.abs(contrib), axis=1, keepdims=True)
    total = np.maximum(total, 1e-8)
    return np.abs(contrib) / total * np.abs(y_pred).reshape(-1, 1)


def _compression_strain_from_stretch(stretch: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(stretch, dtype=float)


def _stretch_from_compression_strain(strain: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(strain, dtype=float)


def _zero_correct_compression(day_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    corrected = day_df.sort_values("deformation").copy()
    strain = _compression_strain_from_stretch(corrected["deformation"].to_numpy(dtype=float))
    stress = np.abs(corrected["stress_pa"].to_numpy(dtype=float))
    x0 = float(strain[0])
    y0 = float(stress[0])
    x1 = float(strain[-1])
    y1 = float(stress[-1])
    if abs(x1 - x0) < 1e-12:
        offset = y0
    else:
        slope = (y1 - y0) / (x1 - x0)
        offset = y0 - slope * x0
    corrected_stress = np.maximum(stress - offset, 0.0)
    corrected["stress_pa"] = corrected_stress
    corrected["zero_correction_pa"] = offset
    return corrected, float(offset)


def _compression_design_and_contrib(
    compression_strain: np.ndarray,
    params: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    a1, a2, b1, b2 = params
    lam = np.clip(_stretch_from_compression_strain(np.asarray(compression_strain, dtype=float).reshape(-1)), 1e-5, None)
    i1b = np.square(lam) + 2.0 / lam - 3.0
    i2b = 2.0 * lam + 1.0 / np.square(lam) - 3.0

    d_i1_terms = np.column_stack(
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
    d_i2_terms = np.column_stack(
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
    contrib = 2.0 * (d_i1_terms * lam.reshape(-1, 1) + d_i2_terms) - 2.0 * (
        d_i1_terms / np.square(lam).reshape(-1, 1) + d_i2_terms / np.power(lam, 3.0).reshape(-1, 1)
    )
    return contrib, contrib


def _fit_nonnegative_day(day_df: pd.DataFrame) -> tuple[dict[str, object], float]:
    x = _compression_strain_from_stretch(day_df["deformation"].to_numpy(dtype=float))
    y = np.abs(day_df["stress_pa"].to_numpy(dtype=float))
    best = None
    for params in PARAM_GRID:
        X_signed, contrib_signed = _compression_design_and_contrib(x, params)
        X = -X_signed
        contrib = -contrib_signed
        scale = np.maximum(np.linalg.norm(X, axis=0), 1e-8)
        Xs = X / scale.reshape(1, -1)
        for alpha in ALPHA_GRID:
            X_aug = np.vstack([Xs, np.sqrt(alpha) * np.eye(Xs.shape[1])])
            y_aug = np.concatenate([y, np.zeros(Xs.shape[1], dtype=float)])
            result = lsq_linear(X_aug, y_aug, bounds=(0.0, np.inf), lsmr_tol="auto", verbose=0)
            coeffs = result.x / scale
            pred = X @ coeffs
            if np.min(pred) < -1e-8:
                continue
            r2 = r2_score(y, pred)
            candidate = (
                r2,
                params,
                alpha,
                coeffs,
                pred,
                contrib * coeffs.reshape(1, -1),
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise RuntimeError("No feasible nonnegative compression model found.")
    r2, params, alpha, coeffs, pred, weighted_contrib = best
    return {
        "params": params,
        "ridge_alpha": alpha,
        "coeffs": coeffs.tolist(),
        "pred": pred,
        "contrib": weighted_contrib,
    }, float(r2)


def _fit_color_compression(summary_df: pd.DataFrame, color: str) -> tuple[dict[float, object], pd.DataFrame]:
    color_df = summary_df.loc[(summary_df["color"] == color) & (summary_df["loading"] == "compression")].copy()
    models = {}
    rows = []
    for day in sorted(color_df["time_days"].unique()):
        raw_day_df = color_df.loc[color_df["time_days"] == day].sort_values("deformation").copy()
        day_df, zero_correction = _zero_correct_compression(raw_day_df)
        fit_bundle, r2 = _fit_nonnegative_day(day_df)
        models[float(day)] = {"frame": day_df, "zero_correction_pa": zero_correction, **fit_bundle}
        rows.append(
            {
                "color": color,
                "time_days": float(day),
                "r2": r2,
                "zero_correction_pa": zero_correction,
            }
        )
    return models, pd.DataFrame(rows)


def _plot_rainbow(color: str, models: dict[float, object], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", label="Exp Mean"),
        plt.Line2D([0], [0], color="black", linewidth=2.0, label="CANN Prediction"),
    ]
    handles.extend(
        [
            plt.matplotlib.patches.Patch(
                facecolor=ARCH_TERM_COLORS[term],
                edgecolor="none",
                label=DISPLAY_LABELS.get(term, term),
            )
            for term in ARCH_TERM_ORDER
        ]
    )
    for ax, day in zip(axes, sorted(models.keys())):
        bundle = models[day]
        frame = bundle["frame"].sort_values("deformation").copy()
        x = _compression_strain_from_stretch(frame["deformation"].to_numpy(dtype=float))
        y = np.abs(frame["stress_pa"].to_numpy(dtype=float))
        pred = np.asarray(bundle["pred"], dtype=float)
        stack = _relative_stack(bundle["contrib"], pred)
        lower = np.zeros_like(pred)
        for idx, term in enumerate(ARCH_TERM_ORDER):
            upper = lower + stack[:, idx]
            ax.fill_between(x, lower, upper, color=ARCH_TERM_COLORS[term], alpha=0.92, linewidth=0.0)
            lower = upper
        ax.plot(x, pred, color="black", linewidth=2.0)
        ax.scatter(x, y, s=55, facecolors="white", edgecolors="black", linewidth=1.1, zorder=5)
        r2 = r2_score(y, pred)
        ax.text(0.03, 0.90, f"R$^2$ = {r2:.3f}", transform=ax.transAxes, fontsize=12)
        ax.set_title(f"{TIME_LABELS[day]} | Compression", fontsize=16)
        ax.set_xlabel("Compression Strain [-]")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Compression Stress (Pa)")
    fig.suptitle(f"Term Contributions Evolution | {COLOR_LABELS[color]}", fontsize=22)
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def _plot_plain(color: str, models: dict[float, object], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", label="Exp Data (Mean)"),
        plt.Line2D([0], [0], color="black", linewidth=2.2, label="CANN Prediction"),
    ]
    for ax, day in zip(axes, sorted(models.keys())):
        bundle = models[day]
        frame = bundle["frame"].sort_values("deformation").copy()
        x = _compression_strain_from_stretch(frame["deformation"].to_numpy(dtype=float))
        y = np.abs(frame["stress_pa"].to_numpy(dtype=float))
        pred = np.asarray(bundle["pred"], dtype=float)
        ax.plot(x, pred, color="black", linewidth=2.2)
        ax.scatter(x, y, s=55, facecolors="white", edgecolors="black", linewidth=1.1)
        r2 = r2_score(y, pred)
        ax.text(0.03, 0.90, f"R$^2$ = {r2:.3f}", transform=ax.transAxes, fontsize=12)
        ax.set_title(f"{TIME_LABELS[day]} | Compression", fontsize=16)
        ax.set_xlabel("Compression Strain [-]")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Compression Stress (Pa)")
    fig.suptitle(f"CANN Fit on Mean Data | {COLOR_LABELS[color]}", fontsize=22)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def _plot_time_evolution(color: str, models: dict[float, object], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {0.0: "#1f77b4", 1.0: "#ff7f0e", 2.0: "#2ca02c"}
    for day in sorted(models.keys()):
        bundle = models[day]
        frame = bundle["frame"].sort_values("deformation").copy()
        x = _compression_strain_from_stretch(frame["deformation"].to_numpy(dtype=float))
        y = np.abs(frame["stress_pa"].to_numpy(dtype=float))
        pred = np.asarray(bundle["pred"], dtype=float)
        ax.scatter(x, y, s=180, facecolors=colors[day], edgecolors=colors[day], alpha=0.28, linewidth=1.8, label=f"Exp Mean ({TIME_LABELS[day]})")
        ax.plot(x, pred, color=colors[day], linewidth=4.5, label=f"CANN ({TIME_LABELS[day]})")
    ax.set_title(f"Time-Evolution Mechanical Response ({COLOR_LABELS[color]})", fontsize=26)
    ax.set_xlabel("Compression Strain [-]", fontsize=20)
    ax.set_ylabel("Compression Stress (Pa)", fontsize=22)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True, fontsize=12, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(root=args.root.resolve())
    raw_df = load_all_workbooks(
        paths.data_dir,
        axial_stress_column=args.axial_stress_column,
        compression_sign=args.compression_sign,
    )
    summary_df, _, _ = preprocess_robust_curves(raw_df)
    summary_df = summary_df.loc[summary_df["loading"] == "compression"].copy()

    out_dir = paths.output_dir / "robust_summary_compression_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "robust_summary_compression.csv", index=False)

    all_metrics = []
    for color in sorted(summary_df["color"].unique()):
        color_dir = out_dir / color
        color_dir.mkdir(parents=True, exist_ok=True)
        models, metrics_df = _fit_color_compression(summary_df, color)
        metrics_df.to_csv(color_dir / "fit_metrics.csv", index=False)
        all_metrics.append(metrics_df)
        serializable = {
            str(day): {
                "term_labels": ARCH_TERM_ORDER,
                "params": bundle["params"],
                "ridge_alpha": bundle["ridge_alpha"],
                "zero_correction_pa": bundle["zero_correction_pa"],
                "coeffs": bundle["coeffs"],
            }
            for day, bundle in models.items()
        }
        (color_dir / "models.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        _plot_rainbow(color, models, color_dir / f"{color}_compression_rainbow.png")
        _plot_plain(color, models, color_dir / f"{color}_compression_plain.png")
        _plot_time_evolution(color, models, color_dir / f"{color}_compression_time_evolution.png")

    pd.concat(all_metrics, ignore_index=True).to_csv(out_dir / "all_fit_metrics.csv", index=False)


if __name__ == "__main__":
    main()
