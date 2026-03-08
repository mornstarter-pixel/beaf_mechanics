from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

from core.data_pipeline import load_all_workbooks, preprocess_robust_curves
from core.invariant_baseline import COLOR_LABELS, LOADING_LABELS, TERM_LABELS, PolynomialInvariantCANN, TIME_LABELS
from core.settings import ProjectPaths


TRAIN_VARIANTS = [
    ("compression_only", "train compression"),
    ("unloading_only", "train unloading"),
    ("shear_only", "train shear"),
    ("joint_comp_shear", "train comp+shear"),
]

VARIANT_LOADINGS = {
    "compression_only": ["compression"],
    "unloading_only": ["unloading"],
    "shear_only": ["shear"],
    "joint_comp_shear": ["compression", "shear"],
}

TOP_K_TERMS = 4
DISCOVERY_TERM_INDICES = [0, 1, 2, 6, 7, 8, 12]
DISCOVERY_TERM_LABELS = [TERM_LABELS[idx] for idx in DISCOVERY_TERM_INDICES]


@dataclass
class InvariantDiscoveryConfig:
    ridge_alpha: float = 1e-6
    seed: int = 42


VARIANT_LASSO_ALPHA = {
    "compression_only": 1e-2,
    "unloading_only": 1e-2,
    "shear_only": 1e-2,
    "joint_comp_shear": 3e-2,
}


class TimeBranchInvariantCANN:
    def __init__(self, coeff_bank: np.ndarray) -> None:
        self.coeff_bank = np.asarray(coeff_bank, dtype=float)
        self._base_model = PolynomialInvariantCANN(seed=42)

    def _expand_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        full = np.zeros((len(TERM_LABELS), 1), dtype=np.float32)
        full[DISCOVERY_TERM_INDICES, 0] = np.asarray(coeffs, dtype=np.float32).reshape(-1)
        return full

    def _predict_with_coeffs(self, x: np.ndarray, loading: str, coeffs: np.ndarray) -> np.ndarray:
        self._base_model.set_coefficients(self._expand_coeffs(coeffs))
        if loading == "shear":
            return self._base_model.predict_shear(x).numpy().reshape(-1)
        return self._base_model.predict_uniaxial(x).numpy().reshape(-1)

    def predict(self, x: np.ndarray, time_days: np.ndarray, loading: str) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        t_idx = np.clip(np.rint(np.asarray(time_days).reshape(-1)).astype(int), 0, 2)
        out = np.zeros(len(x), dtype=float)
        for day_idx in range(3):
            mask = t_idx == day_idx
            if np.any(mask):
                out[mask] = self._predict_with_coeffs(x[mask], loading, self.coeff_bank[:, day_idx : day_idx + 1])
        return out

    def contributions(self, x: np.ndarray, time_days: np.ndarray, loading: str) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        t_idx = np.clip(np.rint(np.asarray(time_days).reshape(-1)).astype(int), 0, 2)
        parts = np.zeros((len(x), len(DISCOVERY_TERM_LABELS)), dtype=float)
        for day_idx in range(3):
            mask = t_idx == day_idx
            if not np.any(mask):
                continue
            for term_idx in range(len(DISCOVERY_TERM_LABELS)):
                coeffs = np.zeros((len(DISCOVERY_TERM_LABELS), 1), dtype=np.float32)
                coeffs[term_idx, 0] = float(self.coeff_bank[term_idx, day_idx])
                parts[mask, term_idx] = self._predict_with_coeffs(x[mask], loading, coeffs)
        return parts

    def export_weights(self) -> dict[str, object]:
        return {
            "terms": DISCOVERY_TERM_LABELS,
            "coeff_bank": self.coeff_bank.tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invariant CANN discovery panels.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    return parser.parse_args()


def _stress_design_matrix(loading: str, deformation: np.ndarray) -> np.ndarray:
    model = PolynomialInvariantCANN(seed=42)
    x = np.asarray(deformation, dtype=np.float32).reshape(-1, 1)
    cols = []
    for idx in range(len(TERM_LABELS)):
        coeffs = np.zeros((len(TERM_LABELS), 1), dtype=np.float32)
        coeffs[idx, 0] = 1.0
        model.set_coefficients(coeffs)
        if loading == "shear":
            cols.append(model.predict_shear(x).numpy().reshape(-1))
        else:
            cols.append(model.predict_uniaxial(x).numpy().reshape(-1))
    return np.stack(cols, axis=1)


def _fit_variant(color_df: pd.DataFrame, variant: str, cfg: InvariantDiscoveryConfig) -> TimeBranchInvariantCANN:
    target_df = color_df.loc[color_df["loading"].isin(VARIANT_LOADINGS[variant])].copy()
    rows = []
    targets = []
    for loading, group in target_df.groupby("loading", sort=False):
        X_loading = _stress_design_matrix(loading, group["deformation"].to_numpy())[:, DISCOVERY_TERM_INDICES]
        day_idx = np.clip(np.rint(group["time_days"].to_numpy()).astype(int), 0, 2)
        X = np.zeros((len(group), len(DISCOVERY_TERM_LABELS) * 3), dtype=float)
        for row_idx, di in enumerate(day_idx):
            start = di * len(DISCOVERY_TERM_LABELS)
            X[row_idx, start : start + len(DISCOVERY_TERM_LABELS)] = X_loading[row_idx]
        rows.append(X)
        targets.append(group["stress_pa"].to_numpy(dtype=float))
    X_all = np.vstack(rows)
    y_all = np.concatenate(targets)
    # Standardized sparse solve to avoid large canceling coefficients that ruin discovery plots.
    scale = np.maximum(np.std(X_all, axis=0), 1e-8)
    X_std = X_all / scale
    alpha = VARIANT_LASSO_ALPHA[variant]
    solver = Lasso(alpha=alpha, fit_intercept=False, max_iter=300000)
    solver.fit(X_std, y_all)
    coeffs = solver.coef_ / scale
    coeff_bank = coeffs.reshape(3, len(DISCOVERY_TERM_LABELS)).T
    return TimeBranchInvariantCANN(coeff_bank)


def _compress_contributions(contrib: np.ndarray, labels: list[str], top_k: int = TOP_K_TERMS) -> tuple[np.ndarray, list[str]]:
    if contrib.shape[1] <= top_k:
        return contrib, labels
    importance = np.mean(np.abs(contrib), axis=0)
    keep_idx = np.argsort(importance)[::-1][:top_k]
    keep_idx = np.sort(keep_idx)
    kept = contrib[:, keep_idx]
    kept_labels = [labels[idx] for idx in keep_idx]
    dropped_idx = [idx for idx in range(contrib.shape[1]) if idx not in set(keep_idx.tolist())]
    others = contrib[:, dropped_idx].sum(axis=1, keepdims=True)
    return np.concatenate([kept, others], axis=1), kept_labels + ["others"]


def _plot_time_panel(color: str, time_days: float, color_df: pd.DataFrame, trained_models: dict[str, TimeBranchInvariantCANN], save_path: Path) -> list[dict]:
    subset = color_df.loc[color_df["time_days"] == time_days].copy()
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    rows: list[dict] = []

    ordered_labels = DISCOVERY_TERM_LABELS + ["others"]
    cmap = plt.cm.turbo(np.linspace(0.03, 0.97, len(ordered_labels)))
    label_to_color = {label: cmap[idx] for idx, label in enumerate(ordered_labels)}

    for row_idx, loading in enumerate(["compression", "unloading", "shear"]):
        display_df = subset.loc[subset["loading"] == loading].sort_values("deformation").copy()
        x = display_df["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1)
        t = display_df["time_days"].to_numpy(dtype=np.float32).reshape(-1, 1)
        y_true = display_df["stress_pa"].to_numpy(dtype=float)

        for col_idx, (variant, variant_title) in enumerate(TRAIN_VARIANTS):
            ax = axes[row_idx, col_idx]
            model = trained_models[variant]
            raw_contrib = model.contributions(x, t, loading)
            y_pred = raw_contrib.sum(axis=1)
            raw_contrib, labels = _compress_contributions(raw_contrib, DISCOVERY_TERM_LABELS)
            display_sign = -1.0 if np.nanmean(y_true) < 0.0 else 1.0
            y_true_plot = display_sign * y_true
            y_pred_plot = display_sign * y_pred
            # For discovery visualization, show relative attribution magnitudes so the stack follows the
            # predicted response instead of plotting raw signed terms that can cancel dramatically.
            denom = np.sum(np.abs(raw_contrib), axis=1, keepdims=True)
            denom = np.maximum(denom, 1e-8)
            contrib_plot = (np.abs(raw_contrib) / denom) * y_pred_plot.reshape(-1, 1)

            lower = np.zeros_like(y_pred_plot)
            for idx in range(contrib_plot.shape[1]):
                term = contrib_plot[:, idx]
                term_color = label_to_color[labels[idx]]
                upper = lower + term
                ax.fill_between(x.reshape(-1), lower, upper, color=term_color, linewidth=0.0, alpha=0.9)
                lower = upper
            ax.plot(x.reshape(-1), y_pred_plot, color="k", linewidth=1.0)
            ax.scatter(x.reshape(-1), y_true_plot, s=28, facecolors="white", edgecolors="gray", linewidth=0.8, zorder=5)
            ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)

            score = r2_score(y_true, y_pred)
            rows.append({"color": color, "time_days": time_days, "loading": loading, "variant": variant, "r2": float(score)})

            ax.set_title(f"{COLOR_LABELS[color]} | {variant_title}", fontsize=11)
            ax.text(0.03, 0.90, f"R$^2$ = {score:.4f}", transform=ax.transAxes, fontsize=10)
            ax.set_xlabel("stretch [-]" if loading != "shear" else "shear strain [-]")
            ax.set_ylabel("stress [Pa]")
            ymax = max(float(np.max(y_true_plot)), float(np.max(y_pred_plot)), 1.0)
            ymin = min(float(np.min(y_true_plot)), float(np.min(y_pred_plot)), 0.0)
            pad = 0.08 * max(ymax - ymin, 1.0)
            ax.set_ylim(ymin - pad, ymax + pad)
            if loading == "compression":
                ax.invert_xaxis()
            ax.grid(alpha=0.15)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="gray", label="data")
    ]
    legend_handles.extend([mpatches.Patch(facecolor=label_to_color[label], edgecolor="none", label=label) for label in ordered_labels])
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"Invariant discovery | {COLOR_LABELS[color]} | {TIME_LABELS[time_days]}", fontsize=18)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(save_path, dpi=240)
    plt.close(fig)
    return rows


def main() -> None:
    args = parse_args()
    cfg = InvariantDiscoveryConfig(ridge_alpha=args.ridge_alpha)
    paths = ProjectPaths(root=args.root.resolve())
    raw_df = load_all_workbooks(paths.data_dir, axial_stress_column=args.axial_stress_column, compression_sign=args.compression_sign)
    summary_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)

    out_dir = paths.output_dir / "invariant_discovery_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "robust_summary_curves.csv", index=False)
    interpolated_df.to_csv(out_dir / "interpolated_individual_curves.csv", index=False)
    diagnostics_df.to_csv(out_dir / "curve_diagnostics.csv", index=False)

    summary_rows: list[dict] = []
    for color in sorted(summary_df["color"].unique()):
        color_df = summary_df.loc[summary_df["color"] == color].copy()
        trained_models = {}
        for variant, _ in TRAIN_VARIANTS:
            model = _fit_variant(color_df, variant, cfg)
            trained_models[variant] = model
            run_dir = out_dir / color / variant
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "weights.json").write_text(json.dumps(model.export_weights(), indent=2), encoding="utf-8")

        for time_days in sorted(color_df["time_days"].unique()):
            rows = _plot_time_panel(
                color=color,
                time_days=float(time_days),
                color_df=color_df,
                trained_models=trained_models,
                save_path=out_dir / color / f"invariant_discovery_{color}_day_{int(time_days)}.png",
            )
            summary_rows.extend(rows)

    pd.DataFrame(summary_rows).to_csv(out_dir / "invariant_discovery_r2_summary.csv", index=False)


if __name__ == "__main__":
    main()
