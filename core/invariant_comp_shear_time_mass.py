from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from core.data_pipeline import load_all_workbooks, preprocess_robust_curves
from core.invariant_baseline import COLOR_LABELS, TIME_LABELS
from core.settings import ProjectPaths


TERM_LABELS = [
    "I1-3",
    "exp(I1-3)-1",
    "ln(1+(I1-3))",
    "(I1-3)^2",
    "exp((I1-3)^2)-1",
    "ln(1+((I1-3)^2))",
    "I2-3",
    "exp(I2-3)-1",
    "ln(1+(I2-3))",
    "(I2-3)^2",
    "exp((I2-3)^2)-1",
    "ln(1+((I2-3)^2))",
]

VARIANT_LABELS = {
    "time_only": "time only",
    "mass_only": "mass only",
    "time_mass": "time + mass",
}


@dataclass
class TimeMassConfig:
    root: Path
    axial_stress_column: str = "stress"
    compression_sign: float = -1.0
    ridge_alpha: float = 1e-4
    mass_csv: str = "quality_mass_g.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invariant compression+shear model with time and mass covariates.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    parser.add_argument("--ridge-alpha", type=float, default=1e-4)
    return parser.parse_args()


def load_mass_table(csv_path: Path) -> pd.DataFrame:
    mass_df = pd.read_csv(csv_path)
    mass_df["color"] = mass_df["color"].astype(str).str.lower()
    mass_df["freshness_index"] = mass_df["freshness_index"].astype(int)
    mass_df["sample_num"] = mass_df["sample_num"].astype(int)
    mass_df["mass_g"] = mass_df["mass_g"].astype(float)
    return mass_df[["color", "freshness_index", "sample_num", "mass_g"]]


def merge_mass_data(interpolated_df: pd.DataFrame, summary_df: pd.DataFrame, mass_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_interpolated = interpolated_df.merge(
        mass_df,
        on=["color", "freshness_index", "sample_num"],
        how="left",
        validate="many_to_one",
    )
    if merged_interpolated["mass_g"].isna().any():
        missing = merged_interpolated.loc[merged_interpolated["mass_g"].isna(), ["color", "freshness_index", "sample_num"]].drop_duplicates()
        raise ValueError(f"Missing mass values for rows: {missing.to_dict('records')}")

    mass_group = (
        mass_df.groupby(["color", "freshness_index"], as_index=False)
        .agg(mass_mean_g=("mass_g", "mean"), mass_median_g=("mass_g", "median"), mass_std_g=("mass_g", "std"))
    )
    merged_summary = summary_df.merge(
        mass_group,
        on=["color", "freshness_index"],
        how="left",
        validate="many_to_one",
    )
    return merged_interpolated, merged_summary


class InvariantCovariateModel:
    def __init__(self, coeffs: np.ndarray, variant: str) -> None:
        self.coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
        self.variant = variant

    @staticmethod
    def _smooth_monotone_project(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        y_iso = iso.fit_transform(x_sorted, y_sorted)
        if len(y_iso) >= 5:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
            kernel /= kernel.sum()
            padded = np.pad(y_iso, (2, 2), mode="edge")
            y_smooth = np.convolve(padded, kernel, mode="valid")
        else:
            y_smooth = y_iso
        y_mono = np.maximum.accumulate(y_smooth)
        out = np.empty_like(y_mono)
        out[order] = y_mono
        return out

    @staticmethod
    def stress_design_matrix(deformation: np.ndarray, loading: str) -> np.ndarray:
        x = np.asarray(deformation, dtype=float).reshape(-1)
        if loading == "shear":
            gamma = x
            i1b = np.square(gamma)
            i2b = np.square(gamma)
            return InvariantCovariateModel._build_shear_terms(gamma, i1b, i2b)
        lam = x
        i1b = np.square(lam) + 2.0 / lam - 3.0
        i2b = 2.0 * lam + 1.0 / np.square(lam) - 3.0
        return InvariantCovariateModel._build_uniaxial_terms(lam, i1b, i2b)

    @staticmethod
    def _feature_derivatives(i1b: np.ndarray, i2b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d_i1 = np.column_stack(
            [
                np.ones_like(i1b),
                np.exp(np.clip(i1b, 0.0, None)),
                1.0 / np.maximum(1.0 + np.clip(i1b, 0.0, None), 1e-8),
                2.0 * i1b,
                2.0 * i1b * np.exp(np.clip(np.square(i1b), 0.0, None)),
                (2.0 * i1b) / np.maximum(1.0 + np.square(i1b), 1e-8),
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
                np.exp(np.clip(i2b, 0.0, None)),
                1.0 / np.maximum(1.0 + np.clip(i2b, 0.0, None), 1e-8),
                2.0 * i2b,
                2.0 * i2b * np.exp(np.clip(np.square(i2b), 0.0, None)),
                (2.0 * i2b) / np.maximum(1.0 + np.square(i2b), 1e-8),
            ]
        )
        return d_i1, d_i2

    @classmethod
    def _build_uniaxial_terms(cls, lam: np.ndarray, i1b: np.ndarray, i2b: np.ndarray) -> np.ndarray:
        d_i1, d_i2 = cls._feature_derivatives(i1b, i2b)
        return 2.0 * (d_i1 * lam.reshape(-1, 1) + d_i2) - 2.0 * (
            d_i1 / np.square(lam).reshape(-1, 1) + d_i2 / np.power(lam, 3.0).reshape(-1, 1)
        )

    @classmethod
    def _build_shear_terms(cls, gamma: np.ndarray, i1b: np.ndarray, i2b: np.ndarray) -> np.ndarray:
        d_i1, d_i2 = cls._feature_derivatives(i1b, i2b)
        return 2.0 * gamma.reshape(-1, 1) * (d_i1 + d_i2)

    def _covariate_matrix(self, time_days: np.ndarray, mass_norm: np.ndarray) -> np.ndarray:
        t = np.asarray(time_days, dtype=float).reshape(-1)
        m = np.asarray(mass_norm, dtype=float).reshape(-1)
        if self.variant == "time_only":
            return np.column_stack([t == 0.0, t == 1.0, t == 2.0]).astype(float)
        if self.variant == "mass_only":
            return np.column_stack([np.ones_like(m), m])
        if self.variant == "time_mass":
            return np.column_stack([t == 0.0, t == 1.0, t == 2.0, m]).astype(float)
        raise ValueError(f"Unknown variant: {self.variant}")

    def design_matrix(self, deformation: np.ndarray, loading: str, time_days: np.ndarray, mass_norm: np.ndarray) -> np.ndarray:
        base = self.stress_design_matrix(deformation, loading)
        cov = self._covariate_matrix(time_days, mass_norm)
        pieces = [base * cov[:, [idx]] for idx in range(cov.shape[1])]
        return np.concatenate(pieces, axis=1)

    def predict_raw(self, deformation: np.ndarray, loading: str, time_days: np.ndarray, mass_norm: np.ndarray) -> np.ndarray:
        return self.design_matrix(deformation, loading, time_days, mass_norm) @ self.coeffs

    def predict(self, deformation: np.ndarray, loading: str, time_days: np.ndarray, mass_norm: np.ndarray) -> np.ndarray:
        pred = self.predict_raw(deformation, loading, time_days, mass_norm)
        if loading == "shear":
            return self._smooth_monotone_project(np.asarray(deformation, dtype=float), pred)
        return pred

    def export(self) -> dict[str, object]:
        return {"variant": self.variant, "term_labels": TERM_LABELS, "coeffs": self.coeffs.tolist()}


def _fit_variant(color_df: pd.DataFrame, variant: str, ridge_alpha: float) -> InvariantCovariateModel:
    model = InvariantCovariateModel(coeffs=np.zeros(1), variant=variant)
    color_df = color_df.sort_values(["loading", "time_days", "sample_num", "point_index"]).copy()
    X = model.design_matrix(
        deformation=color_df["deformation"].to_numpy(),
        loading=color_df["loading"].to_numpy()[0] if color_df["loading"].nunique() == 1 else color_df["loading"].to_numpy(),
        time_days=color_df["time_days"].to_numpy(),
        mass_norm=color_df["mass_norm"].to_numpy(),
    )
    # design_matrix currently expects one loading label; build blockwise below instead
    raise RuntimeError("internal")


def _build_variant_design(frame: pd.DataFrame, variant: str) -> np.ndarray:
    model = InvariantCovariateModel(coeffs=np.zeros(1), variant=variant)
    rows = []
    for loading, group in frame.groupby("loading", sort=False):
        rows.append(
            model.design_matrix(
                deformation=group["deformation"].to_numpy(),
                loading=loading,
                time_days=group["time_days"].to_numpy(),
                mass_norm=group["mass_norm"].to_numpy(),
            )
        )
    return np.vstack(rows)


def fit_variant_model(sample_df: pd.DataFrame, color: str, variant: str, ridge_alpha: float) -> InvariantCovariateModel:
    frame = sample_df.loc[(sample_df["color"] == color) & (sample_df["loading"].isin(["compression", "shear"]))].copy()
    frame = frame.sort_values(["loading", "time_days", "sample_num", "point_index"]).reset_index(drop=True)
    X_blocks = []
    y_blocks = []
    w_blocks = []
    model = InvariantCovariateModel(coeffs=np.zeros(1), variant=variant)
    for loading, group in frame.groupby("loading", sort=False):
        X = model.design_matrix(
            deformation=group["deformation"].to_numpy(),
            loading=loading,
            time_days=group["time_days"].to_numpy(),
            mass_norm=group["mass_norm"].to_numpy(),
        )
        y = group["stress_pa"].to_numpy(dtype=float)
        curve_w = group.get("curve_weight", pd.Series(np.ones(len(group)))).to_numpy(dtype=float)
        scale = max(np.max(np.abs(y)), 1.0)
        balance_w = 1.0 / scale
        X_blocks.append(X)
        y_blocks.append(y)
        w_blocks.append(curve_w * balance_w)
    X_all = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)
    w_all = np.concatenate(w_blocks)
    x_scale = np.maximum(np.std(X_all, axis=0), 1e-8)
    X_std = X_all / x_scale
    sqrt_w = np.sqrt(np.clip(w_all, 1e-8, None))[:, None]
    solver = Ridge(alpha=ridge_alpha, fit_intercept=False)
    solver.fit(X_std * sqrt_w, y_all * sqrt_w[:, 0])
    coeffs = solver.coef_ / x_scale
    return InvariantCovariateModel(coeffs=coeffs, variant=variant)


def evaluate_variant(model: InvariantCovariateModel, sample_df: pd.DataFrame, summary_df: pd.DataFrame, color: str, variant: str) -> pd.DataFrame:
    rows = []
    for day in sorted(summary_df.loc[summary_df["color"] == color, "time_days"].unique()):
        for loading in ["compression", "shear"]:
            sample_group = sample_df.loc[
                (sample_df["color"] == color) & (sample_df["time_days"] == day) & (sample_df["loading"] == loading)
            ].copy()
            pred_sample = model.predict(
                deformation=sample_group["deformation"].to_numpy(),
                loading=loading,
                time_days=sample_group["time_days"].to_numpy(),
                mass_norm=sample_group["mass_norm"].to_numpy(),
            )
            sample_r2 = r2_score(sample_group["stress_pa"], pred_sample)

            summary_group = summary_df.loc[
                (summary_df["color"] == color) & (summary_df["time_days"] == day) & (summary_df["loading"] == loading)
            ].copy()
            summary_mass = summary_group["mass_median_g"].to_numpy(dtype=float)
            pred_summary = model.predict(
                deformation=summary_group["deformation"].to_numpy(),
                loading=loading,
                time_days=summary_group["time_days"].to_numpy(),
                mass_norm=summary_mass,
            )
            summary_r2 = r2_score(summary_group["stress_pa"], pred_summary)
            rows.append(
                {
                    "color": color,
                    "variant": variant,
                    "time_days": float(day),
                    "loading": loading,
                    "sample_r2": float(sample_r2),
                    "summary_r2": float(summary_r2),
                }
            )
    return pd.DataFrame(rows)


def _plot_variant(color: str, variant: str, model: InvariantCovariateModel, summary_df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    color_df = summary_df.loc[(summary_df["color"] == color) & (summary_df["loading"].isin(["compression", "shear"]))].copy()
    for col_idx, day in enumerate(sorted(color_df["time_days"].unique())):
        day_df = color_df.loc[color_df["time_days"] == day].copy()
        for row_idx, loading in enumerate(["compression", "shear"]):
            ax = axes[row_idx, col_idx]
            group = day_df.loc[day_df["loading"] == loading].sort_values("deformation").copy()
            mass_norm = group["mass_median_g"].to_numpy(dtype=float)
            pred = model.predict(
                deformation=group["deformation"].to_numpy(),
                loading=loading,
                time_days=group["time_days"].to_numpy(),
                mass_norm=mass_norm,
            )
            y_true = group["stress_pa"].to_numpy(dtype=float)
            y_plot = np.abs(y_true) if loading == "compression" else y_true
            p_plot = np.abs(pred) if loading == "compression" else pred
            ax.plot(group["deformation"], p_plot, color="black", linewidth=1.3)
            ax.scatter(group["deformation"], y_plot, s=26, facecolors="white", edgecolors="gray", linewidth=0.8)
            ax.fill_between(
                group["deformation"],
                np.abs(group["stress_q25_pa"]) if loading == "compression" else group["stress_q25_pa"],
                np.abs(group["stress_q75_pa"]) if loading == "compression" else group["stress_q75_pa"],
                color="#8ecae6",
                alpha=0.3,
                linewidth=0.0,
            )
            ax.set_title(f"{TIME_LABELS[float(day)]} | {loading}")
            ax.text(0.03, 0.90, f"R$^2$={r2_score(y_true, pred):.4f}", transform=ax.transAxes, fontsize=10)
            ax.set_xlabel("stretch [-]" if loading == "compression" else "shear strain [-]")
            ax.set_ylabel("stress [Pa]")
            if loading == "compression":
                ax.invert_xaxis()
            ax.grid(alpha=0.15)
    fig.suptitle(f"Invariant comp+shear with {VARIANT_LABELS[variant]} | {COLOR_LABELS[color]}", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def _plot_mass_effects(color: str, variant: str, model: InvariantCovariateModel, sample_df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    color_df = sample_df.loc[(sample_df["color"] == color) & (sample_df["loading"].isin(["compression", "shear"]))].copy()
    for ax, loading, deformation_label, deformation_value in [
        (axes[0], "compression", "stretch", 0.90),
        (axes[1], "shear", "gamma", 0.10),
    ]:
        for day in sorted(color_df["time_days"].unique()):
            masses = np.linspace(color_df["mass_g"].min(), color_df["mass_g"].max(), 40)
            mass_norm = (masses - color_df["mass_g"].mean()) / max(color_df["mass_g"].std(ddof=0), 1e-8)
            pred = model.predict(
                deformation=np.full_like(masses, deformation_value, dtype=float),
                loading=loading,
                time_days=np.full_like(masses, day, dtype=float),
                mass_norm=mass_norm,
            )
            pred_plot = np.abs(pred) if loading == "compression" else pred
            ax.plot(masses, pred_plot, linewidth=1.4, label=TIME_LABELS[float(day)])
        ax.set_title(f"{loading} | fixed {deformation_label}={deformation_value}")
        ax.set_xlabel("mass [g]")
        ax.set_ylabel("predicted stress [Pa]")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(f"Mass effect | {COLOR_LABELS[color]} | {VARIANT_LABELS[variant]}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = TimeMassConfig(
        root=args.root.resolve(),
        axial_stress_column=args.axial_stress_column,
        compression_sign=args.compression_sign,
        ridge_alpha=args.ridge_alpha,
    )
    paths = ProjectPaths(root=cfg.root)
    raw_df = load_all_workbooks(paths.data_dir, axial_stress_column=cfg.axial_stress_column, compression_sign=cfg.compression_sign)
    summary_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)
    mass_df = load_mass_table(paths.data_dir / cfg.mass_csv)
    interpolated_df, summary_df = merge_mass_data(interpolated_df, summary_df, mass_df)
    interpolated_df = interpolated_df.loc[
        interpolated_df["loading"].isin(["compression", "shear"]) & (~interpolated_df["is_curve_outlier"])
    ].copy()
    mass_mean = float(interpolated_df["mass_g"].mean())
    mass_std = float(interpolated_df["mass_g"].std(ddof=0))
    mass_std = mass_std if mass_std > 0 else 1.0
    interpolated_df["mass_norm"] = (interpolated_df["mass_g"] - mass_mean) / mass_std
    summary_df["mass_median_g"] = summary_df["mass_median_g"].fillna(summary_df["mass_mean_g"])
    summary_df["mass_norm"] = (summary_df["mass_median_g"] - mass_mean) / mass_std

    out_dir = paths.output_dir / "fig1_comp_shear_time_mass"
    out_dir.mkdir(parents=True, exist_ok=True)
    interpolated_df.to_csv(out_dir / "interpolated_individual_curves_with_mass.csv", index=False)
    summary_df.to_csv(out_dir / "robust_summary_curves_with_mass.csv", index=False)
    diagnostics_df.to_csv(out_dir / "curve_diagnostics.csv", index=False)
    mass_df.to_csv(out_dir / "quality_mass_g.csv", index=False)

    metrics_rows = []
    for color in sorted(summary_df["color"].unique()):
        color_dir = out_dir / color
        color_dir.mkdir(parents=True, exist_ok=True)
        for variant in ["time_only", "mass_only", "time_mass"]:
            model = fit_variant_model(interpolated_df, color, variant, ridge_alpha=cfg.ridge_alpha)
            (color_dir / f"{variant}_model.json").write_text(json.dumps(model.export(), indent=2), encoding="utf-8")
            metrics_df = evaluate_variant(model, interpolated_df, summary_df, color, variant)
            metrics_rows.append(metrics_df)
            _plot_variant(color, variant, model, summary_df, color_dir / f"{variant}_three_days.png")
            _plot_mass_effects(color, variant, model, interpolated_df, color_dir / f"{variant}_mass_effects.png")

    pd.concat(metrics_rows, ignore_index=True).to_csv(out_dir / "variant_metrics.csv", index=False)


if __name__ == "__main__":
    main()
