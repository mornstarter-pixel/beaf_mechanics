from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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

TERM_COLORS = {
    "I1-3": "#8B0000",
    "exp(I1-3)-1": "#C00000",
    "ln(1+(I1-3))": "#FF1F00",
    "(I1-3)^2": "#F57C00",
    "exp((I1-3)^2)-1": "#F4C430",
    "ln(1+((I1-3)^2))": "#8DB255",
    "I2-3": "#2CC7B8",
    "exp(I2-3)-1": "#00B4D8",
    "ln(1+(I2-3))": "#1E88E5",
    "(I2-3)^2": "#4F6BD7",
    "exp((I2-3)^2)-1": "#6A1B9A",
    "ln(1+((I2-3)^2))": "#9C27B0",
}


@dataclass
class Fig1CompShearConfig:
    root: Path
    axial_stress_column: str = "stress"
    compression_sign: float = -1.0
    ridge_alpha: float = 1e-6
    scale_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    shear_weight_grid: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    objective_mix: float = 0.2
    shear_monotonic_penalty: float = 3.0
    shear_end_penalty: float = 1.0
    reference_lambda_points: tuple[float, ...] = (0.95, 0.90, 0.85)
    reference_gamma_points: tuple[float, ...] = (0.05, 0.10, 0.15)


class Fig1JointModel:
    def __init__(self, params: tuple[float, float, float, float], coeffs: np.ndarray) -> None:
        self.params = tuple(float(v) for v in params)
        self.coeffs = np.asarray(coeffs, dtype=float).reshape(-1)

    @staticmethod
    def _feature_derivatives(i1b: np.ndarray, i2b: np.ndarray, params: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        a1, a2, b1, b2 = params
        d_i1 = np.column_stack(
            [
                np.ones_like(i1b),
                a1 * np.exp(np.clip(a1 * i1b, 0.0, None)),
                b1 / np.maximum(1.0 + np.clip(b1 * i1b, 0.0, None), 1e-8),
                2.0 * i1b,
                2.0 * a2 * i1b * np.exp(np.clip(a2 * np.square(i1b), 0.0, None)),
                (2.0 * b2 * i1b) / np.maximum(1.0 + b2 * np.square(i1b), 1e-8),
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
                b1 / np.maximum(1.0 + np.clip(b1 * i2b, 0.0, None), 1e-8),
                2.0 * i2b,
                2.0 * a2 * i2b * np.exp(np.clip(a2 * np.square(i2b), 0.0, None)),
                (2.0 * b2 * i2b) / np.maximum(1.0 + b2 * np.square(i2b), 1e-8),
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
        y_out = np.empty_like(y_sorted)
        y_out[order] = y_sorted
        return y_out

    def predict_raw(self, deformation: np.ndarray, loading: str) -> np.ndarray:
        return self.stress_design_matrix(deformation, loading) @ self.coeffs

    def predict(self, deformation: np.ndarray, loading: str) -> np.ndarray:
        pred = self.predict_raw(deformation, loading)
        if loading == "shear":
            return self._monotone_project(np.asarray(deformation, dtype=float), pred)
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
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fig.1-style invariant joint compression+shear time analysis.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    return parser.parse_args()


def _fit_day_joint_model(day_df: pd.DataFrame, cfg: Fig1CompShearConfig) -> tuple[Fig1JointModel, dict[str, float]]:
    comp_df = day_df.loc[day_df["loading"] == "compression"].sort_values("deformation").copy()
    shear_df = day_df.loc[day_df["loading"] == "shear"].sort_values("deformation").copy()
    best: tuple[float, tuple[float, float, float, float], float, np.ndarray, float, float] | None = None

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
                    for shear_weight in cfg.shear_weight_grid:
                        X = np.vstack([Xc, np.sqrt(shear_weight) * Xs])
                        y = np.concatenate([yc, np.sqrt(shear_weight) * ys])
                        solver = Ridge(alpha=cfg.ridge_alpha, fit_intercept=False)
                        solver.fit(X, y)
                        coeffs = solver.coef_.astype(float)
                        pred_c = Xc @ coeffs
                        pred_s = Fig1JointModel(params=params, coeffs=coeffs).predict(shear_df["deformation"].to_numpy(), "shear")
                        r2_c = r2_score(yc, pred_c)
                        r2_s = r2_score(ys, pred_s)
                        # Penalize non-monotonic shear predictions; the experimental shear response should
                        # not develop an artificial mid-range dip.
                        dy = np.diff(pred_s)
                        shear_scale = max(float(np.max(np.abs(ys))), 1.0)
                        monotonic_violation = float(np.sum(np.abs(np.minimum(dy, 0.0))) / shear_scale)
                        end_shortfall = float(max(np.max(pred_s) - pred_s[-1], 0.0) / shear_scale)
                        objective = (
                            min(r2_c, r2_s)
                            + cfg.objective_mix * (r2_c + r2_s)
                            - cfg.shear_monotonic_penalty * monotonic_violation
                            - cfg.shear_end_penalty * end_shortfall
                        )
                        candidate = (objective, params, shear_weight, coeffs, r2_c, r2_s)
                        if best is None or candidate[0] > best[0]:
                            best = candidate

    assert best is not None
    _, params, shear_weight, coeffs, r2_c, r2_s = best
    model = Fig1JointModel(params=params, coeffs=coeffs)
    metrics = {
        "compression_r2": float(r2_c),
        "shear_r2": float(r2_s),
        "shear_weight": float(shear_weight),
        "exp_scale_linear": float(params[0]),
        "exp_scale_quadratic": float(params[1]),
        "log_scale_linear": float(params[2]),
        "log_scale_quadratic": float(params[3]),
    }
    return model, metrics


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
    )
    paths = ProjectPaths(root=cfg.root)
    raw_df = load_all_workbooks(paths.data_dir, axial_stress_column=cfg.axial_stress_column, compression_sign=cfg.compression_sign)
    summary_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)
    summary_df = summary_df.loc[summary_df["loading"].isin(["compression", "shear"])].copy()

    out_dir = paths.output_dir / "fig1_comp_shear_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "robust_summary_curves.csv", index=False)
    interpolated_df.to_csv(out_dir / "interpolated_individual_curves.csv", index=False)
    diagnostics_df.to_csv(out_dir / "curve_diagnostics.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    all_metrics = []
    for color in sorted(summary_df["color"].unique()):
        color_df = summary_df.loc[summary_df["color"] == color].copy()
        models_by_day: dict[float, Fig1JointModel] = {}
        metrics_rows = []
        color_dir = out_dir / color
        color_dir.mkdir(parents=True, exist_ok=True)

        for day in sorted(color_df["time_days"].unique()):
            day_df = color_df.loc[color_df["time_days"] == day].copy()
            model, meta = _fit_day_joint_model(day_df, cfg)
            models_by_day[float(day)] = model
            for loading in ["compression", "shear"]:
                panel_df = day_df.loc[day_df["loading"] == loading].copy()
                pred = model.predict(panel_df["deformation"].to_numpy(), loading)
                score = r2_score(panel_df["stress_pa"], pred)
                metrics_rows.append({"color": color, "time_days": float(day), "loading": loading, "r2": float(score), **meta})
            (color_dir / f"day_{int(day)}_model.json").write_text(json.dumps(model.export(), indent=2), encoding="utf-8")

        metrics_df = pd.DataFrame(metrics_rows)
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
