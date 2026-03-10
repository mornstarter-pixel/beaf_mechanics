from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from core.data_pipeline import load_all_workbooks, preprocess_robust_curves
from core.invariant_baseline import COLOR_LABELS, TIME_LABELS
from core.invariant_comp_shear_time_mass import (
    InvariantCovariateModel,
    load_mass_table,
    merge_mass_data,
)
from core.settings import ProjectPaths


LOADING_ORDER = ["compression", "shear"]
DAY_ORDER = [0.0, 1.0, 2.0]


@dataclass
class MultiRoundConfig:
    root: Path
    axial_stress_column: str = "stress"
    compression_sign: float = -1.0
    ridge_alpha: float = 1e-4
    max_triplets_per_color: int = 5
    good_r2_threshold: float = 0.85
    merge_group_count: int = 3
    mass_csv: str = "quality_mass_g.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-round invariant CANN testing.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    parser.add_argument("--ridge-alpha", type=float, default=1e-4)
    parser.add_argument("--max-triplets-per-color", type=int, default=5)
    parser.add_argument("--good-r2-threshold", type=float, default=0.85)
    parser.add_argument("--merge-group-count", type=int, default=3)
    return parser.parse_args()


def _prepare_merged_inputs(cfg: MultiRoundConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = ProjectPaths(root=cfg.root)
    raw_df = load_all_workbooks(
        paths.data_dir,
        axial_stress_column=cfg.axial_stress_column,
        compression_sign=cfg.compression_sign,
    )
    summary_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)
    mass_df = load_mass_table(paths.data_dir / cfg.mass_csv)
    interpolated_df, summary_df = merge_mass_data(interpolated_df, summary_df, mass_df)

    sample_df = interpolated_df.loc[
        interpolated_df["loading"].isin(LOADING_ORDER) & (~interpolated_df["is_curve_outlier"])
    ].copy()
    summary_df = summary_df.loc[summary_df["loading"].isin(LOADING_ORDER)].copy()

    mass_mean = float(sample_df["mass_g"].mean())
    mass_std = float(sample_df["mass_g"].std(ddof=0))
    mass_std = mass_std if mass_std > 0 else 1.0
    sample_df["mass_norm"] = (sample_df["mass_g"] - mass_mean) / mass_std
    summary_df["mass_median_g"] = summary_df["mass_median_g"].fillna(summary_df["mass_mean_g"])
    summary_df["mass_norm"] = (summary_df["mass_median_g"] - mass_mean) / mass_std
    return raw_df, summary_df, sample_df, diagnostics_df


def _aggregate_selected_subset(sample_subset: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "color",
        "cut_name",
        "freshness_index",
        "time_hours",
        "time_days",
        "mode",
        "loading",
        "point_index",
    ]
    rows: list[dict[str, object]] = []
    for key, group in sample_subset.groupby(group_cols, sort=True):
        color, cut_name, freshness_index, time_hours, time_days, mode, loading, point_index = key
        rows.append(
            {
                "file_name": "selected_subset",
                "color": color,
                "cut_name": cut_name,
                "freshness_index": freshness_index,
                "time_hours": time_hours,
                "time_days": time_days,
                "sample_num": 0,
                "mode": mode,
                "loading": loading,
                "point_index": int(point_index),
                "deformation": float(np.median(group["deformation"])),
                "stress_pa": float(np.median(group["stress_pa"])),
                "stress_q25_pa": float(np.quantile(group["stress_pa"], 0.25)),
                "stress_q75_pa": float(np.quantile(group["stress_pa"], 0.75)),
                "mass_median_g": float(np.median(group["mass_g"])),
                "mass_mean_g": float(np.mean(group["mass_g"])),
                "mass_norm": float(np.median(group["mass_norm"])),
                "n_used": int(group["sample_num"].nunique()),
                "n_total": int(group["sample_num"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(group_cols).reset_index(drop=True)


def _curve_signature(group: pd.DataFrame, loading: str, n_grid: int = 60) -> tuple[np.ndarray, float]:
    work = group.sort_values("deformation").copy()
    if work.empty:
        return np.zeros(n_grid, dtype=float), 0.0
    x = work["deformation"].to_numpy(dtype=float)
    y = work["stress_pa"].to_numpy(dtype=float)
    if loading == "compression":
        y = np.abs(y)
    x_norm = np.linspace(0.0, 1.0, n_grid)
    if np.isclose(x.max(), x.min()):
        sig = np.repeat(y.mean(), n_grid)
    else:
        x_src = (x - x.min()) / (x.max() - x.min())
        sig = np.interp(x_norm, x_src, y)
    peak = float(np.max(y))
    scale = max(peak, 1.0)
    sig = sig / scale
    return sig, peak


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _triplet_score(sample_df: pd.DataFrame, color: str, files_by_day: dict[float, str]) -> dict[str, object]:
    score_parts = []
    peak_penalty = 0.0
    trend_penalty = 0.0
    loading_details: dict[str, dict[str, float]] = {}
    for loading in LOADING_ORDER:
        sigs = []
        peaks = []
        for day in DAY_ORDER:
            group = sample_df.loc[
                (sample_df["color"] == color)
                & (sample_df["time_days"] == day)
                & (sample_df["loading"] == loading)
                & (sample_df["file_name"] == files_by_day[day])
            ].copy()
            sig, peak = _curve_signature(group, loading)
            sigs.append(sig)
            peaks.append(peak)
        pairwise = [
            _cosine_similarity(sigs[0], sigs[1]),
            _cosine_similarity(sigs[0], sigs[2]),
            _cosine_similarity(sigs[1], sigs[2]),
        ]
        mean_similarity = float(np.mean(pairwise))
        peak_cv = float(np.std(peaks) / max(np.mean(peaks), 1e-8))
        if peaks[1] > peaks[0] * 1.05:
            trend_penalty += (peaks[1] / max(peaks[0], 1e-8)) - 1.05
        if peaks[2] > max(peaks[0], peaks[1]) * 1.05:
            trend_penalty += (peaks[2] / max(max(peaks[0], peaks[1]), 1e-8)) - 1.05
        score_parts.append(mean_similarity)
        peak_penalty += peak_cv
        loading_details[loading] = {"shape_similarity": mean_similarity, "peak_cv": peak_cv}
    total_score = float(np.mean(score_parts) - 0.35 * peak_penalty - 0.5 * trend_penalty)
    return {
        "color": color,
        "files_by_day": files_by_day,
        "score": total_score,
        "trend_penalty": float(trend_penalty),
        "peak_penalty": float(peak_penalty),
        "compression_shape_similarity": loading_details["compression"]["shape_similarity"],
        "compression_peak_cv": loading_details["compression"]["peak_cv"],
        "shear_shape_similarity": loading_details["shear"]["shape_similarity"],
        "shear_peak_cv": loading_details["shear"]["peak_cv"],
    }


def _select_triplets(sample_df: pd.DataFrame, color: str, max_triplets: int) -> pd.DataFrame:
    file_table = (
        sample_df.loc[sample_df["color"] == color, ["file_name", "time_days", "loading"]]
        .drop_duplicates()
        .groupby(["file_name", "time_days"], as_index=False)
        .agg(n_loadings=("loading", "nunique"))
    )
    valid_files = file_table.loc[file_table["n_loadings"] == len(LOADING_ORDER)].copy()
    by_day = {day: valid_files.loc[valid_files["time_days"] == day, "file_name"].tolist() for day in DAY_ORDER}
    if any(len(by_day[day]) == 0 for day in DAY_ORDER):
        return pd.DataFrame()
    candidates = []
    for combo in itertools.product(by_day[0.0], by_day[1.0], by_day[2.0]):
        files_by_day = {0.0: combo[0], 1.0: combo[1], 2.0: combo[2]}
        candidates.append(_triplet_score(sample_df, color, files_by_day))
    cand_df = pd.DataFrame(candidates).sort_values("score", ascending=False).reset_index(drop=True)
    used_by_day: dict[float, set[str]] = {day: set() for day in DAY_ORDER}
    selected_rows = []
    for _, row in cand_df.iterrows():
        files_by_day = row["files_by_day"]
        if any(files_by_day[day] in used_by_day[day] for day in DAY_ORDER):
            continue
        selected_rows.append(row.to_dict())
        for day in DAY_ORDER:
            used_by_day[day].add(files_by_day[day])
        if len(selected_rows) >= max_triplets:
            break
    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        return selected_df
    selected_df["group_id"] = [f"{color}_triplet_{idx+1:02d}" for idx in range(len(selected_df))]
    return selected_df


def _fit_loading_model(frame: pd.DataFrame, loading: str, variant: str, ridge_alpha: float) -> InvariantCovariateModel:
    subset = frame.loc[frame["loading"] == loading].sort_values(["time_days", "sample_num", "point_index"]).copy()
    model = InvariantCovariateModel(coeffs=np.zeros(1), variant=variant)
    X = model.design_matrix(
        deformation=subset["deformation"].to_numpy(),
        loading=loading,
        time_days=subset["time_days"].to_numpy(),
        mass_norm=subset["mass_norm"].to_numpy(),
    )
    y = subset["stress_pa"].to_numpy(dtype=float)
    weights = subset.get("curve_weight", pd.Series(np.ones(len(subset), dtype=float))).to_numpy(dtype=float)
    scale = max(float(np.max(np.abs(y))), 1.0)
    x_scale = np.maximum(np.std(X, axis=0), 1e-8)
    X_std = X / x_scale
    sqrt_w = np.sqrt(np.clip(weights / scale, 1e-8, None))[:, None]
    gram = (X_std * sqrt_w).T @ (X_std * sqrt_w) + ridge_alpha * np.eye(X_std.shape[1])
    rhs = (X_std * sqrt_w).T @ (y * sqrt_w[:, 0])
    coeffs_std = np.linalg.solve(gram, rhs)
    coeffs = coeffs_std / x_scale
    return InvariantCovariateModel(coeffs=coeffs, variant=variant)


def _fit_joint_model(frame: pd.DataFrame, variant: str, ridge_alpha: float) -> InvariantCovariateModel:
    subset = frame.loc[frame["loading"].isin(LOADING_ORDER)].sort_values(
        ["loading", "time_days", "sample_num", "point_index"]
    ).copy()
    model = InvariantCovariateModel(coeffs=np.zeros(1), variant=variant)
    X_blocks = []
    y_blocks = []
    w_blocks = []
    for loading in LOADING_ORDER:
        group = subset.loc[subset["loading"] == loading].copy()
        X_blocks.append(
            model.design_matrix(
                deformation=group["deformation"].to_numpy(),
                loading=loading,
                time_days=group["time_days"].to_numpy(),
                mass_norm=group["mass_norm"].to_numpy(),
            )
        )
        y = group["stress_pa"].to_numpy(dtype=float)
        y_blocks.append(y)
        curve_w = group.get("curve_weight", pd.Series(np.ones(len(group), dtype=float))).to_numpy(dtype=float)
        scale = max(float(np.max(np.abs(y))), 1.0)
        w_blocks.append(curve_w / scale)
    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    weights = np.concatenate(w_blocks)
    x_scale = np.maximum(np.std(X, axis=0), 1e-8)
    X_std = X / x_scale
    sqrt_w = np.sqrt(np.clip(weights, 1e-8, None))[:, None]
    gram = (X_std * sqrt_w).T @ (X_std * sqrt_w) + ridge_alpha * np.eye(X_std.shape[1])
    rhs = (X_std * sqrt_w).T @ (y * sqrt_w[:, 0])
    coeffs_std = np.linalg.solve(gram, rhs)
    coeffs = coeffs_std / x_scale
    return InvariantCovariateModel(coeffs=coeffs, variant=variant)


def _evaluate_model(
    model: InvariantCovariateModel,
    summary_frame: pd.DataFrame,
    loadings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows = []
    metric_rows = []
    for loading in loadings:
        for day in DAY_ORDER:
            group = summary_frame.loc[
                (summary_frame["loading"] == loading) & (summary_frame["time_days"] == day)
            ].sort_values("deformation").copy()
            if group.empty:
                continue
            pred = model.predict(
                deformation=group["deformation"].to_numpy(),
                loading=loading,
                time_days=group["time_days"].to_numpy(),
                mass_norm=group["mass_norm"].to_numpy(),
            )
            pred_plot = np.abs(pred) if loading == "compression" else pred
            true_plot = (
                np.abs(group["stress_pa"].to_numpy(dtype=float))
                if loading == "compression"
                else group["stress_pa"].to_numpy(dtype=float)
            )
            for idx, (_, row) in enumerate(group.iterrows()):
                rec = row.to_dict()
                rec["predicted_stress_pa"] = float(pred[idx])
                rec["predicted_stress_plot_pa"] = float(pred_plot[idx])
                pred_rows.append(rec)
            metric_rows.append(
                {
                    "loading": loading,
                    "time_days": day,
                    "r2": float(r2_score(true_plot, pred_plot)),
                    "n_points": int(len(group)),
                }
            )
    return pd.DataFrame(pred_rows), pd.DataFrame(metric_rows)


def _fit_line(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return float("nan")
    return float(np.polyfit(xs, ys, deg=1)[0])


def _time_effect_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (loading, day), group in predictions_df.groupby(["loading", "time_days"], sort=True):
        g = group.sort_values("deformation").copy()
        x = g["deformation"].to_numpy(dtype=float)
        y = g["predicted_stress_plot_pa"].to_numpy(dtype=float)
        n_slope = max(3, int(round(0.15 * len(g))))
        rows.append(
            {
                "loading": loading,
                "time_days": float(day),
                "initial_modulus": _fit_line(x[:n_slope], y[:n_slope]),
                "peak_stress_pa": float(np.max(y)),
                "peak_deformation": float(x[np.argmax(y)]),
                "end_stress_pa": float(y[-1]),
                "area": float(np.trapezoid(y, x)),
                "mean_stress_pa": float(np.mean(y)),
            }
        )
    return pd.DataFrame(rows)


def _plot_time_effects(effects_df: pd.DataFrame, loadings: list[str], save_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [
        ("initial_modulus", "Initial modulus"),
        ("peak_stress_pa", "Peak stress [Pa]"),
        ("end_stress_pa", "End stress [Pa]"),
        ("area", "Area"),
    ]
    for ax, (metric, label) in zip(axes.flat, metrics):
        for loading in loadings:
            group = effects_df.loc[effects_df["loading"] == loading].sort_values("time_days")
            ax.plot(group["time_days"], group[metric], marker="o", label=loading)
        ax.set_title(label)
        ax.set_xlabel("time_days")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _plot_predictions(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    loadings: list[str],
    save_path: Path,
    title: str,
) -> None:
    n_rows = len(loadings)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.2 * n_rows))
    if n_rows == 1:
        axes = np.asarray([axes])
    for row_idx, loading in enumerate(loadings):
        for col_idx, day in enumerate(DAY_ORDER):
            ax = axes[row_idx, col_idx]
            group = predictions_df.loc[
                (predictions_df["loading"] == loading) & (predictions_df["time_days"] == day)
            ].sort_values("deformation").copy()
            if group.empty:
                ax.axis("off")
                continue
            x = group["deformation"].to_numpy(dtype=float)
            y_true = (
                np.abs(group["stress_pa"].to_numpy(dtype=float))
                if loading == "compression"
                else group["stress_pa"].to_numpy(dtype=float)
            )
            y_pred = group["predicted_stress_plot_pa"].to_numpy(dtype=float)
            q25 = (
                np.abs(group["stress_q25_pa"].to_numpy(dtype=float))
                if loading == "compression"
                else group["stress_q25_pa"].to_numpy(dtype=float)
            )
            q75 = (
                np.abs(group["stress_q75_pa"].to_numpy(dtype=float))
                if loading == "compression"
                else group["stress_q75_pa"].to_numpy(dtype=float)
            )
            ax.fill_between(x, q25, q75, color="#8ecae6", alpha=0.28, linewidth=0.0)
            ax.plot(x, y_pred, color="black", linewidth=1.4)
            ax.scatter(x, y_true, s=26, facecolors="white", edgecolors="gray", linewidth=0.8, zorder=5)
            r2 = metrics_df.loc[
                (metrics_df["loading"] == loading) & (metrics_df["time_days"] == day), "r2"
            ].iloc[0]
            ax.text(0.03, 0.90, f"R$^2$ = {r2:.4f}", transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{TIME_LABELS[day]} | {loading}")
            ax.set_xlabel("stretch [-]" if loading == "compression" else "shear strain [-]")
            ax.set_ylabel("stress [Pa]")
            if loading == "compression":
                ax.invert_xaxis()
            ax.grid(alpha=0.15)
    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def _write_processing_note(
    save_path: Path,
    title: str,
    description: list[str],
    files_by_day: dict[float, list[str]] | None = None,
    metrics_df: pd.DataFrame | None = None,
) -> None:
    lines = [title, ""]
    lines.extend(description)
    if files_by_day is not None:
        lines.extend(["", "Selected files by day:"])
        for day in DAY_ORDER:
            lines.append(f"- {TIME_LABELS[day]}: {', '.join(files_by_day.get(day, []))}")
    if metrics_df is not None and not metrics_df.empty:
        lines.extend(["", "R2 summary:"])
        for _, row in metrics_df.sort_values(["loading", "time_days"]).iterrows():
            lines.append(f"- {row['loading']} | {TIME_LABELS[row['time_days']]}: {row['r2']:.4f}")
    save_path.write_text("\n".join(lines), encoding="utf-8")


def _export_model(save_path: Path, model: InvariantCovariateModel) -> None:
    save_path.write_text(json.dumps(model.export(), indent=2), encoding="utf-8")


def _run_round_1(round_dir: Path, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for color in sorted(summary_df["color"].unique()):
        color_dir = round_dir / color
        color_dir.mkdir(parents=True, exist_ok=True)
        color_summary = summary_df.loc[(summary_df["color"] == color) & (summary_df["loading"].isin(LOADING_ORDER))].copy()
        color_summary["mass_norm"] = 0.0
        color_summary["curve_weight"] = 1.0
        for loading in LOADING_ORDER:
            loading_dir = color_dir / loading
            loading_dir.mkdir(parents=True, exist_ok=True)
            model = _fit_loading_model(color_summary, loading=loading, variant="time_only", ridge_alpha=1e-6)
            pred_df, metrics_df = _evaluate_model(model, color_summary, [loading])
            effects_df = _time_effect_table(pred_df)
            _export_model(loading_dir / "model.json", model)
            pred_df.to_csv(loading_dir / "predictions.csv", index=False)
            metrics_df.to_csv(loading_dir / "metrics.csv", index=False)
            effects_df.to_csv(loading_dir / "time_effects.csv", index=False)
            _plot_predictions(
                pred_df,
                metrics_df,
                [loading],
                loading_dir / f"{loading}_three_days.png",
                f"Round 1 | {COLOR_LABELS[color]} | {loading} | robust summary",
            )
            _plot_time_effects(
                effects_df,
                [loading],
                loading_dir / "time_effects.png",
                f"Round 1 time effects | {COLOR_LABELS[color]} | {loading}",
            )
            _write_processing_note(
                loading_dir / "processing_scheme.txt",
                f"Round 1 | {COLOR_LABELS[color]} | {loading}",
                [
                    "Data source: robust summary curves after single-curve cleaning, common-grid interpolation, and curve-level outlier handling.",
                    "Model: invariant CANN, independent loading model.",
                    "Covariates: time only. Mass is not used because the fitting target is the robust summary curve.",
                    "Training uses 100% of the three-day robust summary data.",
                ],
                metrics_df=metrics_df,
            )
            metrics_out = metrics_df.copy()
            metrics_out["color"] = color
            metrics_out["round"] = "round_1"
            metrics_out["group_id"] = "robust_summary"
            metrics_out["model_type"] = "independent"
            rows.append(metrics_out)
    return pd.concat(rows, ignore_index=True)


def _files_by_day_from_triplet(row: pd.Series) -> dict[float, list[str]]:
    files = row["files_by_day"]
    out = {}
    for day in DAY_ORDER:
        value = files[day]
        if isinstance(value, list):
            out[day] = value
        else:
            out[day] = [value]
    return out


def _subset_from_files(sample_df: pd.DataFrame, files_by_day: dict[float, list[str]]) -> pd.DataFrame:
    masks = []
    for day, files in files_by_day.items():
        masks.append((sample_df["time_days"] == day) & (sample_df["file_name"].isin(files)))
    subset = sample_df.loc[np.logical_or.reduce(masks)].copy()
    return subset.sort_values(["loading", "time_days", "file_name", "point_index"]).reset_index(drop=True)


def _run_grouped_round(
    round_dir: Path,
    round_name: str,
    sample_df: pd.DataFrame,
    selected_groups: pd.DataFrame,
    joint: bool,
) -> pd.DataFrame:
    rows = []
    if selected_groups.empty:
        return pd.DataFrame()
    for _, group_row in selected_groups.iterrows():
        color = group_row["color"]
        group_id = group_row["group_id"]
        files_by_day = _files_by_day_from_triplet(group_row)
        subset = _subset_from_files(sample_df, files_by_day)
        if subset.empty:
            continue
        summary_subset = _aggregate_selected_subset(subset)
        if summary_subset.empty:
            continue
        group_dir = round_dir / color / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        summary_subset.to_csv(group_dir / "summary_subset.csv", index=False)
        subset.to_csv(group_dir / "sample_subset.csv", index=False)

        if joint:
            model = _fit_joint_model(subset, variant="time_mass", ridge_alpha=1e-4)
            pred_df, metrics_df = _evaluate_model(model, summary_subset, LOADING_ORDER)
            effects_df = _time_effect_table(pred_df)
            _export_model(group_dir / "joint_model.json", model)
            _plot_predictions(
                pred_df,
                metrics_df,
                LOADING_ORDER,
                group_dir / "joint_six_panel.png",
                f"{round_name} | {COLOR_LABELS[color]} | {group_id} | joint compression + shear",
            )
            _plot_time_effects(
                effects_df,
                LOADING_ORDER,
                group_dir / "time_effects.png",
                f"{round_name} time effects | {COLOR_LABELS[color]} | {group_id}",
            )
            _write_processing_note(
                group_dir / "processing_scheme.txt",
                f"{round_name} | {COLOR_LABELS[color]} | {group_id}",
                [
                    "Data source: selected sample-level curves (not averaged for training).",
                    "Selection rule: similar curve shape, similar peak stress, and non-increasing/stable peak trend across days.",
                    "Model: invariant CANN, joint compression + shear model.",
                    "Covariates: time + mass.",
                    f"Selection score = {group_row['score']:.4f}.",
                ],
                files_by_day=files_by_day,
                metrics_df=metrics_df,
            )
            pred_df.to_csv(group_dir / "predictions.csv", index=False)
            metrics_df.to_csv(group_dir / "metrics.csv", index=False)
            effects_df.to_csv(group_dir / "time_effects.csv", index=False)
            metrics_out = metrics_df.copy()
            metrics_out["color"] = color
            metrics_out["round"] = round_name
            metrics_out["group_id"] = group_id
            metrics_out["model_type"] = "joint"
            rows.append(metrics_out)
        else:
            for loading in LOADING_ORDER:
                loading_dir = group_dir / loading
                loading_dir.mkdir(parents=True, exist_ok=True)
                model = _fit_loading_model(subset, loading=loading, variant="time_mass", ridge_alpha=1e-4)
                pred_df, metrics_df = _evaluate_model(model, summary_subset, [loading])
                effects_df = _time_effect_table(pred_df)
                _export_model(loading_dir / "model.json", model)
                _plot_predictions(
                    pred_df,
                    metrics_df,
                    [loading],
                    loading_dir / f"{loading}_three_days.png",
                    f"{round_name} | {COLOR_LABELS[color]} | {group_id} | {loading}",
                )
                _plot_time_effects(
                    effects_df,
                    [loading],
                    loading_dir / "time_effects.png",
                    f"{round_name} time effects | {COLOR_LABELS[color]} | {group_id} | {loading}",
                )
                _write_processing_note(
                    loading_dir / "processing_scheme.txt",
                    f"{round_name} | {COLOR_LABELS[color]} | {group_id} | {loading}",
                    [
                        "Data source: selected sample-level curves (not averaged for training).",
                        "Selection rule: similar curve shape, similar peak stress, and non-increasing/stable peak trend across days.",
                        "Model: invariant CANN, independent loading model.",
                        "Covariates: time + mass.",
                        f"Selection score = {group_row['score']:.4f}.",
                    ],
                    files_by_day=files_by_day,
                    metrics_df=metrics_df,
                )
                pred_df.to_csv(loading_dir / "predictions.csv", index=False)
                metrics_df.to_csv(loading_dir / "metrics.csv", index=False)
                effects_df.to_csv(loading_dir / "time_effects.csv", index=False)
                metrics_out = metrics_df.copy()
                metrics_out["color"] = color
                metrics_out["round"] = round_name
                metrics_out["group_id"] = group_id
                metrics_out["model_type"] = "independent"
                rows.append(metrics_out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _choose_merge_groups(
    selected_groups: pd.DataFrame,
    round_metrics: pd.DataFrame,
    threshold: float,
    merge_group_count: int,
    joint: bool,
) -> pd.DataFrame:
    if selected_groups.empty or round_metrics.empty:
        return pd.DataFrame()
    metric_means = (
        round_metrics.groupby(["color", "group_id"], as_index=False)
        .agg(mean_r2=("r2", "mean"))
    )
    merged = selected_groups.merge(metric_means, on=["color", "group_id"], how="left")
    rows = []
    for color, group in merged.groupby("color", sort=True):
        good_group = group.loc[group["mean_r2"] >= threshold].sort_values(
            ["mean_r2", "score"], ascending=[False, False]
        )
        top = good_group.head(merge_group_count)
        if top.empty:
            top = group.sort_values(["mean_r2", "score"], ascending=[False, False]).head(merge_group_count)
        if top.empty:
            continue
        combined_files = {day: [] for day in DAY_ORDER}
        for files_by_day in top["files_by_day"]:
            for day in DAY_ORDER:
                combined_files[day].append(files_by_day[day])
        rows.append(
            {
                "color": color,
                "group_id": f"{color}_{'joint' if joint else 'indep'}_merged_{len(top):02d}",
                "files_by_day": {day: sorted(set(files)) for day, files in combined_files.items()},
                "score": float(top["score"].mean()),
                "source_groups": top["group_id"].tolist(),
                "mean_r2": float(top["mean_r2"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=["color", "group_id", "files_by_day", "score", "source_groups", "mean_r2"])


def main() -> None:
    args = parse_args()
    cfg = MultiRoundConfig(
        root=args.root.resolve(),
        axial_stress_column=args.axial_stress_column,
        compression_sign=args.compression_sign,
        ridge_alpha=args.ridge_alpha,
        max_triplets_per_color=args.max_triplets_per_color,
        good_r2_threshold=args.good_r2_threshold,
        merge_group_count=args.merge_group_count,
    )
    _, summary_df, sample_df, diagnostics_df = _prepare_merged_inputs(cfg)
    paths = ProjectPaths(root=cfg.root)
    out_dir = paths.output_dir / "multi_round_invariant_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
    summary_df.to_csv(out_dir / "robust_summary_curves.csv", index=False)
    sample_df.to_csv(out_dir / "interpolated_individual_curves.csv", index=False)
    diagnostics_df.to_csv(out_dir / "curve_diagnostics.csv", index=False)

    round1_dir = out_dir / "round_1_robust_summary_independent"
    round1_dir.mkdir(parents=True, exist_ok=True)
    round1_metrics = _run_round_1(round1_dir, summary_df)
    round1_metrics.to_csv(round1_dir / "metrics_summary.csv", index=False)

    candidate_frames = []
    for color in sorted(sample_df["color"].unique()):
        selected = _select_triplets(sample_df, color, cfg.max_triplets_per_color)
        if not selected.empty:
            candidate_frames.append(selected)
    candidate_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    candidate_df.to_csv(out_dir / "selected_triplet_candidates.csv", index=False)

    round2_dir = out_dir / "round_2_single_group_independent"
    round2_dir.mkdir(parents=True, exist_ok=True)
    round2_metrics = _run_grouped_round(round2_dir, "round_2", sample_df, candidate_df, joint=False)
    round2_metrics.to_csv(round2_dir / "metrics_summary.csv", index=False)

    round3_dir = out_dir / "round_3_single_group_joint"
    round3_dir.mkdir(parents=True, exist_ok=True)
    round3_metrics = _run_grouped_round(round3_dir, "round_3", sample_df, candidate_df, joint=True)
    round3_metrics.to_csv(round3_dir / "metrics_summary.csv", index=False)

    merged_round4_groups = _choose_merge_groups(
        candidate_df,
        round2_metrics,
        threshold=cfg.good_r2_threshold,
        merge_group_count=cfg.merge_group_count,
        joint=False,
    )
    round4_dir = out_dir / "round_4_merged_groups_independent"
    round4_dir.mkdir(parents=True, exist_ok=True)
    round4_metrics = _run_grouped_round(round4_dir, "round_4", sample_df, merged_round4_groups, joint=False)
    round4_metrics.to_csv(round4_dir / "metrics_summary.csv", index=False)
    merged_round4_groups.to_csv(round4_dir / "merged_group_selection.csv", index=False)

    merged_round5_groups = _choose_merge_groups(
        candidate_df,
        round3_metrics,
        threshold=cfg.good_r2_threshold,
        merge_group_count=cfg.merge_group_count,
        joint=True,
    )
    round5_dir = out_dir / "round_5_merged_groups_joint"
    round5_dir.mkdir(parents=True, exist_ok=True)
    round5_metrics = _run_grouped_round(round5_dir, "round_5", sample_df, merged_round5_groups, joint=True)
    round5_metrics.to_csv(round5_dir / "metrics_summary.csv", index=False)
    merged_round5_groups.to_csv(round5_dir / "merged_group_selection.csv", index=False)

    overall_metrics = pd.concat(
        [round1_metrics, round2_metrics, round3_metrics, round4_metrics, round5_metrics],
        ignore_index=True,
    )
    overall_metrics.to_csv(out_dir / "overall_metrics.csv", index=False)


if __name__ == "__main__":
    main()
