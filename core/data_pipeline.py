from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.settings import COLOR_TO_CUT, FRESHNESS_TO_HOURS


FILE_RE = re.compile(r"^(green|red)([123])-([1-5])\.xls$", re.IGNORECASE)


@dataclass(frozen=True)
class SampleMeta:
    file_name: str
    color: str
    cut_name: str
    freshness_index: int
    time_hours: int
    time_days: float
    sample_num: int


def parse_sample_meta(file_path: Path) -> SampleMeta:
    match = FILE_RE.match(file_path.name)
    if not match:
        raise ValueError(f"Unexpected file name: {file_path.name}")
    color, freshness_idx, sample_num = match.groups()
    freshness_index = int(freshness_idx)
    time_hours = FRESHNESS_TO_HOURS[freshness_index]
    return SampleMeta(
        file_name=file_path.name,
        color=color.lower(),
        cut_name=COLOR_TO_CUT[color.lower()],
        freshness_index=freshness_index,
        time_hours=time_hours,
        time_days=time_hours / 24.0,
        sample_num=int(sample_num),
    )


def _read_numeric_sheet(file_path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(file_path, sheet_name=sheet_name)
    columns = [str(value).strip().lower().replace(" ", "_") for value in raw.iloc[0].tolist()]
    data = raw.iloc[2:].reset_index(drop=True).copy()
    data.columns = columns
    return data.apply(pd.to_numeric, errors="coerce")


def _estimate_reference_gap(axial_df: pd.DataFrame) -> float:
    stretch = 1.0 + axial_df["strain"] / 100.0
    ref_gap = axial_df["gap"] / stretch
    return float(ref_gap.mean())


def _uniaxial_frame(
    meta: SampleMeta,
    data: pd.DataFrame,
    loading: str,
    reference_gap: float,
    stress_column: str = "stress",
    stress_sign: float = 1.0,
) -> pd.DataFrame:
    deformation = data["gap"] / reference_gap
    out = pd.DataFrame(
        {
            "point_index": np.arange(len(data), dtype=int),
            "deformation": deformation.astype(float),
            "stress_pa": stress_sign * data[stress_column].astype(float),
            "loading": loading,
            "mode": "uniaxial",
        }
    )
    return _attach_meta(meta, out)


def _shear_frame(meta: SampleMeta, data: pd.DataFrame) -> pd.DataFrame:
    deformation = data["strain"] / 100.0
    out = pd.DataFrame(
        {
            "point_index": np.arange(len(data), dtype=int),
            "deformation": deformation.astype(float),
            "stress_pa": data["stress"].astype(float),
            "loading": "shear",
            "mode": "shear",
        }
    )
    return _attach_meta(meta, out)


def _collapse_duplicate_deformation(frame: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    work = frame.copy()
    work["_deformation_key"] = work["deformation"].round(decimals)
    collapsed = (
        work.groupby("_deformation_key", sort=True, as_index=False)
        .agg(
            {
                "file_name": "first",
                "color": "first",
                "cut_name": "first",
                "freshness_index": "first",
                "time_hours": "first",
                "time_days": "first",
                "sample_num": "first",
                "mode": "first",
                "loading": "first",
                "deformation": "median",
                "stress_pa": "median",
            }
        )
        .drop(columns="_deformation_key")
    )
    collapsed["point_index"] = np.arange(len(collapsed), dtype=int)
    return collapsed[frame.columns]


def _clean_single_curve(frame: pd.DataFrame, max_passes: int = 2, z_threshold: float = 3.5) -> pd.DataFrame:
    ordered = _collapse_duplicate_deformation(frame).sort_values("deformation").reset_index(drop=True).copy()
    if len(ordered) < 5:
        ordered["point_index"] = np.arange(len(ordered), dtype=int)
        return ordered

    keep_mask = np.ones(len(ordered), dtype=bool)
    for _ in range(max_passes):
        work = ordered.loc[keep_mask].reset_index(drop=True)
        if len(work) < 5:
            break

        x = work["deformation"].to_numpy(dtype=float)
        y = work["stress_pa"].to_numpy(dtype=float)
        prev_x, next_x = x[:-2], x[2:]
        prev_y, next_y = y[:-2], y[2:]
        mid_x = x[1:-1]
        span = np.maximum(next_x - prev_x, 1e-8)
        expected = prev_y + (next_y - prev_y) * (mid_x - prev_x) / span
        residual = y[1:-1] - expected
        z = np.abs(modified_z_scores(residual))

        left_slope = y[1:-1] - prev_y
        right_slope = next_y - y[1:-1]
        slope_flip = np.sign(left_slope) != np.sign(right_slope)

        to_drop_inner = (z > z_threshold) & slope_flip
        if not np.any(to_drop_inner):
            break

        work_keep = np.ones(len(work), dtype=bool)
        work_keep[1:-1] = ~to_drop_inner
        surviving_index = ordered.index[keep_mask]
        keep_mask[surviving_index] = work_keep

    cleaned = ordered.loc[keep_mask].reset_index(drop=True).copy()
    cleaned["point_index"] = np.arange(len(cleaned), dtype=int)
    return cleaned


def _attach_meta(meta: SampleMeta, frame: pd.DataFrame) -> pd.DataFrame:
    meta_dict = asdict(meta)
    for key, value in meta_dict.items():
        frame[key] = value
    ordered = [
        "file_name",
        "color",
        "cut_name",
        "freshness_index",
        "time_hours",
        "time_days",
        "sample_num",
        "mode",
        "loading",
        "point_index",
        "deformation",
        "stress_pa",
    ]
    return frame[ordered]


def load_workbook(
    file_path: Path,
    axial_stress_column: str = "stress",
    compression_sign: float = 1.0,
) -> pd.DataFrame:
    meta = parse_sample_meta(file_path)
    compression = _read_numeric_sheet(file_path, "Axial - 1")
    tension = _read_numeric_sheet(file_path, "Axial - 2")
    shear = _read_numeric_sheet(file_path, "Peak hold - 3")
    reference_gap = _estimate_reference_gap(compression)
    compression_clean = _clean_single_curve(
        _uniaxial_frame(
            meta,
            compression,
            "compression",
            reference_gap=reference_gap,
            stress_column=axial_stress_column,
            stress_sign=compression_sign,
        )
    )
    unloading_clean = _clean_single_curve(
        _uniaxial_frame(
            meta,
            tension,
            "unloading",
            reference_gap=reference_gap,
            stress_column=axial_stress_column,
            stress_sign=1.0,
        )
    )
    shear_clean = _clean_single_curve(_shear_frame(meta, shear))
    frames = [
        compression_clean,
        unloading_clean,
        shear_clean,
    ]
    return pd.concat(frames, ignore_index=True)


def load_all_workbooks(
    data_dir: Path,
    axial_stress_column: str = "stress",
    compression_sign: float = 1.0,
) -> pd.DataFrame:
    frames = [
        load_workbook(
            path,
            axial_stress_column=axial_stress_column,
            compression_sign=compression_sign,
        )
        for path in sorted(data_dir.glob("*.xls"))
    ]
    if not frames:
        raise FileNotFoundError(f"No .xls files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def modified_z_scores(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if np.isclose(mad, 0.0):
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - median) / mad


def robust_z_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if np.isclose(mad, 0.0):
        return np.zeros_like(values, dtype=float)
    return (values - median) / (1.4826 * mad)


def weighted_huber_mean(
    values: np.ndarray,
    weights: np.ndarray,
    delta: float = 1.5,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return float("nan")

    location = float(np.median(values))
    scale = float(np.median(np.abs(values - location)))
    if np.isclose(scale, 0.0):
        denom = np.sum(weights)
        return float(np.sum(weights * values) / denom) if denom > 0.0 else float(np.mean(values))

    for _ in range(max_iter):
        residual = values - location
        scaled = np.abs(residual) / max(delta * scale, 1e-8)
        huber_weights = np.ones_like(scaled)
        np.divide(1.0, scaled, out=huber_weights, where=scaled > 1.0)
        total_weights = weights * huber_weights
        denom = np.sum(total_weights)
        if denom <= 0.0:
            break
        updated = float(np.sum(total_weights * values) / denom)
        if abs(updated - location) < tol:
            location = updated
            break
        location = updated

    return location


def weighted_huber_curve(curve_matrix: np.ndarray, curve_weights: np.ndarray) -> np.ndarray:
    return np.asarray(
        [weighted_huber_mean(curve_matrix[:, idx], curve_weights) for idx in range(curve_matrix.shape[1])],
        dtype=float,
    )


def aggregate_group_curves(
    raw_df: pd.DataFrame,
    z_threshold: float = 3.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["color", "cut_name", "freshness_index", "time_hours", "time_days", "mode", "loading", "point_index"]
    aggregate_rows: list[dict] = []
    outlier_rows: list[dict] = []

    for group_key, group in raw_df.groupby(group_cols, sort=True):
        group = group.sort_values("sample_num").reset_index(drop=True)
        z_scores = modified_z_scores(group["stress_pa"].to_numpy(dtype=float))
        keep_mask = np.abs(z_scores) <= z_threshold
        if keep_mask.sum() < 3:
            keep_mask = np.ones_like(keep_mask, dtype=bool)

        group = group.assign(
            modified_z=z_scores,
            is_outlier=~keep_mask,
        )
        outlier_rows.extend(group.to_dict("records"))

        kept = group.loc[keep_mask]
        first = kept.iloc[0]
        aggregate_rows.append(
            {
                "color": first["color"],
                "cut_name": first["cut_name"],
                "freshness_index": int(first["freshness_index"]),
                "time_hours": int(first["time_hours"]),
                "time_days": float(first["time_days"]),
                "mode": first["mode"],
                "loading": first["loading"],
                "point_index": int(first["point_index"]),
                "deformation": kept["deformation"].mean(),
                "stress_pa": kept["stress_pa"].mean(),
                "stress_std_pa": kept["stress_pa"].std(ddof=0),
                "n_used": int(len(kept)),
                "n_total": int(len(group)),
                "used_samples": ",".join(str(int(v)) for v in kept["sample_num"].tolist()),
                "dropped_samples": ",".join(str(int(v)) for v in group.loc[~keep_mask, "sample_num"].tolist()),
            }
        )

    aggregated = pd.DataFrame(aggregate_rows).sort_values(group_cols).reset_index(drop=True)
    outliers = pd.DataFrame(outlier_rows).sort_values(group_cols + ["sample_num"]).reset_index(drop=True)
    return aggregated, outliers


def interpolate_and_aggregate_curves(
    raw_df: pd.DataFrame,
    num_uniaxial_points: int = 40,
    num_shear_points: int = 50,
    z_threshold: float = 3.5,
    min_support_samples: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows: list[dict] = []
    diagnostics_rows: list[dict] = []

    group_cols = ["color", "cut_name", "freshness_index", "time_hours", "time_days", "mode", "loading"]
    for group_key, group in raw_df.groupby(group_cols, sort=True):
        color, cut_name, freshness_index, time_hours, time_days, mode, loading = group_key
        sample_curves: list[tuple[str, int, np.ndarray, np.ndarray]] = []
        for (file_name, sample_num), sample_df in group.groupby(["file_name", "sample_num"], sort=True):
            ordered = sample_df.sort_values("deformation")
            x = ordered["deformation"].to_numpy(dtype=float)
            y = ordered["stress_pa"].to_numpy(dtype=float)
            if len(x) < 2:
                continue
            sample_curves.append((file_name, int(sample_num), x, y))

        if len(sample_curves) < 2:
            continue

        lower = min(curve[2].min() for curve in sample_curves)
        upper = max(curve[2].max() for curve in sample_curves)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            continue

        n_points = num_shear_points if mode == "shear" else num_uniaxial_points
        grid = np.linspace(lower, upper, n_points)
        sample_matrix = np.full((len(sample_curves), n_points), np.nan, dtype=float)
        sample_ids: list[int] = []
        sample_names: list[str] = []

        for idx, (file_name, sample_num, x, y) in enumerate(sample_curves):
            support_mask = (grid >= x.min()) & (grid <= x.max())
            if np.any(support_mask):
                sample_matrix[idx, support_mask] = np.interp(grid[support_mask], x, y)
            sample_ids.append(sample_num)
            sample_names.append(file_name)

        for point_idx, deformation in enumerate(grid):
            values = sample_matrix[:, point_idx]
            support_mask = np.isfinite(values)
            if int(support_mask.sum()) < min_support_samples:
                continue

            supported_values = values[support_mask]
            z_scores_local = modified_z_scores(supported_values)
            z_scores = np.full_like(values, np.nan, dtype=float)
            z_scores[support_mask] = z_scores_local
            keep_mask = np.abs(z_scores) <= z_threshold
            keep_mask &= support_mask
            if keep_mask.sum() < min_support_samples:
                keep_mask = support_mask.copy()

            kept = values[keep_mask]
            aggregate_rows.append(
                {
                    "file_name": "aggregated",
                    "color": color,
                    "cut_name": cut_name,
                    "freshness_index": int(freshness_index),
                    "time_hours": int(time_hours),
                    "time_days": float(time_days),
                    "sample_num": 0,
                    "mode": mode,
                    "loading": loading,
                    "point_index": point_idx,
                    "deformation": float(deformation),
                    "stress_pa": float(np.mean(kept)),
                    "stress_std_pa": float(np.std(kept, ddof=0)),
                    "n_used": int(keep_mask.sum()),
                    "n_total": int(support_mask.sum()),
                    "used_samples": ",".join(str(sample_ids[i]) for i in np.flatnonzero(keep_mask)),
                    "used_files": ",".join(sample_names[i] for i in np.flatnonzero(keep_mask)),
                    "dropped_samples": ",".join(str(sample_ids[i]) for i in np.flatnonzero(~keep_mask)),
                }
            )
            for sample_idx, value in enumerate(values):
                if not support_mask[sample_idx]:
                    continue
                diagnostics_rows.append(
                    {
                        "color": color,
                        "cut_name": cut_name,
                        "freshness_index": int(freshness_index),
                        "time_hours": int(time_hours),
                        "time_days": float(time_days),
                        "mode": mode,
                        "loading": loading,
                        "point_index": point_idx,
                        "deformation": float(deformation),
                        "file_name": sample_names[sample_idx],
                        "sample_num": sample_ids[sample_idx],
                        "interpolated_stress_pa": float(value),
                        "modified_z": float(z_scores[sample_idx]),
                        "is_outlier": bool(not keep_mask[sample_idx]),
                    }
                )

    aggregated = pd.DataFrame(aggregate_rows).sort_values(group_cols + ["point_index"]).reset_index(drop=True)
    diagnostics = pd.DataFrame(diagnostics_rows).sort_values(group_cols + ["point_index", "sample_num"]).reset_index(drop=True)
    return aggregated, diagnostics


def preprocess_robust_curves(
    raw_df: pd.DataFrame,
    num_uniaxial_points: int = 80,
    num_shear_points: int = 100,
    local_outlier_threshold: float = 3.0,
    local_outlier_fraction_threshold: float = 0.2,
    global_z_threshold: float = 2.5,
    feature_z_threshold: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    representative_rows: list[dict] = []
    interpolated_rows: list[dict] = []
    diagnostics_rows: list[dict] = []

    group_cols = ["color", "cut_name", "freshness_index", "time_hours", "time_days", "mode", "loading"]
    for group_key, group in raw_df.groupby(group_cols, sort=True):
        color, cut_name, freshness_index, time_hours, time_days, mode, loading = group_key
        samples = []
        for (file_name, sample_num), sample_df in group.groupby(["file_name", "sample_num"], sort=True):
            ordered = sample_df.sort_values("deformation")
            x = ordered["deformation"].to_numpy(dtype=float)
            y = ordered["stress_pa"].to_numpy(dtype=float)
            if len(x) < 2:
                continue
            samples.append((file_name, int(sample_num), x, y))
        if len(samples) < 2:
            continue

        lower = max(curve[2].min() for curve in samples)
        upper = min(curve[2].max() for curve in samples)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            continue

        n_points = num_shear_points if mode == "shear" else num_uniaxial_points
        grid = np.linspace(lower, upper, n_points)
        sample_matrix = np.vstack([np.interp(grid, x, y) for _, _, x, y in samples])
        sample_names = [name for name, *_ in samples]
        sample_ids = [sid for _, sid, *_ in samples]

        median_curve = np.median(sample_matrix, axis=0)
        mad_curve = np.median(np.abs(sample_matrix - median_curve), axis=0)
        safe_mad = mad_curve + 1e-6

        rel_dev = (sample_matrix - median_curve) / safe_mad
        global_distance = np.sqrt(np.mean(rel_dev**2, axis=1))
        local_outlier_fraction = np.mean(np.abs(rel_dev) > local_outlier_threshold, axis=1)
        feature_score = _curve_feature_score(grid, sample_matrix)

        global_z = np.abs(robust_z_scores(global_distance))
        feature_z = np.abs(robust_z_scores(feature_score))
        curve_score = global_z + feature_z + local_outlier_fraction
        is_curve_outlier = (global_z > global_z_threshold) & (
            (local_outlier_fraction > local_outlier_fraction_threshold) | (feature_z > feature_z_threshold)
        )
        if np.all(is_curve_outlier):
            is_curve_outlier = np.zeros_like(is_curve_outlier, dtype=bool)

        curve_weight = 1.0 / (1.0 + (curve_score / 3.0) ** 2)
        curve_weight = np.clip(curve_weight, 0.1, 1.0)

        kept_matrix = sample_matrix[~is_curve_outlier]
        kept_weights = curve_weight[~is_curve_outlier]
        rep_curve = weighted_huber_curve(kept_matrix, kept_weights)
        q25 = np.quantile(kept_matrix, 0.25, axis=0)
        q75 = np.quantile(kept_matrix, 0.75, axis=0)

        for idx, (file_name, sample_num, _, _) in enumerate(samples):
            diagnostics_rows.append(
                {
                    "color": color,
                    "cut_name": cut_name,
                    "freshness_index": int(freshness_index),
                    "time_hours": int(time_hours),
                    "time_days": float(time_days),
                    "mode": mode,
                    "loading": loading,
                    "file_name": file_name,
                    "sample_num": sample_num,
                    "global_distance": float(global_distance[idx]),
                    "global_z": float(global_z[idx]),
                    "local_outlier_fraction": float(local_outlier_fraction[idx]),
                    "feature_score": float(feature_score[idx]),
                    "feature_z": float(feature_z[idx]),
                    "curve_score": float(curve_score[idx]),
                    "curve_weight": float(curve_weight[idx]),
                    "is_curve_outlier": bool(is_curve_outlier[idx]),
                }
            )
            for point_idx, deformation in enumerate(grid):
                interpolated_rows.append(
                    {
                        "file_name": file_name,
                        "color": color,
                        "cut_name": cut_name,
                        "freshness_index": int(freshness_index),
                        "time_hours": int(time_hours),
                        "time_days": float(time_days),
                        "sample_num": int(sample_num),
                        "mode": mode,
                        "loading": loading,
                        "point_index": point_idx,
                        "deformation": float(deformation),
                        "stress_pa": float(sample_matrix[idx, point_idx]),
                        "curve_weight": float(curve_weight[idx]),
                        "is_curve_outlier": bool(is_curve_outlier[idx]),
                    }
                )

        for point_idx, deformation in enumerate(grid):
            representative_rows.append(
                {
                    "file_name": "robust_summary",
                    "color": color,
                    "cut_name": cut_name,
                    "freshness_index": int(freshness_index),
                    "time_hours": int(time_hours),
                    "time_days": float(time_days),
                    "sample_num": 0,
                    "mode": mode,
                    "loading": loading,
                    "point_index": point_idx,
                    "deformation": float(deformation),
                    "stress_pa": float(rep_curve[point_idx]),
                    "stress_q25_pa": float(q25[point_idx]),
                    "stress_q75_pa": float(q75[point_idx]),
                    "n_used": int(kept_matrix.shape[0]),
                    "n_total": int(sample_matrix.shape[0]),
                    "used_samples": ",".join(str(sample_ids[i]) for i in np.flatnonzero(~is_curve_outlier)),
                    "dropped_samples": ",".join(str(sample_ids[i]) for i in np.flatnonzero(is_curve_outlier)),
                }
            )

    representative_df = pd.DataFrame(representative_rows)
    interpolated_df = pd.DataFrame(interpolated_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    if not representative_df.empty:
        representative_df = representative_df.sort_values(group_cols + ["point_index"]).reset_index(drop=True)
    if not interpolated_df.empty:
        interpolated_df = interpolated_df.sort_values(group_cols + ["sample_num", "point_index"]).reset_index(drop=True)
    if not diagnostics_df.empty:
        diagnostics_df = diagnostics_df.sort_values(group_cols + ["sample_num"]).reset_index(drop=True)
    return representative_df, interpolated_df, diagnostics_df


def _curve_feature_score(grid: np.ndarray, sample_matrix: np.ndarray) -> np.ndarray:
    feature_rows = []
    n_points = sample_matrix.shape[1]
    slope_window = max(3, int(0.1 * n_points))
    x0 = grid[:slope_window]
    for curve in sample_matrix:
        start_slope = np.polyfit(x0, curve[:slope_window], deg=1)[0]
        end_stress = curve[-1]
        area = np.trapezoid(curve, grid)
        max_abs_stress = np.max(np.abs(curve))
        diff = np.diff(curve)
        monotonicity_breaks = np.sum(np.sign(diff[1:]) != np.sign(diff[:-1]))
        feature_rows.append([start_slope, end_stress, area, max_abs_stress, monotonicity_breaks])
    feature_matrix = np.asarray(feature_rows, dtype=float)
    feature_z = np.vstack([np.abs(robust_z_scores(feature_matrix[:, i])) for i in range(feature_matrix.shape[1])]).T
    return np.mean(feature_z, axis=1)


def build_training_payload(
    sample_level_df: pd.DataFrame,
    color: str,
    random_state: int = 42,
    preferred_test_sample_num: int = 5,
) -> dict[str, np.ndarray]:
    subset = sample_level_df.loc[sample_level_df["color"] == color].copy()
    if subset.empty:
        raise ValueError(f"No sample-level data available for color={color}")

    if "is_curve_outlier" in subset.columns:
        subset = subset.loc[~subset["is_curve_outlier"]].copy()
    elif "is_outlier" in subset.columns:
        subset = subset.loc[~subset["is_outlier"]].copy()
    if subset.empty:
        raise ValueError(f"All sample-level rows were excluded as outliers for color={color}")

    uniaxial = subset.loc[subset["mode"] == "uniaxial"].sort_values(["time_days", "point_index"])
    shear = subset.loc[subset["mode"] == "shear"].sort_values(["time_days", "point_index"])
    if uniaxial.empty or shear.empty:
        raise ValueError(f"Color={color} requires both uniaxial and shear data")

    uni = {
        "stretch": uniaxial["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1),
        "time_days": uniaxial["time_days"].to_numpy(dtype=np.float32).reshape(-1, 1),
        "stress_pa": uniaxial["stress_pa"].to_numpy(dtype=np.float32).reshape(-1, 1),
        "sample_weight": _extract_sample_weights(uniaxial).reshape(-1, 1),
        "loading": uniaxial["loading"].to_numpy(),
    }
    shr = {
        "gamma": shear["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1),
        "time_days": shear["time_days"].to_numpy(dtype=np.float32).reshape(-1, 1),
        "stress_pa": shear["stress_pa"].to_numpy(dtype=np.float32).reshape(-1, 1),
        "sample_weight": _extract_sample_weights(shear).reshape(-1, 1),
    }

    train_files, test_files = _train_test_sample_files_by_group(
        subset,
        preferred_test_sample_num=preferred_test_sample_num,
        random_state=random_state,
    )
    uni_train_idx, uni_test_idx = _membership_indices(uniaxial["file_name"].to_numpy(), train_files, test_files)
    shear_train_idx, shear_test_idx = _membership_indices(shear["file_name"].to_numpy(), train_files, test_files)

    uniaxial_scale = float(np.max(np.abs(uni["stress_pa"][uni_train_idx])))
    shear_scale = float(np.max(np.abs(shr["stress_pa"][shear_train_idx])))
    uniaxial_scale = uniaxial_scale if uniaxial_scale > 0.0 else 1.0
    shear_scale = shear_scale if shear_scale > 0.0 else 1.0

    return {
        "color": color,
        "cut_name": subset["cut_name"].iloc[0],
        "uniaxial_stretch_train": uni["stretch"][uni_train_idx],
        "uniaxial_time_train": uni["time_days"][uni_train_idx],
        "uniaxial_stress_train": uni["stress_pa"][uni_train_idx],
        "uniaxial_weight_train": uni["sample_weight"][uni_train_idx],
        "uniaxial_stretch_test": uni["stretch"][uni_test_idx],
        "uniaxial_time_test": uni["time_days"][uni_test_idx],
        "uniaxial_stress_test": uni["stress_pa"][uni_test_idx],
        "uniaxial_weight_test": uni["sample_weight"][uni_test_idx],
        "shear_gamma_train": shr["gamma"][shear_train_idx],
        "shear_time_train": shr["time_days"][shear_train_idx],
        "shear_stress_train": shr["stress_pa"][shear_train_idx],
        "shear_weight_train": shr["sample_weight"][shear_train_idx],
        "shear_gamma_test": shr["gamma"][shear_test_idx],
        "shear_time_test": shr["time_days"][shear_test_idx],
        "shear_stress_test": shr["stress_pa"][shear_test_idx],
        "shear_weight_test": shr["sample_weight"][shear_test_idx],
        "full_uniaxial_table": uniaxial.reset_index(drop=True),
        "full_shear_table": shear.reset_index(drop=True),
        "uniaxial_scale_pa": uniaxial_scale,
        "shear_scale_pa": shear_scale,
        "train_files": sorted(train_files),
        "test_files": sorted(test_files),
    }


def build_full_training_payload(sample_level_df: pd.DataFrame, color: str) -> dict[str, object]:
    subset = sample_level_df.loc[sample_level_df["color"] == color].copy()
    if "is_curve_outlier" in subset.columns:
        subset = subset.loc[~subset["is_curve_outlier"]].copy()
    elif "is_outlier" in subset.columns:
        subset = subset.loc[~subset["is_outlier"]].copy()
    if subset.empty:
        raise ValueError(f"No sample-level data available for color={color}")

    uniaxial = subset.loc[subset["mode"] == "uniaxial"].sort_values(
        ["time_days", "loading", "sample_num", "point_index"]
    )
    shear = subset.loc[subset["mode"] == "shear"].sort_values(["time_days", "sample_num", "point_index"])

    uniaxial_scale = float(np.max(np.abs(uniaxial["stress_pa"].to_numpy(dtype=np.float32))))
    shear_scale = float(np.max(np.abs(shear["stress_pa"].to_numpy(dtype=np.float32))))
    uniaxial_scale = uniaxial_scale if uniaxial_scale > 0.0 else 1.0
    shear_scale = shear_scale if shear_scale > 0.0 else 1.0

    all_files = sorted(subset["file_name"].drop_duplicates().tolist())
    return {
        "color": color,
        "cut_name": subset["cut_name"].iloc[0],
        "full_uniaxial_table": uniaxial.reset_index(drop=True),
        "full_shear_table": shear.reset_index(drop=True),
        "uniaxial_scale_pa": uniaxial_scale,
        "shear_scale_pa": shear_scale,
        "train_files": all_files,
        "test_files": [],
    }


def build_full_training_payload_from_curves(curves_df: pd.DataFrame, color: str) -> dict[str, object]:
    subset = curves_df.loc[curves_df["color"] == color].copy()
    if subset.empty:
        raise ValueError(f"No aggregated curves available for color={color}")
    if "file_name" not in subset.columns:
        subset["file_name"] = "aggregated"
    if "sample_num" not in subset.columns:
        subset["sample_num"] = 0

    uniaxial = subset.loc[subset["mode"] == "uniaxial"].sort_values(["time_days", "loading", "point_index"])
    shear = subset.loc[subset["mode"] == "shear"].sort_values(["time_days", "point_index"])
    uniaxial_scale = float(np.max(np.abs(uniaxial["stress_pa"].to_numpy(dtype=np.float32))))
    shear_scale = float(np.max(np.abs(shear["stress_pa"].to_numpy(dtype=np.float32))))
    uniaxial_scale = uniaxial_scale if uniaxial_scale > 0.0 else 1.0
    shear_scale = shear_scale if shear_scale > 0.0 else 1.0
    return {
        "color": color,
        "cut_name": subset["cut_name"].iloc[0],
        "full_uniaxial_table": uniaxial.reset_index(drop=True),
        "full_shear_table": shear.reset_index(drop=True),
        "uniaxial_scale_pa": uniaxial_scale,
        "shear_scale_pa": shear_scale,
        "train_files": ["robust_summary"],
        "test_files": [],
    }


def _extract_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    if "curve_weight" in frame.columns:
        return frame["curve_weight"].to_numpy(dtype=np.float32)
    return np.ones(len(frame), dtype=np.float32)


def save_preprocessed_outputs(
    raw_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(tables_dir / "raw_measurements_long.csv", index=False)
    aggregated_df.to_csv(tables_dir / "aggregated_curves.csv", index=False)
    outlier_df.to_csv(tables_dir / "outlier_diagnostics.csv", index=False)

    outlier_col = "is_curve_outlier" if "is_curve_outlier" in outlier_df.columns else "is_outlier"
    summary = {
        "n_raw_rows": int(len(raw_df)),
        "n_aggregated_rows": int(len(aggregated_df)),
        "n_flagged_outliers": int(outlier_df[outlier_col].sum()) if outlier_col in outlier_df.columns else 0,
        "colors": sorted(raw_df["color"].unique().tolist()),
    }
    (tables_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _train_test_sample_files_by_group(
    subset: pd.DataFrame,
    preferred_test_sample_num: int,
    random_state: int,
) -> tuple[set[str], set[str]]:
    grouped = subset[["file_name", "freshness_index", "sample_num"]].drop_duplicates()
    test_files: set[str] = set()
    train_files: set[str] = set()

    rng = np.random.default_rng(random_state)
    for freshness_index, group in grouped.groupby("freshness_index", sort=True):
        preferred = group.loc[group["sample_num"] == preferred_test_sample_num, "file_name"].tolist()
        if preferred:
            selected = preferred[0]
        else:
            candidates = group.sort_values("sample_num")["file_name"].tolist()
            selected = candidates[int(rng.integers(0, len(candidates)))]
        test_files.add(selected)
        train_files.update(set(group["file_name"].tolist()) - {selected})

    if not test_files or not train_files:
        raise ValueError("Failed to create grouped train/test split")
    return train_files, test_files


def _membership_indices(
    file_names: np.ndarray,
    train_files: set[str],
    test_files: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.flatnonzero(np.isin(file_names, list(train_files)))
    test_idx = np.flatnonzero(np.isin(file_names, list(test_files)))
    return train_idx, test_idx


def describe_group_sizes(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["color", "freshness_index", "time_hours", "mode", "loading"]
    return raw_df.groupby(cols).size().rename("n_rows").reset_index()
