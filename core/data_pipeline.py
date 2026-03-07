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


def _uniaxial_frame(meta: SampleMeta, data: pd.DataFrame, loading: str) -> pd.DataFrame:
    deformation = 1.0 + data["strain"] / 100.0
    out = pd.DataFrame(
        {
            "point_index": np.arange(len(data), dtype=int),
            "deformation": deformation.astype(float),
            "stress_pa": data["stress"].astype(float),
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


def load_workbook(file_path: Path) -> pd.DataFrame:
    meta = parse_sample_meta(file_path)
    compression = _read_numeric_sheet(file_path, "Axial - 1")
    tension = _read_numeric_sheet(file_path, "Axial - 2")
    shear = _read_numeric_sheet(file_path, "Peak hold - 3")
    frames = [
        _uniaxial_frame(meta, compression, "compression"),
        _uniaxial_frame(meta, tension, "tension"),
        _shear_frame(meta, shear),
    ]
    return pd.concat(frames, ignore_index=True)


def load_all_workbooks(data_dir: Path) -> pd.DataFrame:
    frames = [load_workbook(path) for path in sorted(data_dir.glob("*.xls"))]
    if not frames:
        raise FileNotFoundError(f"No .xls files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def modified_z_scores(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if np.isclose(mad, 0.0):
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - median) / mad


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


def build_training_payload(
    sample_level_df: pd.DataFrame,
    color: str,
    random_state: int = 42,
    preferred_test_sample_num: int = 5,
) -> dict[str, np.ndarray]:
    subset = sample_level_df.loc[sample_level_df["color"] == color].copy()
    if subset.empty:
        raise ValueError(f"No sample-level data available for color={color}")

    if "is_outlier" in subset.columns:
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
        "sample_weight": np.ones((len(uniaxial), 1), dtype=np.float32),
        "loading": uniaxial["loading"].to_numpy(),
    }
    shr = {
        "gamma": shear["deformation"].to_numpy(dtype=np.float32).reshape(-1, 1),
        "time_days": shear["time_days"].to_numpy(dtype=np.float32).reshape(-1, 1),
        "stress_pa": shear["stress_pa"].to_numpy(dtype=np.float32).reshape(-1, 1),
        "sample_weight": np.ones((len(shear), 1), dtype=np.float32),
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

    summary = {
        "n_raw_rows": int(len(raw_df)),
        "n_aggregated_rows": int(len(aggregated_df)),
        "n_flagged_outliers": int(outlier_df["is_outlier"].sum()),
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
