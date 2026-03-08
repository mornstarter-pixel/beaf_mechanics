from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from core.data_pipeline import load_all_workbooks, preprocess_robust_curves
from core.settings import ProjectPaths


TIME_LABELS = {
    0.0: "Day 0",
    1.0: "Day 1",
    2.0: "Day 2",
}

COLOR_LABELS = {
    "green": "Sirloin steak",
    "red": "New York strip steak",
}

LOADING_ORDER = ["compression", "unloading", "shear"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot robust-summary grouped beef curves.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--axial-stress-column", choices=["stress", "normal_stress"], default="stress")
    parser.add_argument("--compression-sign", type=float, default=-1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(root=args.root.resolve())
    raw_df = load_all_workbooks(
        paths.data_dir,
        axial_stress_column=args.axial_stress_column,
        compression_sign=args.compression_sign,
    )
    summary_df, interpolated_df, diagnostics_df = preprocess_robust_curves(raw_df)

    out_dir = paths.output_dir / "robust_group_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "robust_summary_curves.csv", index=False)
    interpolated_df.to_csv(out_dir / "interpolated_individual_curves.csv", index=False)
    diagnostics_df.to_csv(out_dir / "curve_diagnostics.csv", index=False)

    for color in sorted(summary_df["color"].unique()):
        fig, axes = plt.subplots(3, 3, figsize=(14, 11))
        color_summary = summary_df.loc[summary_df["color"] == color].copy()
        color_interpolated = interpolated_df.loc[interpolated_df["color"] == color].copy()

        for row_idx, loading in enumerate(LOADING_ORDER):
            for col_idx, time_days in enumerate(sorted(color_summary["time_days"].unique())):
                ax = axes[row_idx, col_idx]
                summary_group = color_summary.loc[
                    (color_summary["loading"] == loading) & (color_summary["time_days"] == time_days)
                ].sort_values("deformation")
                sample_group = color_interpolated.loc[
                    (color_interpolated["loading"] == loading) & (color_interpolated["time_days"] == time_days)
                ].copy()

                for sample_num, sample_df in sample_group.groupby("sample_num", sort=True):
                    ordered = sample_df.sort_values("deformation")
                    outlier = bool(ordered["is_curve_outlier"].iloc[0]) if "is_curve_outlier" in ordered.columns else False
                    alpha = 0.55 if outlier else 0.35
                    color_line = "#d62728" if outlier else "#7a7a7a"
                    ax.scatter(
                        ordered["deformation"],
                        ordered["stress_pa"],
                        s=12,
                        alpha=alpha,
                        color=color_line,
                        label="Curve outlier" if outlier and row_idx == 0 and col_idx == 0 else None,
                    )
                    ax.plot(
                        ordered["deformation"],
                        ordered["stress_pa"],
                        linewidth=0.9,
                        alpha=alpha,
                        color=color_line,
                    )

                if not summary_group.empty:
                    ax.fill_between(
                        summary_group["deformation"],
                        summary_group["stress_q25_pa"],
                        summary_group["stress_q75_pa"],
                        color="#8ecae6",
                        alpha=0.35,
                        linewidth=0.0,
                        label="IQR" if row_idx == 0 and col_idx == 0 else None,
                    )
                    ax.scatter(
                        summary_group["deformation"],
                        summary_group["stress_pa"],
                        s=24,
                        color="#023047",
                        label="Robust median" if row_idx == 0 and col_idx == 0 else None,
                        zorder=5,
                    )
                    ax.plot(
                        summary_group["deformation"],
                        summary_group["stress_pa"],
                        color="#023047",
                        linewidth=2.0,
                        zorder=4,
                    )

                if loading == "compression":
                    ax.invert_xaxis()

                ax.set_title(f"{loading} | {TIME_LABELS.get(float(time_days), time_days)}")
                ax.set_xlabel("stretch [-]" if loading != "shear" else "shear strain [-]")
                ax.set_ylabel("stress [Pa]")
                ax.grid(alpha=0.2)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(frameon=False, fontsize=8)

        fig.suptitle(f"Robust grouped response | {COLOR_LABELS.get(color, color)}", fontsize=18)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"{color}_robust_group_scatter.png", dpi=240)
        plt.close(fig)


if __name__ == "__main__":
    main()
