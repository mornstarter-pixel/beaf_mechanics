from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from core.settings import ProjectPaths


COLOR_LABELS = {
    "green": "Sirloin steak",
    "red": "New York strip steak",
}

TIME_LABELS = {
    0.0: "Day 0",
    1.0: "Day 1",
    2.0: "Day 2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves used by the Fig.1 comp+shear model.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(root=args.root.resolve())
    base_dir = paths.output_dir / "fig1_comp_shear_time"
    summary_df = pd.read_csv(base_dir / "robust_summary_curves.csv")
    sample_df = pd.read_csv(base_dir / "interpolated_individual_curves.csv")
    summary_df = summary_df.loc[summary_df["loading"].isin(["compression", "shear"])].copy()
    sample_df = sample_df.loc[sample_df["loading"].isin(["compression", "shear"])].copy()

    for color in sorted(summary_df["color"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        color_summary = summary_df.loc[summary_df["color"] == color].copy()
        color_samples = sample_df.loc[(sample_df["color"] == color) & (~sample_df["is_curve_outlier"])].copy()

        for row_idx, loading in enumerate(["compression", "shear"]):
            for col_idx, day in enumerate(sorted(color_summary["time_days"].unique())):
                ax = axes[row_idx, col_idx]
                s_group = color_summary.loc[
                    (color_summary["loading"] == loading) & (color_summary["time_days"] == day)
                ].sort_values("deformation")
                i_group = color_samples.loc[
                    (color_samples["loading"] == loading) & (color_samples["time_days"] == day)
                ].copy()

                for _, sample_curve in i_group.groupby("file_name", sort=True):
                    sample_curve = sample_curve.sort_values("deformation")
                    y_curve = sample_curve["stress_pa"].abs() if loading == "compression" else sample_curve["stress_pa"]
                    ax.plot(sample_curve["deformation"], y_curve, color="#9a9a9a", linewidth=1.0, alpha=0.35)
                    ax.scatter(sample_curve["deformation"], y_curve, color="#9a9a9a", s=10, alpha=0.25)

                if not s_group.empty:
                    y_mid = s_group["stress_pa"].abs() if loading == "compression" else s_group["stress_pa"]
                    y_q25 = s_group["stress_q25_pa"].abs() if loading == "compression" else s_group["stress_q25_pa"]
                    y_q75 = s_group["stress_q75_pa"].abs() if loading == "compression" else s_group["stress_q75_pa"]
                    ax.fill_between(
                        s_group["deformation"],
                        y_q25,
                        y_q75,
                        color="#8ecae6",
                        alpha=0.35,
                        linewidth=0.0,
                    )
                    ax.plot(s_group["deformation"], y_mid, color="#023047", linewidth=2.2)
                    ax.scatter(s_group["deformation"], y_mid, color="#023047", s=22, zorder=5)

                if loading == "compression":
                    ax.invert_xaxis()
                    ax.set_xlabel("stretch [-]")
                else:
                    ax.set_xlabel("shear strain [-]")
                ax.set_ylabel("stress [Pa]")
                ax.set_title(f"{TIME_LABELS[float(day)]} | {loading}")
                ax.grid(alpha=0.2)

        fig.suptitle(f"Training curves and robust summaries | {COLOR_LABELS[color]}", fontsize=17)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(base_dir / color / f"{color}_robust_summary_training_data.png", dpi=240)
        plt.close(fig)


if __name__ == "__main__":
    main()
