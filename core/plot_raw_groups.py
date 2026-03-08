from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from core.data_pipeline import load_all_workbooks
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
    parser = argparse.ArgumentParser(description="Plot raw grouped beef data as scatter plots.")
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

    out_dir = paths.output_dir / "raw_group_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for color in sorted(raw_df["color"].unique()):
        fig, axes = plt.subplots(3, 3, figsize=(14, 11))
        subset = raw_df.loc[raw_df["color"] == color].copy()
        for row_idx, loading in enumerate(LOADING_ORDER):
            for col_idx, time_days in enumerate(sorted(subset["time_days"].unique())):
                ax = axes[row_idx, col_idx]
                group = subset.loc[(subset["loading"] == loading) & (subset["time_days"] == time_days)].copy()
                for sample_num, sample_df in group.groupby("sample_num", sort=True):
                    ordered = sample_df.sort_values("deformation")
                    ax.scatter(
                        ordered["deformation"],
                        ordered["stress_pa"],
                        s=18,
                        alpha=0.75,
                        label=f"S{sample_num}",
                    )
                    ax.plot(ordered["deformation"], ordered["stress_pa"], linewidth=0.8, alpha=0.45)

                if loading == "compression":
                    ax.invert_xaxis()

                ax.set_title(f"{loading} | {TIME_LABELS.get(float(time_days), time_days)}")
                ax.set_xlabel("stretch [-]" if loading != "shear" else "shear strain [-]")
                ax.set_ylabel("stress [Pa]")
                ax.grid(alpha=0.2)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(frameon=False, fontsize=8, ncol=3)

        fig.suptitle(f"Raw grouped data | {COLOR_LABELS.get(color, color)}", fontsize=18)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"{color}_raw_group_scatter.png", dpi=240)
        plt.close(fig)


if __name__ == "__main__":
    main()
