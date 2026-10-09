"""Figures for the paper, built from the published SRR metrics.

Reads sim2real/metrics.csv from the Hugging Face dataset akcit-rl/offline-benchmark
(or a local copy via --csv) and writes:

  nominal_vs_perturbed.pdf  one panel per algorithm: nominal vs. perturbed score of
                            every checkpoint, the identity line, the joint linear fit
                            and the band N < 0.05 where the SRR is not computed.
  srr_decomposition.pdf     SRR of every valid checkpoint grouped by task (top) and by
                            algorithm (bottom) on a shared axis, with group means.

Conventions match visao_geral_repositorios.md: suite humanoid_gym_relative_v2, the
orphan row DT-H1JoystickGaitTracking-f12bb1c4 is dropped, the AWAC run without a
score counts as nominal = perturbed = 0, and SRR uses only ratio_valid rows.

Usage (from the CORL root):
  python -m scripts.paper_figures --out ../overleaf_paper/figures
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUITE = "humanoid_gym_relative_v2"  # v1 on Go2 + friction fix on G1/H1 (scripts/merge_srr_v2.py)
ORPHAN = "DT-H1JoystickGaitTracking-f12bb1c4"
FLOOR = 0.05  # SRR eligibility threshold on the nominal score
ALGOS = ["IQL", "DT", "BC", "AWAC", "TD3-BC", "CQL"]  # nominal ranking order

# Reference palette (dataviz skill), validated on the light surface.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
CONTEXT = "#cfcdc8"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e4df"

plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def load(csv_path=None):
    if csv_path is None:
        from huggingface_hub import hf_hub_download
        csv_path = hf_hub_download("akcit-rl/offline-benchmark", "sim2real/metrics.csv", repo_type="dataset")
    df = pd.read_csv(csv_path)
    df = df[(df["suite"] == SUITE) & (df["checkpoint"] != ORPHAN)].copy()
    df[["nominal_score", "score"]] = df[["nominal_score", "score"]].fillna(0.0)
    return df


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def nominal_vs_perturbed(df, out):
    a, b = np.polyfit(df["nominal_score"], df["score"], 1)
    lo, hi = -0.6, 1.15
    fig, axes = plt.subplots(1, 6, figsize=(7.0, 1.55), sharex=True, sharey=True)
    for ax, algo in zip(axes, ALGOS):
        style(ax)
        ax.axvspan(lo, FLOOR, color=GRID, alpha=0.6, linewidth=0, zorder=0)
        rest = df[df["algorithm"] != algo]
        mine = df[df["algorithm"] == algo]
        ax.scatter(rest["nominal_score"], rest["score"], s=3, color=CONTEXT, linewidths=0, zorder=1)
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=2)
        xs = np.array([lo, hi])
        ax.plot(xs, a * xs + b, color=ORANGE, linewidth=1.0, zorder=3)
        ax.scatter(mine["nominal_score"], mine["score"], s=5, color=BLUE,
                   edgecolors="white", linewidths=0.2, zorder=4)
        valid = mine[mine["ratio_valid"] == True]  # noqa: E712
        srr = valid.groupby(["task", "dataset"])["srr"].mean().mean()  # equal cell weights
        ax.set_title(f"{algo}  (SRR {srr:.2f})", color=INK, pad=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_aspect("equal")
    axes[0].set_ylabel("Perturbed score")
    fig.supxlabel("Nominal score", fontsize=7, y=0.02)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=3, color=BLUE, label="Checkpoints of the algorithm"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=3, color=CONTEXT, label="Other algorithms"),
        plt.Line2D([], [], color=INK_2, linewidth=0.7, linestyle=(0, (3, 2)), label="Perturbed = nominal"),
        plt.Line2D([], [], color=ORANGE, linewidth=1.0, label=f"Joint fit: P = {a:.2f} N + {b:.3f}"),
        plt.matplotlib.patches.Patch(color=GRID, alpha=0.6, label=f"N < {FLOOR}: no SRR"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(pad=0.3, w_pad=0.4)
    path = os.path.join(out, "nominal_vs_perturbed.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, a, b


def srr_decomposition(df, out):
    v = df[df["ratio_valid"] == True].copy()  # noqa: E712
    xmin, xmax = 0.0, 1.2
    clipped = int(((v["srr"] < xmin) | (v["srr"] > xmax)).sum())
    v["srr_plot"] = v["srr"].clip(xmin, xmax)
    # Group means use equal cell weights, as in the paper: first the mean of the
    # eligible checkpoints of each algorithm x task x dataset cell, then the mean of
    # the cells of the group.
    cells = v.groupby(["algorithm", "task", "dataset"])["srr"].mean().reset_index()
    group_mean = {key: cells.groupby(key)["srr"].mean() for key in ("task", "algorithm")}
    task_order = group_mean["task"].sort_values(ascending=False).index.tolist()
    algo_order = group_mean["algorithm"].sort_values(ascending=False).index.tolist()
    rng = np.random.default_rng(0)

    fig, (ax_t, ax_a) = plt.subplots(
        2, 1, figsize=(3.33, 3.0), sharex=True,
        gridspec_kw={"height_ratios": [len(task_order), len(algo_order)], "hspace": 0.32},
    )
    for ax, key, order, label in ((ax_t, "task", task_order, "By task"), (ax_a, "algorithm", algo_order, "By algorithm")):
        style(ax)
        ax.grid(False, axis="y")
        means = group_mean[key]
        for i, name in enumerate(order):
            vals = v.loc[v[key] == name, "srr_plot"].to_numpy()
            y = i + rng.uniform(-0.22, 0.22, size=vals.size)
            ax.scatter(vals, y, s=3, color=BLUE, alpha=0.28, linewidths=0, zorder=2)
            ax.plot([means[name]] * 2, [i - 0.36, i + 0.36], color=INK, linewidth=1.4,
                    solid_capstyle="round", zorder=3)
        span = means.max() - means.min()
        ax.axvspan(means.min(), means.max(), color=ORANGE, alpha=0.10, linewidth=0, zorder=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_title(f"{label}: range of means {span:.2f}", loc="left", color=INK, pad=3)
        ax.axvline(1.0, color=INK_2, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax_t.text(1.0, -0.75, "full retention", ha="center", va="bottom", fontsize=6, color=INK_2)
    ax_a.set_xlim(xmin, xmax)
    ax_a.set_xlabel("Shift Retention Ratio (SRR)")
    path = os.path.join(out, "srr_decomposition.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, clipped, len(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Local metrics.csv (default: download from Hugging Face).")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load(args.csv)
    p1, a, b = nominal_vs_perturbed(df, args.out)
    p2, clipped, n_valid = srr_decomposition(df, args.out)
    print(f"{p1}  (fit P = {a:.3f} N + {b:.3f}, {len(df)} checkpoints)")
    print(f"{p2}  ({n_valid} valid checkpoints, {clipped} clipped to the axis range)")


if __name__ == "__main__":
    main()
