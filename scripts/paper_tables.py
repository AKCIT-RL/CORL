"""Write the result tables of Section 4 of the paper from paper_stats.json.

Usage (from the CORL root):
  python -m scripts.paper_tables --out ../overleaf_paper/tables
"""

import argparse
import json
import os

ALGOS = ["BC", "TD3-BC", "AWAC", "IQL", "CQL", "DT"]
NAME = {"TD3-BC": "TD3+BC"}


def neg(s):
    return s.replace("[-", "[$-$").replace(", -", ", $-$").replace("{-", "{$-$")


TIER_LABELS = {"tier1": "Fixed command", "tier2": "Variable command", "tier3": "Two-leg balance",
               "tier4": "Fall recovery", "tier5": "Disturbed", "global": "Global"}


def _tier_rows(nt, with_ci):
    rows = []
    for i, k in enumerate(TIER_LABELS):
        best = max(ALGOS, key=lambda a: nt[a][k]["mean"])
        cells = []
        for a in ALGOS:
            x = nt[a][k]
            m = f"{x['mean']:.1f}".replace("-", "$-$") if x["mean"] < 0 else f"{x['mean']:.1f}"
            m = f"\\textbf{{{m}}}" if a == best else m
            cells.append(neg(f"{m} {{\\scriptsize[{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]}}") if with_ci else m)
        label = TIER_LABELS[k] if k == "global" else f"{i + 1} {TIER_LABELS[k]}"
        rows.append(label + " & " + " & ".join(cells) + " \\\\")
    return rows


def tiers(r):
    """Compact table for the paper: means only. tiers_ci() gives the intervals."""
    rows = _tier_rows(r["nominal_tiers"], with_ci=False)
    return ("\\begin{table}[t]\n"
            "\\caption{Mean nominal score per tier (normalized, 100 corresponds to the best expert checkpoint). "
            "Each entry averages, with equal weights, the cells of the tier (two tasks times four datasets), "
            "each cell being the mean of three training runs. The highest estimate in each row is in bold, "
            "which does not by itself establish a significant lead. The supplementary material gives the "
            "95\\% bootstrap intervals.}\n"
            "\\label{tab:tiers}\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
            "Tier & " + " & ".join(NAME.get(a, a) for a in ALGOS) + " \\\\\n\\midrule\n"
            + "\n".join(rows[:5]) + "\n\\midrule\n" + rows[5] + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def tiers_ci(r):
    """Full table with intervals, for the supplementary material."""
    rows = _tier_rows(r["nominal_tiers"], with_ci=True)
    return ("\\begin{table*}[ht]\n"
            "\\caption{Mean nominal score per tier, as in Table~3 of the paper, with 95\\% bootstrap confidence "
            "intervals in brackets. Each run is scored over 100 nominal episodes.}\n"
            "\\label{tab:tiers-ci}\n\\small\n\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
            "Tier & " + " & ".join(NAME.get(a, a) for a in ALGOS) + " \\\\\n\\midrule\n"
            + "\n".join(rows[:5]) + "\n\\midrule\n" + rows[5] + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def algorithms(r):
    at = r["algorithm_table"]
    f = lambda x, c: f"{x:.3f} {{\\scriptsize[{c[0]:.3f}, {c[1]:.3f}]}}"
    rows = []
    for a in ["IQL", "DT", "BC", "AWAC", "TD3-BC", "CQL"]:
        x = at[a]
        rows.append(f"{NAME.get(a, a)} & {f(x['N'], x['N_ci'])} & {f(x['P'], x['P_ci'])} & "
                    f"{f(x['SRR'], x['SRR_ci'])} & {x['SRR_sd_seed_means']:.3f} & {x['D_N']:.1f} & "
                    f"{x['D_P']:.1f} & {x['valid']} & {x['cells_with_eligible']} \\\\")
    return ("\\begin{table*}[t]\n"
            "\\caption{Nominal score $N$, perturbed score $P$ and SRR per algorithm, with 95\\% bootstrap "
            "confidence intervals in brackets, ordered by nominal score. All columns weight the 40 "
            "combinations of task and dataset equally. The SRR and the relative-degradation rates $D_N$ and "
            "$D_P$ (\\%) first average the eligible checkpoints ($N \\geq 0.05$) of each combination, then "
            "the combinations that have at least one. ``Valid'' is the number of eligible checkpoints out "
            "of 120, and ``Cells'' is the number of combinations, out of 40, over which these three columns "
            "are averaged. ``SD seeds'' is the standard deviation of the three seed-level mean SRRs, a description "
            "of seed variation and not a confidence interval.}\n"
            "\\label{tab:algorithms}\n\\small\n\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
            "Algorithm & $N$ & $P$ & SRR & SD seeds & $D_N$ & $D_P$ & Valid & Cells \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def heldout(r):
    h = r["heldout"]
    rows = []
    for a in ALGOS:
        if a == "CQL":
            continue
        c = []
        for t in ["go2-push-recovery", "go2-rough-terrain"]:
            x = h[f"{a}|{t}|expert"]
            c.append(f"{x['ratio']:.2f} {{\\scriptsize[{x['ratio_ci'][0]:.2f}, {x['ratio_ci'][1]:.2f}]}}")
        rows.append(f"{NAME.get(a, a)} & " + " & ".join(c) + " \\\\")
    return ("\\begin{table}[t]\n"
            "\\caption{Retention under the held-out regime of the disturbed-locomotion tier, on the expert datasets: "
            "mean held-out score divided by mean nominal score over the three runs, with 95\\% bootstrap "
            "intervals over runs. Both scores come from the same evaluation of each checkpoint, with 100 "
            "episodes each. CQL is omitted because its nominal score on these datasets is below zero.}\n"
            "\\label{tab:heldout}\n\\small\n\\begin{tabular}{@{}lrr@{}}\n\\toprule\n"
            "Algorithm & Stronger pushes & Held-out terrain \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def sensitivity(r):
    sv = r["sensitivity_srr"]
    lab = [("threshold_0.05", "Main analysis ($N \\geq 0.05$)"),
           ("threshold_0.01", "Threshold $N \\geq 0.01$"),
           ("threshold_0.1", "Threshold $N \\geq 0.10$"),
           ("common_cohort_6", "Cohort where all six pass"),
           ("without_CQL", "Without CQL"),
           ("go2_only", "Go2 tasks only"),
           ("without_h1", "Without \\texttt{h1-gait-tracking}"),
           ("normalized_on_raw_cohort", "Normalized, cohort with $\\bar{R}_N > 0$"),
           ("raw_returns", "Raw returns, same cohort")]
    rows = [f"{l} & {sv[k]['n']} & {sv[k]['eta2_task']:.2f} & {sv[k]['eta2_algorithm']:.2f} & "
            f"{sv[k]['span_task']:.2f} & {sv[k]['span_algorithm']:.2f} \\\\" for k, l in lab]
    rm = sv["ratio_of_means"]
    rows.append(f"Ratio of mean scores & 612 & -- & -- & {rm['span_task']:.2f} & {rm['span_algorithm']:.2f} \\\\")
    return ("\\begin{table}[t]\n"
            "\\caption{Sensitivity of the SRR decomposition. $\\eta^2$ is the share of the variance of the "
            "per-checkpoint SRR that each one-way grouping explains, and span is the range of the group means "
            "over checkpoints. The two ``cohort'' rows use the same 606 checkpoints, which pass $N \\geq 0.05$ "
            "and have a positive mean raw nominal return, so that they isolate the effect of the reference "
            "$R_{\\min}$. The last row replaces the mean of per-checkpoint ratios by the ratio of mean scores.}\n"
            "\\label{tab:sensitivity}\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
            "Variant & $n$ & $\\eta^2_\\text{task}$ & $\\eta^2_\\text{alg}$ & Span task & Span alg \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", default="analysis/paper_stats/results/paper_stats.json")
    parser.add_argument("--out", default="../overleaf_paper/tables")
    args = parser.parse_args()
    r = json.load(open(args.stats))
    os.makedirs(args.out, exist_ok=True)
    for name, fn in [("tiers", tiers), ("tiers_ci", tiers_ci), ("algorithms", algorithms), ("heldout", heldout), ("sensitivity", sensitivity)]:
        with open(os.path.join(args.out, f"{name}.tex"), "w") as f:
            f.write(fn(r))
    print("wrote", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
