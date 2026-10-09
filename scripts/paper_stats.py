"""All statistics reported in Section 4 of the paper, with confidence intervals.

Inputs (analysis/paper_stats/inputs/):
  runs.csv                 metadata of the 720 runs and the end-of-training scores
                           (50 episodes), from scripts/export_wandb_scores.py
  heldout_metrics.csv      held-out regime of the disturbed-locomotion tier, scored by
                           the SRR pipeline (100 episodes per arm), from
                           scripts/unpaired_srr_metrics.py --suite heldout
  sim2real_metrics.csv     published SRR evaluation (Hugging Face akcit-rl/offline-benchmark)
  unpaired_srr_metrics.csv per-checkpoint relative-degradation rates, from
                           scripts/unpaired_srr_metrics.py on the 720 metrics JSONs
  dataset_refs.csv         mean normalized score of each dataset and R_min / R_max

Every nominal score comes from the nominal arm of the SRR evaluation (100 episodes),
so that N is the same number in every section. The 50-episode end-of-training score
of runs.csv is kept as score_50 but not used.

Estimand (fixed benchmark, reviewer STAT-07): for an algorithm a,
  theta_a = (1/40) * sum over the 40 task x dataset cells of the mean over the
  three training runs of the per-run score.
Tier means average the 8 cells of a tier. SRR summaries use the same equal cell
weights: the mean SRR of the eligible checkpoints (N >= 0.05) of each cell, averaged
over the cells that have at least one eligible checkpoint. Contrasts between two
algorithms use the cells where both have one. The eta^2 decompositions are a
different quantity, computed over individual checkpoints.

Confidence intervals: percentile bootstrap (B draws) that resamples the training
runs with replacement inside each algorithm x task x dataset cell, independently
for each algorithm. Nominal and perturbed scores of a checkpoint are resampled
together. For the SRR, only eligible checkpoints are resampled inside each cell,
so the set of cells (the support) stays fixed in every draw. A secondary bootstrap
resamples whole tasks. Percentile intervals are pointwise; for the 15 pairwise
contrasts we also report Bonferroni-simultaneous intervals (level 1 - 0.05/15),
from their own B_bonf draws (default 50,000) on a separate random stream.

Usage (from the CORL root):
  python -m scripts.paper_stats --inputs analysis/paper_stats/inputs \
      --out analysis/paper_stats/results
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ALGOS = ["BC", "TD3-BC", "AWAC", "IQL", "CQL", "DT"]
TIERS = {
    "go2-flat-forward": 1, "h1-gait-tracking": 1,
    "go2-joystick-direction": 2, "g1-joystick-direction": 2,
    "go2-footstand": 3, "go2-handstand": 3,
    "go2-getup": 4, "go2-getup-walk": 4,
    "go2-push-recovery": 5, "go2-rough-terrain": 5,
}
SUITE = "humanoid_gym_relative_v2"  # v1 on Go2 + friction fix on G1/H1 (scripts/merge_srr_v2.py)
ORPHAN = "DT-H1JoystickGaitTracking-f12bb1c4"
FLOOR = 0.05


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def ci(samples, level=0.95):
    lo, hi = np.percentile(samples, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return [float(lo), float(hi)]


def eta2(values, groups):
    values = np.asarray(values, float)
    m = values.mean()
    sst = ((values - m) ** 2).sum()
    df = pd.DataFrame({"v": values, "g": np.asarray(groups)})
    ssb = sum(len(x) * (x.v.mean() - m) ** 2 for _, x in df.groupby("g"))
    return float(ssb / sst)


class StratifiedBootstrap:
    """Resample row indices with replacement inside each stratum."""

    def __init__(self, df, strata, rng):
        self.groups = [np.asarray(ix) for ix in df.groupby(strata, sort=False).indices.values()]
        self.rng = rng

    def draw(self):
        return np.concatenate([g[self.rng.integers(0, len(g), len(g))] for g in self.groups])


def ci_bonf(samples, m=15):
    return ci(samples, level=1 - 0.05 / m)


def cell_weighted(df, value, by=("algorithm",)):
    cells = df.groupby(list(by) + ["task", "dataset"])[value].mean()
    return cells.groupby(level=list(range(len(by)))).mean()


def load_checkpoints(inputs):
    m = pd.read_csv(os.path.join(inputs, "sim2real_metrics.csv"))
    ck = m[(m.suite == SUITE) & (m.checkpoint != ORPHAN)].copy()
    ck[["nominal_score", "score"]] = ck[["nominal_score", "score"]].fillna(0.0)
    return ck


def load_runs(inputs):
    """runs.csv with `score` replaced by the 100-episode nominal arm of the SRR
    evaluation (x100), and the 50-episode end-of-training score kept as score_50."""
    runs = pd.read_csv(os.path.join(inputs, "runs.csv")).rename(columns={"score": "score_50"})
    n = load_checkpoints(inputs).set_index("checkpoint").nominal_score * 100
    runs["score"] = runs.run.map(n)
    assert runs.score.notna().all()
    runs["tier"] = runs.task.map(TIERS)
    return runs


def load(inputs):
    runs = load_runs(inputs)
    ck = load_checkpoints(inputs)
    ck = ck.rename(columns={"nominal_score": "N", "score": "P"})
    ck["valid"] = ck.N >= FLOOR
    ck["srr"] = np.where(ck.valid, ck.P / ck.N, np.nan)
    ck["tier"] = ck.task.map(TIERS)
    u = pd.read_csv(os.path.join(inputs, "unpaired_srr_metrics.csv"))
    ck = ck.merge(u[["checkpoint", "failure_rate_50_nominal", "failure_rate_50", "nominal_sd"]],
                  on="checkpoint", how="left")
    ck["deg_excess"] = ck.failure_rate_50 - ck.failure_rate_50_nominal
    refs = pd.read_csv(os.path.join(inputs, "dataset_refs.csv"))
    assert len(ck) == 720 and ck.groupby("algorithm").size().eq(120).all()
    assert len(runs) == 720
    return runs, ck, refs


def nominal_tiers(runs, B, rng):
    out = {}
    point_tier = runs.groupby(["algorithm", "tier", "task", "dataset"]).score.mean().groupby(["algorithm", "tier"]).mean()
    point_glob = cell_weighted(runs, "score")
    boot = StratifiedBootstrap(runs, ["algorithm", "task", "dataset"], rng)
    bt, bg, brange = [], [], []
    for _ in range(B):
        r = runs.iloc[boot.draw()]
        t = r.groupby(["algorithm", "tier", "task", "dataset"]).score.mean().groupby(["algorithm", "tier"]).mean()
        bt.append(t)
        bg.append(cell_weighted(r, "score"))
        tt = t.unstack("algorithm").drop(columns="CQL")
        brange.append(tt.max(axis=1) - tt.min(axis=1))
    bt, bg, brange = pd.concat(bt, axis=1), pd.concat(bg, axis=1), pd.concat(brange, axis=1)
    for (a, tier), v in point_tier.items():
        out.setdefault(a, {})[f"tier{tier}"] = {"mean": float(v), "ci": ci(bt.loc[(a, tier)])}
    for a, v in point_glob.items():
        out[a]["global"] = {"mean": float(v), "ci": ci(bg.loc[a])}
    pt = point_tier.unstack("algorithm").drop(columns="CQL")
    ranges = (pt.max(axis=1) - pt.min(axis=1))
    largest = brange.idxmax(axis=0)
    rng_out = {
        f"tier{t}": {"range": float(ranges[t]), "ci": ci(brange.loc[t]),
                     "p_largest": float((largest == t).mean())}
        for t in ranges.index
    }
    return out, rng_out


def interaction_anova(runs):
    """Does the difference between algorithms vary across tasks? Classical F and
    a heteroskedasticity-robust (HC3) Wald F as sensitivity check. The residual
    spread differs a lot between algorithms, so the classical F may be optimistic."""
    import statsmodels.formula.api as smf  # local: load_runs() is imported without statsmodels
    from statsmodels.stats.anova import anova_lm

    model = smf.ols("score ~ C(algorithm) * C(task) + C(dataset)", data=runs).fit()
    tab = anova_lm(model, typ=2)
    row = tab.loc["C(algorithm):C(task)"]
    rob = anova_lm(model, typ=2, robust="hc3").loc["C(algorithm):C(task)"]
    sd = (runs.score - model.fittedvalues).groupby(runs.algorithm).std()
    return {"F": float(row.F), "df": [int(row.df), int(tab.loc["Residual"].df)], "p": float(row["PR(>F)"]),
            "F_hc3": float(rob.F), "p_hc3": float(rob["PR(>F)"]),
            "residual_sd_by_algorithm": {a: float(v) for a, v in sd.items()}}


def tier3_gains(runs, refs, B, rng):
    r = runs.merge(refs[["task", "dataset", "data_mean"]], on=["task", "dataset"])
    r["gain"] = r.score - r.data_mean
    t3 = r[r.tier == 3]
    point = t3.groupby(["algorithm", "dataset"]).gain.mean()
    above = (r.groupby(["algorithm", "task", "dataset"]).gain.mean() > 0).groupby("algorithm").mean()
    boot = StratifiedBootstrap(t3, ["algorithm", "task", "dataset"], rng)
    bs = pd.concat([t3.iloc[boot.draw()].groupby(["algorithm", "dataset"]).gain.mean() for _ in range(B)], axis=1)
    return {
        "tier3_gain_by_quality": {f"{a}|{d}": {"mean": float(v), "ci": ci(bs.loc[(a, d)])} for (a, d), v in point.items()},
        "share_cells_above_dataset_mean": {a: float(v) for a, v in above.items()},
    }


def gains_over_bc(runs, B, rng):
    """Mean score minus the BC score on the same task and dataset (equal cell weights).

    Every learned policy, BC included, acts deterministically at evaluation, while the
    medium datasets were collected stochastically, so BC on the same data is the
    comparator for improvement beyond imitation.
    """
    keys = ["algorithm", "task", "dataset"]

    def summarize(r):
        cell = r.groupby(keys).score.mean().unstack("algorithm")
        gain = cell.sub(cell["BC"], axis=0).drop(columns="BC").stack().rename("gain").reset_index()
        gain["tier"] = gain.task.map(TIERS)
        by_tier = gain.groupby(["tier", "algorithm"]).gain.mean()
        by_dataset = gain.groupby(["tier", "algorithm", "dataset"]).gain.mean()
        return by_tier, by_dataset, gain

    by_tier, by_dataset, gain = summarize(runs)
    boot = StratifiedBootstrap(runs, keys, rng)
    bt, bd = [], []
    for _ in range(B):
        t, d, _ = summarize(runs.iloc[boot.draw()])
        bt.append(t); bd.append(d)
    bt, bd = pd.concat(bt, axis=1), pd.concat(bd, axis=1)
    above = (gain.gain > 0).groupby([gain.tier, gain.algorithm]).agg(["sum", "count"])
    return {
        "by_tier": {f"tier{t}|{a}": {"mean": float(v), "ci": ci(bt.loc[(t, a)])} for (t, a), v in by_tier.items()},
        "by_tier_dataset": {f"tier{t}|{a}|{d}": {"mean": float(v), "ci": ci(bd.loc[(t, a, d)])}
                            for (t, a, d), v in by_dataset.items()},
        "cells_above_bc": {f"tier{t}|{a}": [int(r["sum"]), int(r["count"])] for (t, a), r in above.iterrows()},
    }


def pairwise(ck, B, rng):
    """All 15 pairs on N, P (40 cells) and SRR (cells where both have eligible checkpoints)."""
    keys = ["algorithm", "task", "dataset"]
    cellN = ck.groupby(keys).N.mean().unstack("algorithm")
    cellP = ck.groupby(keys).P.mean().unstack("algorithm")
    v = ck[ck.valid]
    cellS = v.groupby(keys).srr.mean().unstack("algorithm")
    boot = StratifiedBootstrap(ck, keys, rng)
    boot_v = StratifiedBootstrap(v, keys, rng)
    bN, bP, bS = [], [], []
    for _ in range(B):
        r = ck.iloc[boot.draw()]
        bN.append(r.groupby(keys).N.mean().unstack("algorithm"))
        bP.append(r.groupby(keys).P.mean().unstack("algorithm"))
        # Only eligible checkpoints are resampled, so every cell keeps its support.
        bS.append(v.iloc[boot_v.draw()].groupby(keys).srr.mean().unstack("algorithm"))
    rows = []
    for a, b in itertools.combinations(ALGOS, 2):
        rec = {"pair": f"{a} vs {b}"}
        for name, point, bl in (("N", cellN, bN), ("P", cellP, bP), ("SRR", cellS, bS)):
            cells = point[[a, b]].dropna().index
            d = point.loc[cells, a] - point.loc[cells, b]
            draws = np.asarray([(x.loc[cells, a] - x.loc[cells, b]).mean() for x in bl], float)
            assert not np.isnan(draws).any(), "support changed inside a draw"
            w = stats.wilcoxon(d.values, zero_method="wilcox", alternative="two-sided")
            rec[name] = {"diff": float(d.mean()), "ci": ci(draws), "ci_bonf15": ci_bonf(draws),
                         "n_cells": int(len(cells)), "W": float(w.statistic), "p": float(w.pvalue),
                         "cells_a_better": int((d > 0).sum())}
        # Interaction: does the shift change the two methods differently? D = (P_a-N_a)-(P_b-N_b)
        Dp = ((cellP[a] - cellN[a]) - (cellP[b] - cellN[b])).mean()
        Dd = np.asarray([((x[a] - y[a]) - (x[b] - y[b])).mean() for x, y in zip(bP, bN)], float)
        rec["D_shift_interaction"] = {"diff": float(Dp), "ci": ci(Dd), "ci_bonf15": ci_bonf(Dd)}
        rows.append(rec)
    ps = [r[m]["p"] for r in rows for m in ("N", "P", "SRR")]
    adj = holm(ps)
    k = 0
    for r in rows:
        for m in ("N", "P", "SRR"):
            r[m]["p_holm45"] = float(adj[k])
            k += 1
    return rows


def pairwise_bonferroni(ck, B, rng):
    """Bonferroni-simultaneous intervals (level 1 - 0.05/15) of the pairwise contrasts,
    from their own B draws (reviewer MLR3-05: with 2,000 draws each tail of these
    intervals rests on about three draws). Same resampling as pairwise(), and the same
    draws for the same rng, with the cell means computed in NumPy so that B can be large."""
    keys = ["algorithm", "task", "dataset"]
    v = ck[ck.valid]

    def cell_means(df, cols):
        """(B, cells, len(cols)) bootstrap cell means and the (algorithm, task, dataset) of each cell."""
        boot = StratifiedBootstrap(df, keys, rng)
        sizes = np.array([len(g) for g in boot.groups])
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        vals = df[cols].to_numpy(float)
        return boot, sizes, starts, vals, list(df.groupby(keys, sort=False).indices)

    bc, sc, stc, vc, cells_c = cell_means(ck, ["N", "P"])
    bv, sv, stv, vv, cells_v = cell_means(v, ["srr"])
    mc = np.empty((B, len(cells_c), 2))
    mv = np.empty((B, len(cells_v)))
    for i in range(B):  # same order of rng calls as pairwise()
        mc[i] = np.add.reduceat(vc[bc.draw()], stc, axis=0) / sc[:, None]
        mv[i] = np.add.reduceat(vv[bv.draw()][:, 0], stv) / sv

    col_c = {c: j for j, c in enumerate(cells_c)}
    col_v = {c: j for j, c in enumerate(cells_v)}
    out = {}
    for a, b in itertools.combinations(ALGOS, 2):
        rec = {}
        td = sorted({c[1:] for c in cells_c if c[0] == a} & {c[1:] for c in cells_c if c[0] == b})
        ia, ib = [col_c[(a, *c)] for c in td], [col_c[(b, *c)] for c in td]
        dN = (mc[:, ia, 0] - mc[:, ib, 0]).mean(axis=1)
        dP = (mc[:, ia, 1] - mc[:, ib, 1]).mean(axis=1)
        tv = sorted({c[1:] for c in cells_v if c[0] == a} & {c[1:] for c in cells_v if c[0] == b})
        dS = (mv[:, [col_v[(a, *c)] for c in tv]] - mv[:, [col_v[(b, *c)] for c in tv]]).mean(axis=1)
        for name, d in (("N", dN), ("P", dP), ("SRR", dS), ("D_shift_interaction", dP - dN)):
            rec[name] = ci_bonf(d)
        out[f"{a} vs {b}"] = rec
    return out


def ranking(ck, B, rng):
    thN = cell_weighted(ck, "N")
    thP = cell_weighted(ck, "P")
    rho_means = stats.spearmanr(thN[ALGOS], thP[ALGOS]).statistic
    boot = StratifiedBootstrap(ck, ["algorithm", "task", "dataset"], rng)
    rho_pool, changes = [], []
    for _ in range(B):
        r = ck.iloc[boot.draw()]
        rho_pool.append(stats.spearmanr(r.N, r.P).statistic)
        n = cell_weighted(r, "N")[ALGOS].rank(ascending=False)
        p = cell_weighted(r, "P")[ALGOS].rank(ascending=False)
        moved = tuple(sorted(a for a in ALGOS if n[a] != p[a]))
        changes.append(moved)
    within = []
    for (t, d), g in ck.groupby(["task", "dataset"]):
        within.append(stats.spearmanr(g.N, g.P).statistic)
    per_algo = {a: float(stats.spearmanr(g.N, g.P).statistic) for a, g in ck.groupby("algorithm")}
    return {
        "theta_N": thN.to_dict(), "theta_P": thP.to_dict(),
        "spearman_pooled": {"rho": float(stats.spearmanr(ck.N, ck.P).statistic), "ci": ci(np.asarray(rho_pool))},
        "spearman_of_six_means": float(rho_means),
        "spearman_within_cell": {"median": float(np.nanmedian(within)), "min": float(np.nanmin(within)),
                                 "max": float(np.nanmax(within)), "n": len(within)},
        "spearman_per_algorithm": per_algo,
        # Within each draw: do the nominal and perturbed orders of the six means coincide?
        "bootstrap_rank_changes": {
            "p_orders_coincide": float(np.mean([len(c) == 0 for c in changes])),
            "p_exactly_two_change": float(np.mean([len(c) == 2 for c in changes])),
            "p_only_DT_IQL_swap": float(np.mean([c == ("DT", "IQL") for c in changes])),
            "most_common_changes": {"|".join(c) or "none": float(f) for c, f in
                                    pd.Series([c for c in changes]).value_counts(normalize=True).head(5).items()},
        },
    }


def linear_fit(ck, B, rng):
    slope, intercept = np.polyfit(ck.N, ck.P, 1)
    pred = slope * ck.N + intercept
    r2 = 1 - ((ck.P - pred) ** 2).sum() / ((ck.P - ck.P.mean()) ** 2).sum()
    boot = StratifiedBootstrap(ck, ["algorithm", "task", "dataset"], rng)
    bs = np.array([np.polyfit(*ck.iloc[boot.draw()][["N", "P"]].T.values, 1) for _ in range(B)])
    # Leave-one-task-out prediction of the perturbed score.
    sse, sst, per_task = 0.0, 0.0, {}
    for t in ck.task.unique():
        tr, te = ck[ck.task != t], ck[ck.task == t]
        s, i = np.polyfit(tr.N, tr.P, 1)
        e = te.P - (s * te.N + i)
        sse += (e ** 2).sum()
        sst += ((te.P - ck.P.mean()) ** 2).sum()
        per_task[t] = float(np.sqrt((e ** 2).mean()))
    return {"slope": float(slope), "slope_ci": ci(bs[:, 0]), "intercept": float(intercept),
            "intercept_ci": ci(bs[:, 1]), "r2_in_sample": float(r2),
            "r2_leave_one_task_out": float(1 - sse / sst), "rmse_leave_one_task_out": per_task}


def decomposition(df, value, B, rng, task_blocks=True):
    point = {"eta2_task": eta2(df[value], df.task), "eta2_algorithm": eta2(df[value], df.algorithm)}
    point["difference"] = point["eta2_task"] - point["eta2_algorithm"]
    tm, am = df.groupby("task")[value].mean(), df.groupby("algorithm")[value].mean()
    point["span_task"] = float(tm.max() - tm.min())
    point["span_algorithm"] = float(am.max() - am.min())
    boot = StratifiedBootstrap(df, ["algorithm", "task", "dataset"], rng)
    draws = []
    for _ in range(B):
        r = df.iloc[boot.draw()]
        et, ea = eta2(r[value], r.task), eta2(r[value], r.algorithm)
        draws.append((et, ea, et - ea))
    draws = np.array(draws)
    out = {k: (float(v) if not isinstance(v, float) else v) for k, v in point.items()}
    out.update({"eta2_task_ci": ci(draws[:, 0]), "eta2_algorithm_ci": ci(draws[:, 1]),
                "difference_ci": ci(draws[:, 2]), "n": int(len(df))})
    if task_blocks:
        tasks = df.task.unique()
        tb = []
        for _ in range(B):
            pick = rng.choice(tasks, len(tasks), replace=True)
            parts = [df[df.task == t].assign(task=f"{t}#{j}") for j, t in enumerate(pick)]
            r = pd.concat(parts)
            tb.append(eta2(r[value], r.task) - eta2(r[value], r.algorithm))
        out["difference_ci_task_blocks"] = ci(np.asarray(tb))
    return out


def joint_anova(df, value, formula="C(task)*C(algorithm) + C(task)*C(dataset) + C(algorithm)*C(dataset)"):
    """Type II sums of squares; shares are SS_effect / SS_total. Use the full
    interaction model only where every task x algorithm and task x dataset cell
    has data (the balanced 720 checkpoints); otherwise use the additive model."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    model = smf.ols(f"{value} ~ {formula}", data=df).fit()
    X = model.model.exog
    assert np.linalg.matrix_rank(X) == X.shape[1], "rank-deficient design"
    tab = anova_lm(model, typ=2)
    sst = ((df[value] - df[value].mean()) ** 2).sum()
    # Type II effect SS normalized by the total SS. In an unbalanced design these
    # are not an additive partition of the variance (they need not sum to 1).
    out = {k: {"ss_share": float(r.sum_sq / sst), "df": int(r.df),
               "F": (None if pd.isna(r.F) else float(r.F)),
               "p": (None if pd.isna(r["PR(>F)"]) else float(r["PR(>F)"]))}
           for k, r in tab.iterrows()}
    out["_model"] = {"r2": float(model.rsquared), "n": int(len(df)), "formula": formula,
                     "sum_of_shares": float(sum(v["ss_share"] for v in out.values()))}
    return out


def sensitivity(ck, refs):
    rows = {}
    base = ck.copy()
    for th in (0.01, 0.05, 0.10):
        d = base[base.N >= th].assign(s=lambda x: x.P / x.N)
        rows[f"threshold_{th}"] = d
    v = base[base.valid].assign(s=lambda x: x.P / x.N)
    rows["without_CQL"] = v[v.algorithm != "CQL"]
    k = ["task", "dataset", "train_seed"]
    full = v.groupby(k).algorithm.nunique()
    rows["common_cohort_6"] = v.set_index(k).loc[full[full == 6].index].reset_index()
    rows["go2_only"] = v[v.task.str.startswith("go2")]
    rows["without_h1"] = v[v.task != "h1-gait-tracking"]
    rr = base.merge(refs[refs.dataset == "expert"][["task", "r_min", "r_max"]], on="task")
    rr["Nraw"] = rr.r_min + rr.N * (rr.r_max - rr.r_min)
    rr["Praw"] = rr.r_min + rr.P * (rr.r_max - rr.r_min)
    same = rr[(rr.N >= FLOOR) & (rr.Nraw > 0)]
    rows["raw_returns"] = same.assign(s=lambda x: x.Praw / x.Nraw)
    # Same 606 checkpoints with the normalized ratio, to separate the effect of the
    # reference from the effect of the cohort.
    rows["normalized_on_raw_cohort"] = same.assign(s=lambda x: x.P / x.N)
    out = {}
    for name, d in rows.items():
        tm, am = d.groupby("task").s.mean(), d.groupby("algorithm").s.mean()
        out[name] = {"n": int(len(d)), "eta2_task": eta2(d.s, d.task), "eta2_algorithm": eta2(d.s, d.algorithm),
                     "span_task": float(tm.max() - tm.min()), "span_algorithm": float(am.max() - am.min())}
    rm = v.groupby("algorithm").apply(lambda x: x.P.mean() / x.N.mean(), include_groups=False)
    rt = v.groupby("task").apply(lambda x: x.P.mean() / x.N.mean(), include_groups=False)
    out["ratio_of_means"] = {"span_task": float(rt.max() - rt.min()), "span_algorithm": float(rm.max() - rm.min()),
                             "per_algorithm": rm.to_dict()}
    lot = []
    for t in v.task.unique():
        d = v[v.task != t]
        lot.append(eta2(d.s, d.task) - eta2(d.s, d.algorithm))
    out["leave_one_task_out_difference"] = {"min": float(min(lot)), "max": float(max(lot))}
    out["leave_one_task_out_by_task"] = {
        t: {"eta2_task": eta2(v[v.task != t].s, v[v.task != t].task),
            "eta2_algorithm": eta2(v[v.task != t].s, v[v.task != t].algorithm)} for t in v.task.unique()}
    return out


def srr_by_nominal_level(ck):
    v = ck[ck.valid].copy()
    v["band"] = pd.cut(v.N, [FLOOR, 0.2, 0.5, 10], right=False, labels=["0.05-0.2", "0.2-0.5", ">=0.5"])
    tab = v.groupby(["task", "band"], observed=True).srr.agg(["mean", "size"])
    overall = v.groupby("band", observed=True).srr.agg(["mean", "size"])
    return {"overall": {str(k): {"mean": float(r["mean"]), "n": int(r["size"])} for k, r in overall.iterrows()},
            "by_task": {f"{t}|{b}": {"mean": float(r["mean"]), "n": int(r["size"])} for (t, b), r in tab.iterrows()}}


def algorithm_table(ck, B, rng):
    """N and P average all checkpoints (3 per cell, so equal cell weights). SRR and
    the degradation rates average the eligible checkpoints of each cell, then the
    cells with at least one eligible checkpoint, with equal weights."""
    keys = ["algorithm", "task", "dataset"]
    v = ck[ck.valid]
    out = {}
    boot_all = StratifiedBootstrap(ck, keys, rng)
    boot_v = StratifiedBootstrap(v, keys, rng)
    bN, bP, bS = [], [], []
    for _ in range(B):
        r = ck.iloc[boot_all.draw()]
        bN.append(cell_weighted(r, "N"))
        bP.append(cell_weighted(r, "P"))
        bS.append(cell_weighted(v.iloc[boot_v.draw()], "srr"))
    bN, bP, bS = pd.concat(bN, axis=1), pd.concat(bP, axis=1), pd.concat(bS, axis=1)
    srr_cell = cell_weighted(v, "srr")
    srr_ckpt = v.groupby("algorithm").srr.mean()
    seed = v.groupby(["algorithm", "train_seed"]).srr.mean().unstack()
    elig = v.groupby(keys).size().unstack("algorithm").reindex(
        ck.groupby(keys).size().unstack("algorithm").index).fillna(0)
    for a in ALGOS:
        va = v[v.algorithm == a]
        out[a] = {
            "N": float(cell_weighted(ck, "N")[a]), "N_ci": ci(bN.loc[a]),
            "P": float(cell_weighted(ck, "P")[a]), "P_ci": ci(bP.loc[a]),
            "SRR": float(srr_cell[a]), "SRR_ci": ci(bS.loc[a]),
            "SRR_checkpoint_mean": float(srr_ckpt[a]),
            "SRR_seed_means": seed.loc[a].round(4).tolist(), "SRR_sd_seed_means": float(seed.loc[a].std(ddof=1)),
            "valid": int(len(va)), "cells_with_eligible": int((elig[a] > 0).sum()),
            "cells_with_one_eligible": int((elig[a] == 1).sum()),
            "D_N": float(cell_weighted(va, "failure_rate_50_nominal")[a]),
            "D_P": float(cell_weighted(va, "failure_rate_50")[a]),
        }
    cv = (v.nominal_sd / v.N)
    out["_nominal_episode_cv"] = {
        "definition": "SD of the nominal episode scores of a checkpoint divided by its mean N, over eligible checkpoints",
        "median": float(cv.median()), "mean": float(cv.mean()),
        "iqr": [float(cv.quantile(0.25)), float(cv.quantile(0.75))],
        "median_by_algorithm": {a: float(x) for a, x in cv.groupby(v.algorithm).median().items()},
    }
    out["_eligible_per_cell"] = {f"{t}|{d}|{a}": int(n) for (t, d), row in elig.iterrows() for a, n in row.items()}
    return out


def heldout(runs, inputs, B, rng):
    """Held-out over nominal score per cell. Both arms come from the same SRR run of
    each checkpoint, so the nominal here is that run's own nominal arm (x100)."""
    hm = pd.read_csv(os.path.join(inputs, "heldout_metrics.csv"))
    h = runs[["run", "algorithm", "task", "dataset"]].merge(
        hm[["checkpoint", "nominal_score", "perturbed_score"]], left_on="run", right_on="checkpoint")
    assert len(h) == 144, f"expected the 144 runs of the tier, got {len(h)}"
    h = h.assign(score=h.nominal_score * 100, shifted=h.perturbed_score * 100)
    floor = FLOOR * 100
    out = {}
    for (a, t, d), g in h.groupby(["algorithm", "task", "dataset"]):
        ratio = g.shifted.mean() / g.score.mean() if g.score.mean() >= floor else np.nan
        draws = []
        for _ in range(B):
            s = g.iloc[rng.integers(0, len(g), len(g))]
            draws.append(s.shifted.mean() / s.score.mean() if s.score.mean() >= floor else np.nan)
        out[f"{a}|{t}|{d}"] = {"nominal": float(g.score.mean()), "heldout": float(g.shifted.mean()),
                               "ratio": (None if np.isnan(ratio) else float(ratio)),
                               "ratio_ci": (None if np.isnan(ratio) else ci(np.asarray(draws)[~np.isnan(draws)]))}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="analysis/paper_stats/inputs")
    parser.add_argument("--out", default="analysis/paper_stats/results")
    parser.add_argument("--B", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--B_bonf", type=int, default=50000,
                        help="Draws for the Bonferroni-simultaneous intervals of the pairwise contrasts.")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    runs, ck, refs = load(args.inputs)
    res = {"config": {"B": args.B, "B_bonf": args.B_bonf, "seed": args.seed, "floor": FLOOR, "suite": SUITE,
                      "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                   "scipy": __import__("scipy").__version__,
                                   "statsmodels": __import__("statsmodels").__version__}}}
    res["nominal_tiers"], res["tier_ranges_without_cql"] = nominal_tiers(runs, args.B, rng)
    res["algorithm_x_task_interaction_nominal"] = interaction_anova(runs)
    res.update(tier3_gains(runs, refs, args.B, rng))
    res["algorithm_table"] = algorithm_table(ck, args.B, rng)
    res["pairwise"] = pairwise(ck, args.B, rng)
    res["ranking"] = ranking(ck, args.B, rng)
    res["linear_fit"] = linear_fit(ck, args.B, rng)
    v = ck[ck.valid]
    res["decomposition_srr"] = decomposition(v, "srr", args.B, rng)
    res["decomposition_perturbed_score"] = decomposition(ck, "P", args.B, rng, task_blocks=False)
    res["decomposition_degradation_excess"] = decomposition(v.dropna(subset=["deg_excess"]), "deg_excess", args.B, rng, task_blocks=False)
    k = ["task", "dataset", "train_seed"]
    full = v.groupby(k).algorithm.nunique()
    common = v.set_index(k).loc[full[full == 6].index].reset_index()
    additive = "C(task) + C(algorithm) + C(dataset)"
    res["joint_anova_srr_eligible_additive"] = joint_anova(v, "srr", additive)
    res["joint_anova_srr_common_cohort_additive"] = joint_anova(common, "srr", additive)
    noc = v[v.algorithm != "CQL"]
    res["joint_anova_srr_without_cql_interactions"] = joint_anova(
        noc, "srr", "C(task)*C(algorithm) + C(task)*C(dataset) + C(algorithm)*C(dataset)")
    res["joint_anova_perturbed_score"] = joint_anova(ck, "P")
    res["sensitivity_srr"] = sensitivity(ck, refs)
    res["srr_by_nominal_level"] = srr_by_nominal_level(ck)
    if os.path.exists(os.path.join(args.inputs, "heldout_metrics.csv")):
        res["heldout"] = heldout(runs, args.inputs, args.B, rng)
    else:
        print("WARNING: no heldout_metrics.csv, the held-out analysis is skipped")
    # Separate stream so that adding this analysis leaves the draws above unchanged.
    res["gains_over_bc"] = gains_over_bc(runs, args.B, np.random.default_rng(args.seed + 1))
    # Separate stream, and more draws, for the simultaneous intervals only (MLR3-05).
    # The intervals from the B draws of pairwise() are kept as ci_bonf15_B.
    bonf = pairwise_bonferroni(ck, args.B_bonf, np.random.default_rng(args.seed + 2))
    for rec in res["pairwise"]:
        for m in ("N", "P", "SRR", "D_shift_interaction"):
            rec[m]["ci_bonf15_B"] = rec[m]["ci_bonf15"]
            rec[m]["ci_bonf15"] = bonf[rec["pair"]][m]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "paper_stats.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"wrote {os.path.join(args.out, 'paper_stats.json')}")


if __name__ == "__main__":
    main()
