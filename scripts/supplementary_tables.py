"""Generate the LaTeX tables of the supplementary material.

Every table is written to <out>/<name>.tex and included by supplementary.tex
with \\input. Sources: the MuJoCo Playground fork (models and configs), the
datasets under datasets/playground, the offline configs, and the analysis
inputs and results of scripts/paper_stats.py.

Usage (from the CORL root, with the CORL environment):
  python -m scripts.supplementary_tables --out ../overleaf_paper/supp
"""

import argparse
import glob
import json
import os

import h5py
import numpy as np
import pandas as pd
import yaml

from scripts.paper_stats import load_runs

TASKS = ["go2-flat-forward", "h1-gait-tracking", "go2-joystick-direction", "g1-joystick-direction",
         "go2-footstand", "go2-handstand", "go2-getup", "go2-getup-walk",
         "go2-push-recovery", "go2-rough-terrain"]
ENV_OF = {"go2-flat-forward": "Go2JoystickFlatTerrain", "go2-joystick-direction": "Go2JoystickFlatTerrain",
          "h1-gait-tracking": "H1JoystickGaitTracking", "g1-joystick-direction": "G1JoystickFlatTerrain",
          "go2-footstand": "Go2Footstand", "go2-handstand": "Go2Handstand", "go2-getup": "Go2Getup",
          "go2-getup-walk": "Go2GetupWalk", "go2-push-recovery": "Go2PushRecovery",
          "go2-rough-terrain": "Go2RoughCurriculum"}
ENVS = list(dict.fromkeys(ENV_OF[t] for t in TASKS))
QUALITIES = ["medium-replay", "medium", "medium-expert", "expert"]
ALGOS = ["BC", "TD3-BC", "AWAC", "IQL", "CQL", "DT"]
NAME = {"TD3-BC": "TD3+BC"}
PPO_OVERRIDES = {"num_evals": 50, "num_minibatches": 64, "num_updates_per_batch": 8,
                 "unroll_length": 40, "num_envs": 16384}


def tt(task):
    return f"\\texttt{{{task}}}"


def num(x, d=1):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def write(out, name, body):
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write(body.strip() + "\n")


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def table(caption, label, colspec, header, rows, star=False, size="\\small"):
    env = "table*" if star else "table"
    return (f"\\begin{{{env}}}[ht]\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n"
            f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            + "\n".join(r + " \\\\" for r in rows)
            + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


# ---------------------------------------------------------------- environments
def env_tables(out):
    from mujoco_playground import registry
    rows_robot, rows_reward = [], []
    for env_name in ENVS:
        env = registry.load(env_name, config_overrides={"impl": "jax"})
        m, cfg = env.mj_model, env._config
        kp = np.unique(np.round(m.actuator_gainprm[:, 0], 1))
        kd = np.unique(np.round(-m.actuator_biasprm[:, 2], 2))
        kp_s = num(kp[0]) if len(kp) == 1 else f"{num(kp.min())}--{num(kp.max())}"
        kd_s = num(kd[0], 2) if len(kd) == 1 else f"{num(kd.min(), 2)}--{num(kd.max(), 2)}"
        if np.allclose(kd, 0):
            dd = np.unique(np.round(m.dof_damping[6:], 2))
            kd_s = f"joint damping {num(dd.min(), 1)}--{num(dd.max(), 1)}"
        tasks = ", ".join(tt(t) for t in TASKS if ENV_OF[t] == env_name)
        rows_robot.append(f"{tasks} & {m.nu} & {m.body_mass.sum():.2f} & {cfg.sim_dt*1000:.0f} & "
                          f"{cfg.action_scale} & {kp_s} & {kd_s} & {cfg.episode_length}")
        scales = dict(cfg.reward_config.scales)
        terms = ", ".join(f"{k.replace('_', chr(92) + '_')} {v:g}" for k, v in scales.items() if v != 0)
        rows_reward.append(f"{tasks} & {terms}")
    write(out, "robots", table(
        "Simulated models and control of each environment. All environments act every 20~ms. "
        "$k_p$ and $k_d$ are the proportional and damping gains of the position actuators "
        "in the MJCF model (a range lists the per-joint values). Masses are those of the "
        "simulated models.",
        "tab:robots", "p{0.25\\textwidth}rrrrrrr",
        "Tasks & Joints & Mass (kg) & Sim. step (ms) & Action scale & $k_p$ & $k_d$ & Episode",
        rows_robot, star=True, size="\\footnotesize"))
    write(out, "rewards", table(
        "Non-zero reward terms and weights of each environment, as configured in the fork.",
        "tab:rewards", "p{0.25\\textwidth}p{0.68\\textwidth}", "Tasks & Terms and weights",
        rows_reward, star=True, size="\\footnotesize"))


def obs_table(out):
    rows = [
        (tt("go2-flat-forward") + ", " + tt("go2-joystick-direction") + ", " + tt("go2-push-recovery")
         + ", " + tt("go2-rough-terrain"), 48,
         "linear velocity (3)$^\\ast$, angular velocity (3), gravity (3), joint positions (12), "
         "joint velocities (12), last action (12), command (3)"),
        (tt("go2-footstand") + ", " + tt("go2-handstand"), 45,
         "as above, without the command"),
        (tt("go2-getup"), 42,
         "angular velocity (3), gravity (3), joint positions (12), joint velocities (12), last action (12)"),
        (tt("go2-getup-walk"), 48,
         "linear velocity (3)$^\\ast$, angular velocity (3), gravity (3), joint positions (12), joint "
         "velocities (12), last action (12), stood flag (1), goal in the robot frame (2)$^\\ast$"),
        (tt("g1-joystick-direction"), 103,
         "linear velocity (3)$^\\ast$, angular velocity (3), gravity (3), command (3), joint positions "
         "(29), joint velocities (29), last action (29), gait phase (4)"),
        (tt("h1-gait-tracking"), 113,
         "angular velocity (3), gravity (3), joint positions (19), joint velocities (19), last action "
         "(19), command (3), current joint velocities (19) and joint position errors relative to the motor targets (19) without the native noise, foot "
         "contacts (2), gait phase (4), gait frequency, gait type and foot height (3)"),
    ]
    write(out, "observations", table(
        "Policy observation of each task, in order, with dimensions in parentheses. The offline "
        "policies never see the privileged state. $^\\ast$Signals that a real robot would have to "
        "estimate. The perturbation suite adds sensor noise to the angular velocity, gravity, joint "
        "positions and joint velocities, and to their copies in the H1 history.",
        "tab:observations", "p{0.25\\textwidth}rp{0.62\\textwidth}", "Tasks & Size & Components",
        [f"{a} & {b} & {c}" for a, b, c in rows], star=True, size="\\footnotesize"))


# ---------------------------------------------------------------- datasets
def ppo_table(out):
    from mujoco_playground.config import locomotion_params as lp
    rows = []
    for env_name in ENVS:
        c = lp.brax_ppo_config(env_name)
        nf = c.network_factory
        steps = "2e9" if env_name.startswith(("G1", "H1")) else "1e9"
        seeds = 6 if env_name.startswith(("G1", "H1")) else 5
        rows.append(f"{env_name} & {seeds} & {steps} & {c.batch_size} & {c.learning_rate:g} & "
                    f"{c.entropy_cost:g} & {c.discounting:g} & {tuple(nf.policy_hidden_layer_sizes)}")
    write(out, "ppo", table(
        "PPO experts. Settings come from the MuJoCo Playground configuration of each environment, "
        "with the overrides shared by all tasks: 50 evaluations (checkpoints) per run, 64 minibatches, "
        "8 updates per batch, unroll length 40, 16{,}384 parallel environments and a value function "
        "that sees the policy observation. No run uses domain randomization.",
        "tab:ppo", "lrrrrrrl", "Environment & Seeds & Steps & Batch & LR & Entropy & $\\gamma$ & Policy MLP",
        rows, star=True, size="\\footnotesize"))


def dataset_tables(out, root):
    stats, refs_rows, sel_rows = [], [], []
    for t in TASKS:
        for q in QUALITIES:
            d = f"{root}/{t}/{q}-v0"
            me, mi = json.load(open(f"{d}/metadata.json")), json.load(open(f"{d}/data/metadata.json"))
            acc = ""
            if q == "expert":
                acc = f"{100 * mi['total_episodes'] / me['num_episodes']:.1f}"
            stats.append((t, q, mi["total_steps"], mi["total_episodes"], me["normalized_score_mean"], acc))
        e = json.load(open(f"{root}/{t}/expert-v0/metadata.json"))
        m = json.load(open(f"{root}/{t}/medium-v0/metadata.json"))
        refs_rows.append(f"{tt(t)} & {num(e['return_min'], 2)} & {num(e['return_expert'], 2)}")
        lab = lambda p: "/".join(str(p).rstrip("/").split("/")[-3::2]) if p else "--"
        sel_rows.append(f"{tt(t)} & {e['num_checkpoints']} & \\texttt{{{lab(e['policy_checkpoint'])}}} & "
                        f"\\texttt{{{lab(m.get('medium_checkpoint'))}}}")
    # overlap between medium-expert and expert episodes (identical first three observations)
    overlap = {}
    for t in TASKS:
        def sig(path):
            f = h5py.File(path, "r")
            return {np.round(np.asarray(f[k]["observations"][:3]).ravel(), 5).tobytes() for k in f.keys()}
        se = sig(f"{root}/{t}/expert-v0/data/main_data.hdf5")
        sm = sig(f"{root}/{t}/medium-expert-v0/data/main_data.hdf5")
        overlap[t] = 100 * len(se & sm) / len(sm)
    rows = []
    for t, q, steps, eps, score, acc in stats:
        ov = f"{overlap[t]:.1f}" if q == "medium-expert" else ""
        rows.append(f"{tt(t) if q == QUALITIES[0] else ''} & {q} & {steps:,} & {eps:,} & {num(score)} & {acc} & {ov}".replace(",", "{,}"))
    write(out, "datasets", table(
        "Dataset statistics. Score is the mean normalized score of the episodes in the dataset. "
        "Acc.\\ is the share of expert rollouts kept by the 90th-percentile filter. Overlap is the share "
        "of medium-expert episodes identical to an expert-dataset episode (same first three "
        "observations).",
        "tab:datasets", "llrrrrr", "Task & Quality & Transitions & Episodes & Score & Acc.\\ (\\%) & Overlap (\\%)",
        rows, size="\\scriptsize"))
    write(out, "refs", table(
        "Normalization references: mean return of the weakest and of the best checkpoint of the pool, "
        "each evaluated on 20 episodes with the commands of the environment.",
        "tab:refs", "lrr", "Task & $R_{\\min}$ & $R_{\\max}$", refs_rows))
    write(out, "checkpoints", table(
        "Size of the checkpoint pool of each task and the checkpoints selected as expert and medium "
        "policies (training run and environment step).",
        "tab:checkpoints", "lrll", "Task & Pool & Expert & Medium", sel_rows, star=True, size="\\scriptsize"))
    # phases of go2-getup-walk
    rows = []
    for q in QUALITIES:
        f = h5py.File(f"{root}/go2-getup-walk/{q}-v0/data/main_data.hdf5", "r")
        st = [np.asarray(f[k]["infos"]["stood"]).max() > 0.5 for k in f.keys()]
        ar = [np.asarray(f[k]["infos"]["arrived"]).max() > 0.5 for k in f.keys()]
        rows.append(f"{q} & {len(st)} & {100*np.mean(st):.1f} & {100*np.mean(ar):.1f}")
    write(out, "phases", table(
        "Phase coverage of \\texttt{go2-getup-walk}: share of episodes in which the robot stands up "
        "and in which it reaches the goal.",
        "tab:phases", "lrrr", "Quality & Episodes & Stands up (\\%) & Reaches goal (\\%)", rows))
    # early termination: share of episodes that end before the time limit, and the mean normalized
    # score of complete and of early-ended episodes
    rows = []
    for t in TASKS:
        e = json.load(open(f"{root}/{t}/expert-v0/metadata.json"))
        lo, hi = e["return_min"], e["return_expert"]
        for q in QUALITIES:
            f = h5py.File(f"{root}/{t}/{q}-v0/data/main_data.hdf5", "r")
            term = np.array([bool(f[k]["terminations"][-1]) for k in f.keys()])
            score = np.array([100 * (float(np.sum(f[k]["rewards"])) - lo) / (hi - lo) for k in f.keys()])
            full = num(score[~term].mean()) if (~term).any() else "--"
            early = num(score[term].mean()) if term.any() else "--"
            rows.append(f"{tt(t) if q == QUALITIES[0] else ''} & {q} & {100 * term.mean():.0f} & {full} & {early}")
    write(out, "termination", table(
        "Episodes that end before the time limit (Early, \\%), because the robot meets a termination "
        "condition of its task, such as a fall, and the mean normalized score of the complete and of the "
        "early-ended episodes. The two getup tasks do not end an episode on a fall.",
        "tab:termination", "llrrr", "Task & Quality & Early (\\%) & Complete & Early-ended", rows,
        size="\\scriptsize"))


# ---------------------------------------------------------------- algorithms
def hyper_tables(out):
    skip = {"project", "name", "group", "checkpoints_path", "device", "load_model", "env", "env_name",
            "seed", "train_seed", "eval_seed", "test_seed", "deterministic_torch", "num_workers"}
    blocks = []
    for algo, key in [("BC", "bc"), ("TD3+BC", "td3_bc"), ("AWAC", "awac"), ("IQL", "iql"),
                      ("CQL", "cql"), ("Decision Transformer", "dt")]:
        ours = yaml.safe_load(open(f"configs/offline/{key}/base.yaml"))
        esc = lambda x: str(x).replace("_", "\\_")
        rows = [f"{esc(k)} & {esc(ours[k])}" for k in sorted(set(ours) - skip)]
        blocks.append(table(
            f"{algo}: configuration used for all tasks.",
            f"tab:hp-{key}", "ll", "Parameter & Value", rows, size="\\scriptsize"))
    write(out, "hyperparameters", "\n".join(blocks))
    reg = yaml.safe_load(open("configs/offline/_datasets.yaml"))["tasks"]
    refs = pd.read_csv("analysis/paper_stats/inputs/dataset_refs.csv")
    rmax = refs[refs.dataset == "expert"].set_index("task").r_max
    rows = [f"{tt(t)} & {reg[t]['dt_target_returns'][0]} & {reg[t]['dt_target_returns'][1]} & {num(rmax[t], 2)}"
            for t in TASKS]
    write(out, "dt_targets", table(
        "Target returns of the Decision Transformer. All results use the first target; the second "
        "was logged but is not used in the paper.",
        "tab:dt-targets", "lrrr", "Task & Target used & Second target & $R_{\\max}$", rows))


def cql_table(out):
    r = load_runs("analysis/paper_stats/inputs")
    c = r[r.algorithm == "CQL"].copy()
    c["at_clip"] = c.cql_alpha_prime >= 1e6 - 1
    g = c.groupby("dataset").agg(runs=("run", "size"), clip=("at_clip", "sum"), score=("score", "mean"))
    rows = [f"{q} & {int(g.loc[q, 'runs'])} & {int(g.loc[q, 'clip'])} & {num(g.loc[q, 'score'])}" for q in QUALITIES]
    rows.append(f"all & {len(c)} & {int(c.at_clip.sum())} & {num(c.score.mean())}")
    write(out, "cql", table(
        "CQL runs whose Lagrange multiplier $\\alpha'$ ended at its upper clip of $10^6$, by dataset "
        "quality, with the mean nominal score of the runs.",
        "tab:cql", "lrrr", "Quality & Runs & At clip & Mean score", rows))


def compute_table(out):
    r = pd.read_csv("analysis/paper_stats/inputs/runs.csv")
    g = r.groupby("algorithm").runtime_h.agg(["size", "median", "sum"])
    rows = [f"{NAME.get(a, a)} & {int(g.loc[a, 'size'])} & {g.loc[a, 'median']:.1f} & {g.loc[a, 'sum']:.0f}" for a in ALGOS]
    gpus = r.gpu.value_counts()
    cap = ("Wall-clock time of offline training per run, from W\\&B. Runs used "
           + ", ".join(f"{k.replace('NVIDIA ', '')} ({v})" for k, v in gpus.items()) + ".")
    write(out, "compute", table(cap, "tab:compute", "lrrr", "Algorithm & Runs & Median (h) & Total (h)", rows))


# ---------------------------------------------------------------- results
def results_tables(out):
    res = json.load(open("analysis/paper_stats/results/paper_stats.json"))
    runs = load_runs("analysis/paper_stats/inputs")
    m = pd.read_csv("analysis/paper_stats/inputs/sim2real_metrics.csv")
    ck = m[(m.suite == "humanoid_gym_relative_v2") & (m.checkpoint != "DT-H1JoystickGaitTracking-f12bb1c4")].copy()
    ck[["nominal_score", "score"]] = ck[["nominal_score", "score"]].fillna(0.0)
    ck["valid"] = ck.nominal_score >= 0.05
    ck["srr"] = np.where(ck.valid, ck.score / ck.nominal_score, np.nan)
    u = pd.read_csv("analysis/paper_stats/inputs/unpaired_srr_metrics.csv")
    ck = ck.merge(u[["checkpoint", "failure_rate_50_nominal", "failure_rate_50"]], on="checkpoint", how="left")
    refs = pd.read_csv("analysis/paper_stats/inputs/dataset_refs.csv")

    # F.1 nominal score per cell
    g = runs.groupby(["task", "dataset", "algorithm"]).score.agg(["mean", "std"])
    rows = []
    for t in TASKS:
        for q in QUALITIES:
            cells = [f"{num(g.loc[(t, q, a), 'mean'])}$\\pm${g.loc[(t, q, a), 'std']:.1f}" for a in ALGOS]
            rows.append(f"{tt(t) if q == QUALITIES[0] else ''} & {q} & " + " & ".join(cells))
    write(out, "cells", table(
        "Nominal score of every task, dataset and algorithm: mean $\\pm$ standard deviation over the "
        "three training runs.",
        "tab:cells", "ll" + "r" * 6, "Task & Quality & " + " & ".join(NAME.get(a, a) for a in ALGOS),
        rows, star=True, size="\\scriptsize"))

    # F.2 N, P, SRR per task x algorithm
    rows = []
    for t in TASKS:
        cells = []
        for a in ALGOS:
            x = ck[(ck.task == t) & (ck.algorithm == a)]
            v = x[x.valid]
            srr = f"{v.srr.mean():.2f}" if len(v) else "--"
            cells.append(f"{x.nominal_score.mean():.2f}/{x.score.mean():.2f}/{srr} ({len(v)})")
        rows.append(f"{tt(t)} & " + " & ".join(cells))
    write(out, "srr_task", table(
        "Nominal score / perturbed score / SRR per task and algorithm, with the number of eligible "
        "checkpoints ($N \\geq 0.05$) out of 12 in parentheses.",
        "tab:srr-task", "l" + "r" * 6, "Task & " + " & ".join(NAME.get(a, a) for a in ALGOS),
        rows, star=True, size="\\scriptsize"))

    # F.3 pairwise
    blocks = []
    for k, title in [("N", "nominal score"), ("P", "perturbed score"), ("SRR", "SRR")]:
        rows = []
        for p in res["pairwise"]:
            x = p[k]
            pair = p["pair"].replace("TD3-BC", "TD3+BC").replace(" vs ", " -- ")
            rows.append(f"{pair} & {num(x['diff'], 3)} & [{num(x['ci'][0], 3)}, {num(x['ci'][1], 3)}] & "
                        f"[{num(x['ci_bonf15'][0], 3)}, {num(x['ci_bonf15'][1], 3)}] & {x['n_cells']} & "
                        f"{x['cells_a_better']} & {x['W']:.0f} & {pfmt(x['p'])} & {pfmt(x['p_holm45'])}")
        blocks.append(table(
            f"Pairwise comparisons on the {title} (first minus second), over the cells used by the contrast. "
            "CI: pointwise 95\\% bootstrap interval. Bonf.: simultaneous interval for the 15 pairs, "
            f"from {res['config']['B_bonf']:,} separate draws. "
            "Better: cells where the first method has the higher mean. $W$ and $p$: two-sided Wilcoxon "
            "signed-rank test on the cell differences; Holm: adjusted over the 45 tests of the three tables.",
            f"tab:pairwise-{k}", "lrrrrrrrr",
            "Pair & Diff. & CI & Bonf. & Cells & Better & $W$ & $p$ & Holm", rows, star=True, size="\\scriptsize"))
    rows = []
    for p in res["pairwise"]:
        d = p["D_shift_interaction"]
        pair = p["pair"].replace("TD3-BC", "TD3+BC").replace(" vs ", " -- ")
        rows.append(f"{pair} & {num(d['diff'], 3)} & [{num(d['ci'][0], 3)}, {num(d['ci'][1], 3)}] & "
                    f"[{num(d['ci_bonf15'][0], 3)}, {num(d['ci_bonf15'][1], 3)}]")
    blocks.append(table(
        "Shift interaction $D = (P_a - N_a) - (P_b - N_b)$: a positive value means that the first method "
        "loses less under perturbation than the second.",
        "tab:pairwise-D", "lrrr", "Pair & $D$ & CI & Bonf.", rows, size="\\scriptsize"))
    write(out, "pairwise", "\n".join(blocks))

    # F.4 joint models
    blocks = []
    for key, title in [("joint_anova_srr_eligible_additive", "SRR, eligible checkpoints, additive model"),
                       ("joint_anova_srr_common_cohort_additive", "SRR, common cohort, additive model"),
                       ("joint_anova_srr_without_cql_interactions", "SRR, without CQL, with interactions"),
                       ("joint_anova_perturbed_score", "Perturbed score, all checkpoints, with interactions")]:
        rows = []
        meta = res[key]["_model"]
        for term, v in res[key].items():
            if term.startswith("_"):
                continue
            f = "" if v["F"] is None else f"{v['F']:.2f}"
            pv = "" if v["p"] is None else (f"{v['p']:.1e}" if v["p"] < 0.001 else f"{v['p']:.3f}")
            rows.append(f"{term.replace('C(', '').replace(')', '').replace(':', ' $\\times$ ')} & {v['df']} & "
                        f"{100 * v['ss_share']:.1f} & {f} & {pv}")
        blocks.append(table(f"Type II decomposition: {title} ($n = {meta['n']}$, $R^2 = {meta['r2']:.2f}$). "
                            "Share is the Type II sum of squares of the term normalized by the total sum of "
                            "squares. In an unbalanced design these shares are not an additive partition of "
                            f"the variance: here they sum to {meta['sum_of_shares']:.3f}.",
                            f"tab:anova-{key}", "lrrrr", "Term & df & Share (\\%) & $F$ & $p$", rows,
                            size="\\scriptsize"))
    write(out, "anova", "\n".join(blocks))

    # F.5 leave-one-task-out
    v = ck[ck.valid].copy()

    def eta2(c, gr):
        mm = c.mean()
        return sum(len(x) * (x.mean() - mm) ** 2 for _, x in c.groupby(gr)) / ((c - mm) ** 2).sum()
    rows = []
    for t in TASKS:
        d = v[v.task != t]
        rows.append(f"{tt(t)} & {len(d)} & {eta2(d.srr, d.task):.2f} & {eta2(d.srr, d.algorithm):.3f}")
    write(out, "loto", table(
        "Decomposition of the SRR when one task is left out. The task-block bootstrap interval of the "
        "difference $\\eta^2_\\text{task} - \\eta^2_\\text{alg}$ is "
        f"[{res['decomposition_srr']['difference_ci_task_blocks'][0]:.2f}, "
        f"{res['decomposition_srr']['difference_ci_task_blocks'][1]:.2f}].",
        "tab:loto", "lrrr", "Task left out & $n$ & $\\eta^2_\\text{task}$ & $\\eta^2_\\text{alg}$", rows))

    # F.6 SRR by nominal level
    bt = res["srr_by_nominal_level"]["by_task"]
    bands = ["0.05-0.2", "0.2-0.5", ">=0.5"]
    rows = []
    for t in TASKS:
        cells = []
        for b in bands:
            x = bt.get(f"{t}|{b}")
            cells.append(f"{x['mean']:.2f} ({x['n']})" if x else "--")
        rows.append(f"{tt(t)} & " + " & ".join(cells))
    write(out, "srr_level", table(
        "Mean SRR per task and band of nominal score, with the number of checkpoints in parentheses.",
        "tab:srr-level", "lrrr", "Task & $0.05 \\le N < 0.2$ & $0.2 \\le N < 0.5$ & $N \\ge 0.5$", rows))

    # F.7 improvement over the data
    r = runs.merge(refs[["task", "dataset", "data_mean"]], on=["task", "dataset"])
    r["gain"] = r.score - r.data_mean
    gg = r.groupby(["task", "dataset", "algorithm"]).gain.mean()
    rows = []
    for t in TASKS:
        for q in QUALITIES:
            dm = refs[(refs.task == t) & (refs.dataset == q)].data_mean.iloc[0]
            rows.append(f"{tt(t) if q == QUALITIES[0] else ''} & {q} & {num(dm)} & "
                        + " & ".join(num(gg.loc[(t, q, a)]) for a in ALGOS))
    write(out, "gains", table(
        "Gain over the data: mean nominal score of the three runs minus the mean normalized score of the "
        "dataset (Data). The medium datasets come from stochastic policies, while every learned policy "
        "acts deterministically at evaluation, so this comparison favors every method, BC included.",
        "tab:gains", "llr" + "r" * 6, "Task & Quality & Data & " + " & ".join(NAME.get(a, a) for a in ALGOS),
        rows, star=True, size="\\scriptsize"))

    # F.7b gain over BC on the same task and dataset
    cell = runs.groupby(["task", "dataset", "algorithm"]).score.mean()
    rows = []
    for t in TASKS:
        for q in QUALITIES:
            bc = cell.loc[(t, q, "BC")]
            rows.append(f"{tt(t) if q == QUALITIES[0] else ''} & {q} & {num(bc)} & "
                        + " & ".join(num(cell.loc[(t, q, a)] - bc) for a in ALGOS if a != "BC"))
    write(out, "gains_bc", table(
        "Gain over BC: mean nominal score of the three runs minus the mean score of BC trained on the "
        "same task and dataset (BC). The paper uses this comparator for the two-leg balance tier.",
        "tab:gains-bc", "llr" + "r" * 5, "Task & Quality & BC & " + " & ".join(NAME.get(a, a) for a in ALGOS if a != "BC"),
        rows, star=True, size="\\scriptsize"))

    # F.8 relative-degradation rate
    rows = []
    for t in TASKS:
        cells = []
        for a in ALGOS:
            x = v[(v.task == t) & (v.algorithm == a)]
            cells.append(f"{x.failure_rate_50_nominal.mean():.0f}/{x.failure_rate_50.mean():.0f}" if len(x) else "--")
        rows.append(f"{tt(t)} & " + " & ".join(cells))
    write(out, "degradation", table(
        "Relative-degradation rate (\\%), nominal / perturbed, per task and algorithm, over the "
        "eligible checkpoints.",
        "tab:degradation", "l" + "r" * 6, "Task & " + " & ".join(NAME.get(a, a) for a in ALGOS),
        rows, star=True, size="\\scriptsize"))

    # F.9 held-out, all datasets
    h = res["heldout"]
    rows = []
    for t in ["go2-push-recovery", "go2-rough-terrain"]:
        for q in QUALITIES:
            cells = []
            for a in ALGOS:
                x = h[f"{a}|{t}|{q}"]
                ratio = "--" if x["ratio"] is None else f"{x['ratio']:.2f}"
                cells.append(f"{num(x['nominal'])}/{num(x['heldout'])}/{ratio}")
            rows.append(f"{tt(t) if q == QUALITIES[0] else ''} & {q} & " + " & ".join(cells))
    write(out, "heldout", table(
        "Held-out regime of the disturbed-locomotion tier: nominal score / held-out score / retention (ratio of "
        "means, omitted when the nominal mean is below 5).",
        "tab:heldout-all", "ll" + "r" * 6, "Task & Quality & " + " & ".join(NAME.get(a, a) for a in ALGOS),
        rows, star=True, size="\\scriptsize"))


def extra_tables(out):
    res = json.load(open("analysis/paper_stats/results/paper_stats.json"))
    # eligible checkpoints per cell
    e = res["algorithm_table"]["_eligible_per_cell"]
    rows = []
    for t in TASKS:
        for q in QUALITIES:
            rows.append(f"{tt(t) if q == QUALITIES[0] else ''} & {q} & " + " & ".join(str(e[f"{t}|{q}|{a}"]) for a in ALGOS))
    write(out, "eligible", table(
        "Number of eligible checkpoints ($N \\geq 0.05$) out of three in each cell. Cells with a single "
        "eligible checkpoint contribute no training variation to the bootstrap intervals of the SRR.",
        "tab:eligible", "ll" + "r" * 6, "Task & Quality & " + " & ".join(NAME.get(a, a) for a in ALGOS),
        rows, star=True, size="\\scriptsize"))
    # association: leave-one-task-out RMSE and Spearman per algorithm
    lf, rk = res["linear_fit"], res["ranking"]
    rows = [f"{tt(t)} & {lf['rmse_leave_one_task_out'][t]:.3f}" for t in TASKS]
    body = table(
        "Prediction error of the pooled linear fit $P = aN + b$ when the task is left out of the fit "
        f"(RMSE on the 0--1 scale). The stacked out-of-task $R^2$ is {lf['r2_leave_one_task_out']:.2f}, "
        "with the denominator taken around the global mean; it is not an average of per-task $R^2$.",
        "tab:loto-rmse", "lr", "Task left out & RMSE", rows)
    rows = [f"{NAME.get(a, a)} & {rk['spearman_per_algorithm'][a]:.2f}" for a in ALGOS]
    body += table(
        "Spearman correlation between nominal and perturbed scores over the 120 checkpoints of each "
        "algorithm.", "tab:rho-algo", "lr", "Algorithm & $\\rho$", rows)
    write(out, "association", body)
    # discrete impulse of the push pulses
    dt = 0.02
    rows = []
    for T in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]:
        n = int(np.round(T / dt))
        ratio = 0.5 * dt / T * np.sin(np.pi * np.arange(n) * dt / T).sum() * np.pi
        regime = "held-out" if T < 0.05 else ("both" if T <= 0.08 else "collection")
        rows.append(f"{T*1000:.0f} & {regime} & {n} & {ratio:.2f}")
    write(out, "impulse", table(
        "Discrete impulse of a push. The force is held constant over each 20~ms control step and the "
        "duration is rounded to whole steps. Ratio is the discrete impulse $\\sum_t F_t \\Delta t$ over the "
        "continuous-time impulse $Mk/\\pi$; it does not depend on $M$ or $k$.",
        "tab:impulse", "lrrr", "Duration (ms) & Regime & Steps & Ratio", rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../overleaf_paper/supp")
    parser.add_argument("--datasets", default="datasets/playground")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    env_tables(args.out)
    obs_table(args.out)
    ppo_table(args.out)
    dataset_tables(args.out, args.datasets)
    hyper_tables(args.out)
    cql_table(args.out)
    compute_table(args.out)
    results_tables(args.out)
    extra_tables(args.out)
    print("tables:", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
