#!/usr/bin/env python
"""Prune local run artifacts whose W&B run no longer exists.

W&B is the source of truth: after deleting duplicate runs there, this finds every
local artifact that belongs to a run W&B no longer has and, with --apply, moves it
to a trash folder (nothing is deleted outright; `rm -rf` the trash when satisfied).

A run is matched by its name (<ALGO>-<Env>-<hash>, the wandb run name) for
  checkpoints/<ALGO>/<name>/, videos/<name>/,
  logs/compare/metrics/<name>[@step].json, logs/compare/algorithms/<name>.txt
and by its id for the local wandb/run-*-<id>/ folders.

It also reports .done_runs markers whose (group, seed) has no finished run left in
W&B: run_offline_matrix.sh would skip retraining those. They are only moved with
--prune-markers.

Usage:
  python scripts/wandb_sync_local.py                  # dry run: report only
  python scripts/wandb_sync_local.py --apply          # move orphans to the trash
  python scripts/wandb_sync_local.py --offline        # reuse the last --cache
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = "akcit-offlinerl/Offline-Benchmark"
ALGO_DIRS = ["AWAC", "BC", "CQL", "DT", "IQL", "TD3-BC"]
RUN_NAME = re.compile(r"^(?:%s)-[A-Za-z0-9]+-[0-9a-f]{8}$" % "|".join(ALGO_DIRS))
WANDB_DIR = re.compile(r"^(?:offline-)?run-\d{8}_\d{6}-(.+)$")
# Anything touched this recently may belong to a run still in flight.
RECENT_SECONDS = 6 * 3600


RECOVERED_LOG = re.compile(r"/([a-z0-9_]+)-[^/]*-seed(\d+)\.log$")


def run_seed(meta: dict) -> str | None:
   """The matrix --seed of a run, also for runs resumed by scripts.recover_proxy.

   A resumed run's metadata is overwritten with the recovery command, whose only
   argument is the matrix log <group>-seed<N>.log. run.config["seed"] is not used:
   DT stored its base-config train_seed there instead of --seed.
   """
   args = meta.get("args") or []
   seed = next((args[i + 1] for i, a in enumerate(args[:-1]) if a == "--seed"), None)
   if seed is None and meta.get("program") == "-m scripts.recover_proxy" and args:
      m = RECOVERED_LOG.search(args[0])
      seed = m.group(2) if m else None
   return seed


def fetch_runs(project: str) -> list[dict]:
   import wandb

   api = wandb.Api(timeout=60)
   out = []
   for run in api.runs(project, per_page=500):
      # DT's seed-10 runs were relabeled as matrix seed 0 (config.matrix_seed).
      seed = run.config.get("matrix_seed", run_seed(run.metadata or {}))
      seed = None if seed is None else str(seed)
      out.append({
         "id": run.id, "name": run.name, "group": run.group,
         "state": run.state, "seed": seed,
      })
   return out


def _mtime(path: Path) -> float:
   # lstat: wandb run folders hold symlinks whose targets are gone.
   if not path.is_dir():
      return path.lstat().st_mtime
   return max((p.lstat().st_mtime for p in path.rglob("*")), default=path.lstat().st_mtime)


def local_artifacts() -> dict[str, list[Path]]:
   """Local paths keyed by run name (checkpoints, videos, SRR outputs)."""
   by_name: dict[str, list[Path]] = collections.defaultdict(list)
   for algo in ALGO_DIRS:
      for d in (REPO / "checkpoints" / algo).glob(f"{algo}-*"):
         if d.is_dir() and RUN_NAME.match(d.name):
            by_name[d.name].append(d)
   for d in (REPO / "videos").glob("*"):
      if d.is_dir() and RUN_NAME.match(d.name):
         by_name[d.name].append(d)
   for f in (REPO / "logs/compare/metrics").glob("*.json"):
      name = f.stem.split("@")[0]
      if RUN_NAME.match(name):
         by_name[name].append(f)
   for f in (REPO / "logs/compare/algorithms").glob("*.txt"):
      if RUN_NAME.match(f.stem):
         by_name[f.stem].append(f)
   return by_name


def local_wandb_dirs() -> dict[str, Path]:
   out = {}
   for d in (REPO / "wandb").iterdir():
      m = WANDB_DIR.match(d.name)
      if d.is_dir() and m:
         out[m.group(1)] = d
   return out


def main() -> int:
   ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
   ap.add_argument("--project", default=PROJECT)
   ap.add_argument("--cache", type=Path, default=REPO / "logs/wandb_runs_sync.json")
   ap.add_argument("--offline", action="store_true", help="reuse --cache instead of the API")
   ap.add_argument("--apply", action="store_true", help="move orphans to the trash")
   ap.add_argument("--prune-markers", action="store_true",
                   help="also move .done_runs markers with no finished W&B run")
   ap.add_argument("--trash", type=Path,
                   default=REPO / ".trash" / time.strftime("sync-%Y%m%d-%H%M%S"))
   args = ap.parse_args()

   if args.offline:
      runs = json.loads(args.cache.read_text())
   else:
      runs = fetch_runs(args.project)
      args.cache.parent.mkdir(parents=True, exist_ok=True)
      args.cache.write_text(json.dumps(runs, indent=1))
   # An empty or failed listing would flag every local run as an orphan.
   if len(runs) < 50:
      print(f"ERRO: só {len(runs)} runs vieram do W&B; abortando por segurança.", file=sys.stderr)
      return 1

   names = {r["name"] for r in runs}
   ids = {r["id"] for r in runs}
   now = time.time()

   orphans: list[Path] = []
   recent: list[Path] = []
   by_algo = collections.Counter()
   for name, paths in sorted(local_artifacts().items()):
      if name in names:
         continue
      for p in paths:
         (recent if now - _mtime(p) < RECENT_SECONDS else orphans).append(p)
      by_algo[name.split("-")[0] if not name.startswith("TD3-BC") else "TD3-BC"] += 1

   for run_id, d in sorted(local_wandb_dirs().items()):
      if run_id not in ids:
         (recent if now - _mtime(d) < RECENT_SECONDS else orphans).append(d)

   finished = {(r["group"], str(r["seed"])) for r in runs if r["state"] == "finished"}
   stale_markers = []
   for m in sorted((REPO / ".done_runs").glob("*.done")):
      mm = re.match(r"^(.+)-seed(\d+)$", m.stem)
      if mm and (mm.group(1), mm.group(2)) not in finished:
         stale_markers.append(m)

   local_names = set(local_artifacts())
   missing_local = sorted(r["name"] for r in runs
                          if r["state"] == "finished" and r["name"] not in local_names)

   print(f"W&B {args.project}: {len(runs)} runs "
         f"({collections.Counter(r['state'] for r in runs).most_common()})")
   print(f"Runs locais sem registro no W&B: {sum(by_algo.values())} {dict(sorted(by_algo.items()))}")
   print(f"  artefatos a mover: {len(orphans)}"
         f" (checkpoints {sum('checkpoints' in p.parts for p in orphans)},"
         f" wandb/ {sum(p.parent.name == 'wandb' for p in orphans)},"
         f" métricas/logs {sum('compare' in p.parts for p in orphans)},"
         f" vídeos {sum('videos' in p.parts for p in orphans)})")
   if recent:
      print(f"  mantidos por terem sido modificados nas últimas {RECENT_SECONDS // 3600} h: {len(recent)}")
      for p in recent:
         print(f"    {p.relative_to(REPO)}")
   print(f"Marcadores .done_runs sem run finished no W&B: {len(stale_markers)}"
         + ("" if args.prune_markers else " (só relatório; use --prune-markers)"))
   for m in stale_markers:
      print(f"    {m.name}")
   print(f"Runs finished no W&B sem nenhum artefato local: {len(missing_local)}")
   for n in missing_local[:20]:
      print(f"    {n}")
   if len(missing_local) > 20:
      print(f"    ... (+{len(missing_local) - 20})")

   listing = REPO / "logs/wandb_sync_orphans.txt"
   listing.write_text("".join(f"{p.relative_to(REPO)}\n" for p in orphans))
   print(f"\nLista completa: {listing.relative_to(REPO)}")

   if not args.apply:
      print("Dry run: nada foi movido. Rode com --apply para mover para a lixeira.")
      return 0

   to_move = orphans + (stale_markers if args.prune_markers else [])
   for p in to_move:
      dest = args.trash / p.relative_to(REPO)
      dest.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(str(p), str(dest))
   print(f"Movidos {len(to_move)} itens para {args.trash}")
   return 0


if __name__ == "__main__":
   sys.exit(main())
