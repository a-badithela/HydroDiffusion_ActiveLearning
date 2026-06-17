"""
round_comparison.py
-------------------
Utilities for comparing model performance across active learning rounds.

Functions
---------
collect_round_metrics   Load predictions for multiple AL rounds, return per-basin NSE/KGE.
get_newly_added_sites   Diff successive split CSVs to find when each basin was promoted.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from analysis.performance_functions import nse, kge


def collect_round_metrics(
    runs_dir: Path,
    run_date: str,
    seeds: List[int],
    rounds: List[int],
    split_df: pd.DataFrame,
    ll_df: pd.DataFrame,
    acq_fn: str | None = None,
) -> pd.DataFrame:
    """Compute per-basin NSE and KGE for each AL round.

    Discovers run directories automatically via glob — no need to specify model,
    split tag, or stride. For each round, loads predictions.npz from all matching
    seed directories, averages preds across seeds, then computes NSE and KGE per basin.

    Parameters
    ----------
    runs_dir    : Path to the top-level runs/ directory.
    run_date    : Date identifier for the experiment. Supports two layouts:
                  - Nested: runs/<run_date>/al-r{n}-.../ (e.g. '20260615')
                  - Flat:   runs/<run_date>_al-r{n}-.../ (e.g. '20260526')
    seeds       : List of seed integers (e.g. [3407, 3408, 3409, 3410, 3411]).
    rounds      : List of AL round indices to compare (e.g. [0, 1, 2]).
    split_df    : DataFrame with columns ['gauge_id', 'split'] (seed/eval/sample labels).
    ll_df       : DataFrame with columns ['gauge_id', 'lat', 'lon'].
    acq_fn      : Optional acquisition function filter (e.g. 'epig', 'random').
                  Only needed when multiple experiments share the same date folder.

    Returns
    -------
    pd.DataFrame with columns: [round, gauge_id, nse, kge, split, lat, lon]
    """
    runs_dir = Path(runs_dir)
    rows = []

    def _find_run_dir(r: int, seed: int) -> Path | None:
        """Glob for the run directory matching round r and seed, optionally filtered by acq_fn."""
        candidates = []
        # Nested layout: runs/<run_date>/al-r{r}-*_seed{seed}/
        nested_parent = runs_dir / run_date
        if nested_parent.is_dir():
            candidates.extend(nested_parent.glob(f"al-r{r}-*_seed{seed}"))
        # Flat layout: runs/<run_date>*al-r{r}-*_seed{seed}/
        candidates.extend(runs_dir.glob(f"{run_date}*al-r{r}-*_seed{seed}"))

        if acq_fn is not None:
            candidates = [c for c in candidates if f"-{acq_fn}_" in c.name]

        if len(candidates) > 1:
            print(f"[round_comparison] Warning: {len(candidates)} matches for round={r} seed={seed}, using first: {candidates[0].name}")
        return candidates[0] if candidates else None

    for r in rounds:
        preds_list = []
        obs_ref = None
        basins_ref = None

        for seed in seeds:
            run_dir = _find_run_dir(r, seed)
            if run_dir is None:
                continue
            pred_path = run_dir / "predictions.npz"
            if not pred_path.exists():
                continue
            d = np.load(pred_path, allow_pickle=True)
            preds_list.append(d["preds"].astype(np.float32))
            if obs_ref is None:
                obs_ref = d["obs"].astype(np.float32)
                basins_ref = d["basins"].astype(str)

        if not preds_list:
            print(f"[round_comparison] Round {r}: no predictions found, skipping.")
            continue

        mean_preds = np.stack(preds_list, axis=0).mean(axis=0)  # (N, H)
        print(f"[round_comparison] Round {r}: averaged {len(preds_list)} seed(s), {len(np.unique(basins_ref))} basins.")

        for gid in np.unique(basins_ref):
            mask = basins_ref == gid
            df_bp = pd.DataFrame({
                "qobs": obs_ref[mask].ravel(),
                "qsim": mean_preds[mask].ravel(),
            })
            kge_val, *_ = kge(df_bp)
            rows.append({
                "round": r,
                "gauge_id": gid,
                "nse": nse(df_bp),
                "kge": kge_val,
            })

    if not rows:
        return pd.DataFrame(columns=["round", "gauge_id", "nse", "kge", "split", "lat", "lon"])

    metrics_df = pd.DataFrame(rows)
    split_df = split_df.copy()
    split_df["gauge_id"] = split_df["gauge_id"].astype(str).str.zfill(8)
    ll_df = ll_df.copy()
    ll_df["gauge_id"] = ll_df["gauge_id"].astype(str).str.zfill(8)
    metrics_df["gauge_id"] = metrics_df["gauge_id"].astype(str).str.zfill(8)

    metrics_df = (
        metrics_df
        .merge(split_df[["gauge_id", "split"]], on="gauge_id", how="left")
        .merge(ll_df[["gauge_id", "lat", "lon"]], on="gauge_id", how="left")
    )
    return metrics_df


def get_newly_added_sites(
    orchestration_dir: Path,
    max_round: int,
) -> pd.DataFrame:
    """Determine which AL round each basin was promoted to 'seed'.

    Reads round{n}/split.csv for n in 0 … max_round. Basins already marked
    'seed' in round 0 are labelled round_added=0 (the initial seed pool).
    For each subsequent round, basins newly flipped to 'seed' get that round's index.
    Basins that remain 'sample' or 'eval' throughout receive round_added=NaN.

    Parameters
    ----------
    orchestration_dir : Path to the orchestration directory that contains round0/, round1/, …
    max_round         : Highest round index to inspect.

    Returns
    -------
    pd.DataFrame with columns: [gauge_id, round_added]
        Only rows where round_added is not NaN (i.e. basins that are/became 'seed').
    """
    orchestration_dir = Path(orchestration_dir)
    splits = {}
    for r in range(max_round + 1):
        csv_path = orchestration_dir / f"round{r}" / "split.csv"
        if not csv_path.exists():
            print(f"[round_comparison] {csv_path} not found, stopping at round {r - 1}.")
            break
        df = pd.read_csv(csv_path, dtype={"gauge_id": str})
        df["gauge_id"] = df["gauge_id"].str.zfill(8)
        splits[r] = df.set_index("gauge_id")["split"]

    if not splits:
        return pd.DataFrame(columns=["gauge_id", "round_added"])

    all_basins = splits[0].index
    round_added = {}

    # Initial seed pool at round 0
    for gid in all_basins:
        if splits[0].get(gid) == "seed":
            round_added[gid] = 0

    # Promoted basins: seed in round r but not in round r-1
    for r in range(1, max(splits.keys()) + 1):
        if r not in splits or r - 1 not in splits:
            continue
        prev, curr = splits[r - 1], splits[r]
        for gid in all_basins:
            if gid not in round_added and curr.get(gid) == "seed":
                round_added[gid] = r

    result = pd.DataFrame(
        [{"gauge_id": gid, "round_added": rnd} for gid, rnd in round_added.items()]
    )
    return result
