"""Shared helpers for model training/evaluation (metrics, sampling, data loading)."""
import os
import sys
import json
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from features.build_features import FEATURE_COLUMNS  # noqa: E402

SEED = 42
KEYS = ["trip_id", "observation_id"]


def load_table(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"trip_id": str})
    df["trip_id"] = df["trip_id"].astype(str).str.zfill(8)
    return df


def load_train(path, reliable_only=True):
    """Training rows. Points whose reference arc is >50 m away (route/GPS mismatch) are dropped."""
    df = load_table(path)
    if reliable_only and "reliable" in df.columns:
        df = df[df["reliable"] == 1].reset_index(drop=True)
    return df


def sample_observations(df, frac, seed=SEED):
    """Sample whole GPS points (all their candidates together), so Top-1 stays computable."""
    if frac >= 1:
        return df
    obs = df[KEYS].drop_duplicates()
    keep = obs.sample(frac=frac, random_state=seed)
    return df.merge(keep, on=KEYS).reset_index(drop=True)


def top1_choice(df, proba):
    """For every GPS point, the candidate with the highest predicted probability."""
    tmp = df[KEYS + ["arc_id", "label"]].copy()
    tmp["score"] = proba
    idx = tmp.groupby(KEYS, sort=False)["score"].idxmax()
    return tmp.loc[idx].sort_values(KEYS).reset_index(drop=True)


def top1_accuracy(df, proba):
    """Observation-level Top-1: fraction of GPS points whose top-scored candidate is the reference arc.
    Points whose reference arc is not in the candidate set count as wrong."""
    return float(top1_choice(df, proba)["label"].mean())


def candidate_miss_rate(df):
    pos = df.groupby(KEYS)["label"].max()
    return float((pos == 0).mean())


_ARCS = {}


def load_arcs(part_dir, tid):
    key = (part_dir, tid)
    if key not in _ARCS:
        a = pd.read_csv(os.path.join(part_dir, tid, f"{tid}.arcs"), sep=r"\s+", header=None, names=["u", "v"])
        _ARCS[key] = (a.u.to_numpy(), a.v.to_numpy())
    return _ARCS[key]


def route_continuity(choice, part_dir):
    """Share of consecutive GPS points whose predicted arcs are the same arc or directly connected
    (end node of previous arc == start node of next arc)."""
    ok = tot = 0
    for tid, g in choice.groupby("trip_id"):
        U, V = load_arcs(part_dir, tid)
        o = g["observation_id"].to_numpy()
        a = g["arc_id"].to_numpy()
        consec = np.diff(o) == 1
        prev, cur = a[:-1][consec], a[1:][consec]
        ok += int(((prev == cur) | (V[prev] == U[cur])).sum())
        tot += int(consec.sum())
    return ok / tot if tot else float("nan")


def feature_list(path=os.path.join("results", "features.json")):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)["features"]
    return list(FEATURE_COLUMNS)
