"""
PathSnap - Step 3: Feature engineering

Input : the candidate file from build_candidates.py (one row per GPS point x candidate arc)
        + the raw trip folders (needed for road geometry and the full GPS track).
Output: the same rows with extra feature columns, plus features.json listing the
        columns to give the models (--features).

No feature uses the label, true_dist_m or reliable columns, so there is no leakage.

Features added
  arc_length_m     length of the road segment
  arc_bearing      compass direction of the road segment u -> v (degrees, 0 = north)
  car_heading      direction the car is moving at this point (degrees, smoothed)
  heading_diff     angle between car heading and road direction, 0..180 degrees
                   -> separates a road from its opposite-direction twin
  heading_cos      cos(heading_diff): +1 same direction, -1 opposite
  heading_valid    1 if the car moved enough to trust the heading, else 0
                   (when 0, heading_diff is set to a neutral 90 and heading_cos to 0)
  speed_mps        smoothed speed of the car (metres per second)
  dt_prev          seconds since the previous GPS point (detects gaps)
  dist_minus_min   this candidate's distance minus the closest candidate's distance
  dist_ratio       this candidate's distance / (closest distance + 1 m)
  dist_prev_m      distance from the PREVIOUS GPS point to this road
  dist_next_m      distance from the NEXT GPS point to this road
  connected_prev   1 if this road is the same as, or directly follows, the previous
                   point's best guess road (nearest road that faces the car's direction)

Usage:
  python build_features.py <candidates_file> <partition_dir> <output_file>
  e.g. python features/build_features.py results/candidates_train.parquet data/train results/features_train.parquet
  (output can end in .parquet or .csv)
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from pyproj import Transformer

TO_METRES = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
HEADING_WINDOW = 2      # use points i-2 .. i+2 to estimate heading/speed (reduces GPS noise)
MIN_MOVE_M = 3.0        # car must move at least this far in the window for heading to count

FEATURE_COLUMNS = [
    "dist_m", "proj_t", "dist_rank", "n_cands",
    "arc_length_m", "heading_diff", "heading_cos", "heading_valid",
    "speed_mps", "dt_prev", "dist_minus_min", "dist_ratio",
    "dist_prev_m", "dist_next_m", "connected_prev",
]


def read_ws(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def seg_dist(px, py, ax, ay, bx, by):
    """Element-wise distance from points (px,py) to segments A->B (all same-length arrays)."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.maximum(L2, 1e-12), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def angle_diff(a, b):
    """Smallest absolute difference between two compass angles, 0..180."""
    d = np.abs(a - b) % 360
    return np.where(d > 180, 360 - d, d)


def trip_features(cdf, trip_dir):
    tid = os.path.basename(trip_dir)
    track = read_ws(os.path.join(trip_dir, f"{tid}.track"), ["lon", "lat", "t"])
    nodes = read_ws(os.path.join(trip_dir, f"{tid}.nodes"), ["lon", "lat"])
    arcs = read_ws(os.path.join(trip_dir, f"{tid}.arcs"), ["u", "v"])

    gx, gy = TO_METRES.transform(track["lon"].to_numpy(), track["lat"].to_numpy())
    gt = track["t"].to_numpy()
    nx, ny = TO_METRES.transform(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    U, V = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    n = len(gx)

    # ---------- per-point motion features (from the full GPS track)
    idx = np.arange(n)
    lo = np.clip(idx - HEADING_WINDOW, 0, n - 1)
    hi = np.clip(idx + HEADING_WINDOW, 0, n - 1)
    mdx, mdy = gx[hi] - gx[lo], gy[hi] - gy[lo]
    move = np.hypot(mdx, mdy)
    heading = (np.degrees(np.arctan2(mdx, mdy)) + 360) % 360       # 0 = north, clockwise
    heading_valid = (move >= MIN_MOVE_M).astype(int)
    span_t = np.maximum(gt[hi] - gt[lo], 1e-6)
    speed = move / span_t
    dt_prev = np.r_[0.0, np.diff(gt)]

    # ---------- per-candidate features
    df = cdf.copy()
    i = df["observation_id"].to_numpy()
    a = df["arc_id"].to_numpy()
    ax, ay, bx, by = nx[U[a]], ny[U[a]], nx[V[a]], ny[V[a]]

    df["arc_length_m"] = np.hypot(bx - ax, by - ay)
    df["arc_bearing"] = (np.degrees(np.arctan2(bx - ax, by - ay)) + 360) % 360
    df["car_heading"] = heading[i]
    df["heading_valid"] = heading_valid[i]
    hd = angle_diff(df["car_heading"].to_numpy(), df["arc_bearing"].to_numpy())
    df["heading_diff"] = np.where(df["heading_valid"] == 1, hd, 90.0)
    df["heading_cos"] = np.where(df["heading_valid"] == 1, np.cos(np.radians(hd)), 0.0)
    df["speed_mps"] = speed[i]
    df["dt_prev"] = dt_prev[i]

    dmin = df.groupby("observation_id")["dist_m"].transform("min")
    df["dist_minus_min"] = df["dist_m"] - dmin
    df["dist_ratio"] = df["dist_m"] / (dmin + 1.0)

    ip, inn = np.maximum(i - 1, 0), np.minimum(i + 1, n - 1)
    df["dist_prev_m"] = seg_dist(gx[ip], gy[ip], ax, ay, bx, by)
    df["dist_next_m"] = seg_dist(gx[inn], gy[inn], ax, ay, bx, by)

    # ---------- connected_prev (uses only GPS + geometry, never the label)
    # best guess per point = nearest candidate whose direction roughly matches the car
    # (heading_diff <= 90); if none qualifies, simply the nearest candidate.
    tmp = df[["observation_id", "arc_id", "dist_m", "heading_diff"]].copy()
    tmp["key"] = tmp["dist_m"] + np.where(tmp["heading_diff"] <= 90, 0.0, 1e6)
    best = tmp.loc[tmp.groupby("observation_id")["key"].idxmin()].set_index("observation_id")["arc_id"]
    prev_best = pd.Series(best.values, index=best.index + 1)        # shift: point i sees i-1's best
    pb = prev_best.reindex(i).to_numpy()                             # NaN if previous point had no candidates
    has_prev = ~np.isnan(pb)
    pb_int = np.where(has_prev, pb, 0).astype(int)
    same = has_prev & (pb_int == a)
    follows = has_prev & (V[pb_int] == U[a])
    df["connected_prev"] = (same | follows).astype(int)
    return df


def main():
    cand_file, part_dir, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
    cands = pd.read_parquet(cand_file) if cand_file.endswith(".parquet") else pd.read_csv(cand_file, dtype={"trip_id": str})
    cands["trip_id"] = cands["trip_id"].astype(str).str.zfill(8)
    if "observation_id" not in cands.columns and "point_idx" in cands.columns:
        # file made by an older build_candidates.py
        cands = cands.rename(columns={"point_idx": "observation_id"})

    out = []
    trips = sorted(cands["trip_id"].unique())
    for k, tid in enumerate(trips, 1):
        out.append(trip_features(cands[cands["trip_id"] == tid], os.path.join(part_dir, tid)))
        print(f"[{k}/{len(trips)}] trip {tid}", flush=True)
    df = pd.concat(out, ignore_index=True)

    if out_file.endswith(".csv"):
        df.to_csv(out_file, index=False)
    else:
        df.to_parquet(out_file, index=False)
    with open(os.path.join(os.path.dirname(out_file) or ".", "features.json"), "w") as f:
        json.dump({"features": FEATURE_COLUMNS,
                   "do_not_use_as_features": ["label", "true_dist_m", "reliable", "trip_id",
                                              "observation_id", "arc_id", "t", "x", "y"]}, f, indent=2)

    print(f"\nrows: {len(df)}  columns: {df.shape[1]}")
    print(f"feature columns ({len(FEATURE_COLUMNS)}): {' '.join(FEATURE_COLUMNS)}")
    print("missing values in features:", int(df[FEATURE_COLUMNS].isna().sum().sum()))
    print("\nmean feature value, true road (label=1) vs wrong road (label=0):")
    print(df.groupby("label")[["dist_m", "heading_diff", "connected_prev", "dist_prev_m"]].mean().round(2))


if __name__ == "__main__":
    main()
