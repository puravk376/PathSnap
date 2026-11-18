"""
PathSnap - Step 2: Candidate generation + per-point labelling (Person B)

What this script does, for every trip:
  1. Loads the 4 raw files (.track, .nodes, .arcs, .route).
  2. Converts lon/lat (degrees) to metres using British National Grid (EPSG:27700),
     because the data is in England and distances in degrees are meaningless.
  3. Builds per-point labels: the .route file lists the true road segments IN ORDER,
     but does not say which GPS point belongs to which segment. We align them with
     dynamic programming (each point gets one route arc, and the arc index can only
     stay the same or move forward in time).
  4. Generates candidates: for every GPS point, every road segment (arc) within
     RADIUS metres. Each (point, candidate arc) pair becomes one row.
  5. label = 1 if the candidate arc is the true arc for that point, else 0.
  6. Reports the candidate miss rate (true arc not inside the candidate set).

Usage:
  python build_candidates.py <partition_dir> <output_file> [radius_m]
  e.g. python build_candidates.py "Student folder/train" candidates_train.parquet 50
"""
import sys
import os
import numpy as np
import pandas as pd
from pyproj import Transformer

RADIUS = 50.0  # metres; tune this by looking at the candidate miss rate

# lon/lat (WGS84) -> British National Grid metres. always_xy=True means (lon, lat) order.
TO_METRES = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


# ---------------------------------------------------------------- loading
def load_trip(trip_dir):
    tid = os.path.basename(trip_dir)
    p = lambda ext: os.path.join(trip_dir, f"{tid}.{ext}")
    track = pd.read_csv(p("track"), sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(p("nodes"), sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(p("arcs"), sep=r"\s+", header=None, names=["u", "v"])
    route = pd.read_csv(p("route"), header=None, names=["arc"])["arc"].to_numpy()
    return tid, track, nodes, arcs, route


# ---------------------------------------------------------------- geometry
def point_to_segments(px, py, ax, ay, bx, by):
    """
    Distance from ONE point to MANY line segments A->B (all numpy arrays, metres).
    Also returns t in [0,1] = where along the segment the closest point lies.
    """
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    t = np.where(seg_len2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.maximum(seg_len2, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


# ---------------------------------------------------------------- labelling
def align_points_to_route(D):
    """
    D: (n_points x n_route) matrix of distances from each GPS point to each route arc.
    Finds j(0) <= j(1) <= ... <= j(n-1) (monotonic, can skip arcs) minimising sum D[i, j(i)].
    O(n*m) dynamic programming using a running prefix-minimum.
    Returns: index into the route for every point.
    """
    n, m = D.shape
    cost = np.empty((n, m))
    back = np.empty((n, m), dtype=np.int32)
    cost[0] = D[0]
    back[0] = np.arange(m)
    for i in range(1, n):
        # best previous cost among route positions <= j
        prev = cost[i - 1]
        pm_val = np.minimum.accumulate(prev)
        # argmin of prefix: index where the running minimum was achieved
        is_new_min = np.r_[True, prev[1:] < pm_val[:-1]]
        pm_idx = np.maximum.accumulate(np.where(is_new_min, np.arange(m), 0))
        cost[i] = D[i] + pm_val
        back[i] = pm_idx
    # backtrack
    j = np.empty(n, dtype=np.int32)
    j[-1] = int(np.argmin(cost[-1]))
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    return j


# ---------------------------------------------------------------- main per-trip
def process_trip(trip_dir, radius=RADIUS):
    tid, track, nodes, arcs, route = load_trip(trip_dir)

    # project to metres
    gx, gy = TO_METRES.transform(track["lon"].to_numpy(), track["lat"].to_numpy())
    nx, ny = TO_METRES.transform(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    ax, ay, bx, by = nx[u], ny[u], nx[v], ny[v]

    # ---- labels: align every GPS point to one arc of the reference route
    rax, ray, rbx, rby = ax[route], ay[route], bx[route], by[route]
    D = np.vstack([point_to_segments(gx[i], gy[i], rax, ray, rbx, rby)[0] for i in range(len(gx))])
    route_pos = align_points_to_route(D)
    true_arc = route[route_pos]
    true_dist = D[np.arange(len(gx)), route_pos]

    # ---- candidates: all arcs within `radius` metres of each point
    rows = []
    misses = 0
    for i in range(len(gx)):
        d, t = point_to_segments(gx[i], gy[i], ax, ay, bx, by)
        cand = np.flatnonzero(d <= radius)
        if true_arc[i] not in cand:
            misses += 1
        order = cand[np.argsort(d[cand])]
        for rank, a in enumerate(order):
            rows.append((tid, i, track["t"].iat[i], gx[i], gy[i], int(a), d[a], t[a], rank,
                         len(order), int(a == true_arc[i]), true_dist[i],
                         int(true_dist[i] <= radius)))

    # true_dist_m: how far the point is from its OWN reference arc.
    # reliable=0 means the reference route does not explain this point (route/GPS mismatch);
    # drop these rows from TRAINING (keep them for honest evaluation reporting).
    df = pd.DataFrame(rows, columns=["trip_id", "point_idx", "t", "x", "y", "arc_id", "dist_m",
                                     "proj_t", "dist_rank", "n_cands", "label",
                                     "true_dist_m", "reliable"])
    stats = dict(trip_id=tid, n_points=len(gx), n_route_arcs=len(route), n_arcs=len(arcs),
                 misses=misses, miss_rate=misses / len(gx),
                 mean_true_dist=float(true_dist.mean()), max_true_dist=float(true_dist.max()),
                 route_arcs_used=len(np.unique(route_pos)),
                 mean_cands=df.groupby("point_idx").size().mean() if len(df) else 0)
    return df, stats


def main():
    part_dir, out_file = sys.argv[1], sys.argv[2]
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else RADIUS
    trips = sorted(d for d in os.listdir(part_dir) if os.path.isdir(os.path.join(part_dir, d)))

    # file-availability check: skip (and report) trips missing any of the 4 raw files
    complete, skipped = [], []
    for tid in trips:
        missing = [e for e in ("track", "nodes", "arcs", "route")
                   if not os.path.isfile(os.path.join(part_dir, tid, f"{tid}.{e}"))]
        (skipped.append((tid, missing)) if missing else complete.append(tid))
    for tid, missing in skipped:
        print(f"WARNING: skipping trip {tid}, missing file(s): {', '.join('.' + m for m in missing)}")
    print(f"{len(complete)} complete trips, {len(skipped)} skipped\n")
    trips = complete

    all_df, all_stats = [], []
    for k, tid in enumerate(trips, 1):
        df, st = process_trip(os.path.join(part_dir, tid), radius)
        all_df.append(df)
        all_stats.append(st)
        print(f"[{k}/{len(trips)}] trip {tid}: {st['n_points']} pts, "
              f"{st['mean_cands']:.1f} cands/pt, miss {st['miss_rate']*100:.1f}%", flush=True)
    out = pd.concat(all_df, ignore_index=True)
    out.to_parquet(out_file, index=False)
    stats = pd.DataFrame(all_stats)
    stats.to_csv(os.path.splitext(out_file)[0] + "_stats.csv", index=False)

    total_pts = stats["n_points"].sum()
    print("\n===== SUMMARY =====")
    print(f"trips: {len(stats)}  points: {total_pts}  candidate rows: {len(out)}")
    print(f"avg candidates per point: {len(out)/total_pts:.2f}")
    print(f"positive rate (label=1): {out['label'].mean()*100:.2f}%")
    print(f"overall candidate miss rate @ {radius:.0f} m: {stats['misses'].sum()/total_pts*100:.2f}%")
    print(f"mean distance GPS -> true arc: {np.average(stats['mean_true_dist'], weights=stats['n_points']):.2f} m")


if __name__ == "__main__":
    main()
