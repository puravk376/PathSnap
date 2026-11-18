"""Per-trip statistics: points, time gaps, missing values, roads, route, GPS noise.

Run as a script to write results/eda/trip_summary.csv (+ .md and split totals)::

    python -m eda.trip_summary
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely

from eda.geo import LocalProjection, haversine_m, point_to_segments_distance
from eda.loader import REPO_ROOT, Trip, iter_trips

OUT_DIR = REPO_ROOT / "results" / "eda"

GAP_SHORT_S = 5        # a "gap" = more than 5 s between fixes (normal is 1 s)
GAP_LONG_S = 60        # a "long gap" = signal lost for over a minute
JUMP_KMH = 200         # a single 1-step speed above this is implausible for a car
SPEED_WINDOW = 10      # steps used for the robust (windowed) speed
CANDIDATE_RADIUS_M = 50
OFF_ROUTE_TRIP_PCT = 95  # trips with < 95% of fixes within 50 m of the route are flagged


@dataclass
class TripDetails:
    """Per-point arrays kept for plotting (not written to the CSV)."""
    trip_id: str
    split: str
    dt_s: np.ndarray            # time between consecutive fixes
    step_m: np.ndarray          # distance between consecutive fixes
    speed_kmh: np.ndarray
    win_speed_kmh: np.ndarray    # speed over SPEED_WINDOW steps (robust)
    dist_to_route_m: np.ndarray  # each fix -> nearest arc of the TRUE route
    dist_to_road_m: np.ndarray   # each fix -> nearest arc of the whole network
    n_candidates: np.ndarray     # directed arcs within CANDIDATE_RADIUS_M of each fix
    n_candidate_roads: np.ndarray  # same, two-way streets counted once
    arc_len_m: np.ndarray


def _missing_values(trip: Trip) -> int:
    return int(trip.track.isna().sum().sum() + trip.nodes.isna().sum().sum()
               + trip.arcs.isna().sum().sum() + trip.route.isna().sum())


def _bad_references(trip: Trip) -> int:
    n_nodes, n_arcs = len(trip.nodes), len(trip.arcs)
    a = trip.arcs.to_numpy()
    r = trip.route.to_numpy()
    return int(((a < 0) | (a >= n_nodes)).sum() + ((r < 0) | (r >= n_arcs)).sum())


def analyse_trip(trip: Trip) -> tuple[dict, TripDetails]:
    t = trip.track
    lon, lat, time = t["lon"].to_numpy(), t["lat"].to_numpy(), t["time"].to_numpy()

    # ---- time and motion --------------------------------------------------
    dt = np.diff(time)
    step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt * 3.6, np.nan)
    duration = float(time[-1] - time[0])
    median_dt = float(np.median(dt)) if len(dt) else np.nan
    expected = duration / median_dt + 1 if median_dt and median_dt > 0 else len(t)
    moving = speed[(dt <= GAP_SHORT_S) & (speed > 2)]
    # Timestamps next to a gap are often misplaced (e.g. "8 s for 30 m" followed
    # by several "1 s for 85 m" steps), so raw 1-step speed spikes are mostly
    # timing artefacts. Speed over a 10-step window is far more trustworthy.
    if len(dt) >= SPEED_WINDOW:
        k = np.ones(SPEED_WINDOW)
        win_speed = np.convolve(step, k, "valid") / np.convolve(dt, k, "valid") * 3.6
    else:
        win_speed = speed

    # ---- road network ------------------------------------------------------
    a = trip.arcs.to_numpy()
    # a two-way street is stored as two arcs (u->v and v->u); count it once
    _, first_of_pair, pair_of_arc = np.unique(np.sort(a, axis=1), axis=0, return_index=True,
                                              return_inverse=True)
    pair_of_arc = pair_of_arc.ravel()
    n_pairs = len(first_of_pair)
    arc_lonlat = trip.arc_coords
    arc_len = haversine_m(arc_lonlat[:, 0, 0], arc_lonlat[:, 0, 1], arc_lonlat[:, 1, 0], arc_lonlat[:, 1, 1])
    reverse_keys = set(map(tuple, a[:, ::-1]))
    two_way = float(np.mean([k in reverse_keys for k in map(tuple, a)]))

    # ---- true route ------------------------------------------------------------
    r = trip.route.to_numpy()
    ra = a[r]
    route_breaks = int((ra[1:, 0] != ra[:-1, 1]).sum())
    route_len_m = float(arc_len[r].sum())
    track_len_m = float(step.sum())

    # ---- GPS noise: distance from every fix to the true route / any road ----
    proj = LocalProjection(np.concatenate([lon, trip.nodes["lon"].to_numpy()]),
                           np.concatenate([lat, trip.nodes["lat"].to_numpy()]))
    px, py = proj.to_metres(lon, lat)
    nx, ny = proj.to_metres(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy())
    nxy = np.column_stack([nx, ny])
    seg_all = np.stack([nxy[a[:, 0]], nxy[a[:, 1]]], axis=1)
    d_route = point_to_segments_distance(px, py, seg_all[np.unique(r)])
    d_road = point_to_segments_distance(px, py, seg_all)

    tree = shapely.STRtree(shapely.linestrings(seg_all))
    pts = shapely.points(px, py)
    hits = tree.query(pts, predicate="dwithin", distance=CANDIDATE_RADIUS_M)
    n_cand = np.bincount(hits[0], minlength=len(pts))  # directed arcs
    # same, but a two-way street (2 arcs) counts as one road piece
    point_pair = np.unique(np.column_stack([hits[0], pair_of_arc[hits[1]]]), axis=0)
    n_cand_roads = np.bincount(point_pair[:, 0], minlength=len(pts))

    row = {
        "trip_id": trip.trip_id,
        "split": trip.split,
        # track
        "n_points": len(t),
        "duration_min": duration / 60,
        "median_dt_s": median_dt,
        "max_gap_s": float(dt.max()) if len(dt) else 0.0,
        f"n_gaps_gt_{GAP_SHORT_S}s": int((dt > GAP_SHORT_S).sum()),
        f"n_gaps_gt_{GAP_LONG_S}s": int((dt > GAP_LONG_S).sum()),
        "pct_samples_missing": 100 * max(0.0, 1 - len(t) / expected),
        "missing_values": _missing_values(trip),
        "bad_references": _bad_references(trip),
        "n_stationary_fixes": int(((step == 0)).sum()),
        "track_km": track_len_m / 1000,
        "avg_moving_speed_kmh": float(np.mean(moving)) if len(moving) else np.nan,
        f"n_steps_over_{JUMP_KMH}kmh": int((speed > JUMP_KMH).sum()),
        f"max_speed_{SPEED_WINDOW}step_kmh": float(np.max(win_speed)) if len(win_speed) else np.nan,
        "centre_lon": float(np.mean(lon)),
        "centre_lat": float(np.mean(lat)),
        "utm_epsg": proj.crs.to_epsg(),
        # roads
        "n_nodes": len(trip.nodes),
        "n_arcs": len(a),
        "n_road_pieces_undirected": n_pairs,
        "pct_arcs_two_way": 100 * two_way,
        "n_duplicate_arcs": int(trip.arcs.duplicated().sum()),
        "n_self_loop_arcs": int((a[:, 0] == a[:, 1]).sum()),
        "median_arc_len_m": float(np.median(arc_len)),
        "road_km_undirected": float(arc_len[first_of_pair].sum()) / 1000,
        # true route
        "n_route_arcs": len(r),
        "n_unique_route_arcs": int(len(np.unique(r))),
        "n_route_breaks": route_breaks,
        "route_km": route_len_m / 1000,
        "points_per_route_arc": len(t) / len(r),
        "track_to_route_len_ratio": track_len_m / route_len_m if route_len_m else np.nan,
        # GPS noise vs truth
        "dist_to_route_median_m": float(np.nanmedian(d_route)),
        "dist_to_route_p95_m": float(np.nanpercentile(d_route, 95)),
        "dist_to_route_max_m": float(np.nanmax(d_route)),
        "pct_within_20m_of_route": 100 * float(np.mean(d_route <= 20)),
        "pct_within_50m_of_route": 100 * float(np.mean(d_route <= CANDIDATE_RADIUS_M)),
        "mean_candidate_arcs_50m": float(np.mean(n_cand)),
        "mean_candidate_roads_50m": float(np.mean(n_cand_roads)),
        "pct_no_candidate_50m": 100 * float(np.mean(n_cand == 0)),
        "pct_nearest_road_is_route": 100 * float(np.mean(np.isclose(d_road, d_route))),
    }
    row["off_route_trip"] = row["pct_within_50m_of_route"] < OFF_ROUTE_TRIP_PCT
    details = TripDetails(trip.trip_id, trip.split, dt, step, speed, win_speed, d_route, d_road, n_cand, n_cand_roads, arc_len)
    return row, details


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per split (+ overall) with totals and typical values."""
    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "trips": len(g),
            "gps_points": int(g["n_points"].sum()),
            "points_per_trip_median": g["n_points"].median(),
            "points_per_trip_min": g["n_points"].min(),
            "points_per_trip_max": g["n_points"].max(),
            "hours_spanned_incl_gaps": g["duration_min"].sum() / 60,
            "track_km": g["track_km"].sum(),
            "arcs_total": int(g["n_arcs"].sum()),
            "arcs_per_trip_median": g["n_arcs"].median(),
            "route_arcs_total": int(g["n_route_arcs"].sum()),
            "missing_values": int(g["missing_values"].sum()),
            "bad_references": int(g["bad_references"].sum()),
            f"trips_with_gap_gt_{GAP_LONG_S}s": int((g[f"n_gaps_gt_{GAP_LONG_S}s"] > 0).sum()),
            "off_route_trips": int(g["off_route_trip"].sum()),
            "pct_points_within_50m_of_route":
                float(np.average(g["pct_within_50m_of_route"], weights=g["n_points"])),
        })
    parts = [agg(g).rename(s) for s, g in df.groupby("split", sort=False)]
    parts.append(agg(df).rename("all"))
    return pd.DataFrame(parts)


def run(verbose: bool = True) -> tuple[pd.DataFrame, list[TripDetails]]:
    rows, details = [], []
    for trip in iter_trips():
        row, det = analyse_trip(trip)
        rows.append(row)
        details.append(det)
        if verbose:
            print(f"  {trip.split:12s} {trip.trip_id}  {row['n_points']:6d} pts  "
                  f"{row['n_arcs']:7d} arcs  median GPS->route {row['dist_to_route_median_m']:5.1f} m")
    return pd.DataFrame(rows), details


def to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    """Plain GitHub markdown table (avoids an extra 'tabulate' dependency)."""
    d = df.rename_axis(df.index.name or "").reset_index() if index else df
    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else ""
        if isinstance(v, (float, np.floating)):
            return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |",
             "|" + "|".join("---" for _ in d.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def save_tables(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(OUT_DIR / "trip_summary.csv", index=False)
    split = split_summary(df)
    # counts come out as floats after building the frame; store them as integers
    split = split.apply(lambda c: c.astype("int64") if (c % 1 == 0).all() else c.round(2))
    split.to_csv(OUT_DIR / "split_summary.csv")

    short_cols = ["trip_id", "split", "n_points", "duration_min", "max_gap_s",
                  f"n_gaps_gt_{GAP_LONG_S}s", "missing_values", "n_arcs", "n_route_arcs",
                  "route_km", "dist_to_route_median_m", "dist_to_route_p95_m",
                  "pct_within_50m_of_route", "mean_candidate_roads_50m", "off_route_trip"]
    with open(OUT_DIR / "trip_summary.md", "w", encoding="utf-8") as f:
        f.write("# Trip summary\n\nGenerated by `python -m eda.run_eda`. "
                "Full table with all columns: `trip_summary.csv`.\n\n")
        f.write("## Per split\n\n")
        f.write(to_markdown(split))
        f.write("\n\n## Per trip\n\n")
        f.write(to_markdown(df[short_cols], index=False))
        f.write("\n")


if __name__ == "__main__":
    summary, _ = run()
    save_tables(summary)
    print(f"wrote {OUT_DIR / 'trip_summary.csv'}")
