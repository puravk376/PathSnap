"""Static EDA figures -> results/eda/figures/*.png

Called by ``python -m eda.run_eda`` (needs the summary table + per-trip details).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from eda.geo import LocalProjection  # noqa: E402
from eda.loader import REPO_ROOT, load_trip  # noqa: E402
from eda.trip_summary import CANDIDATE_RADIUS_M, TripDetails  # noqa: E402

FIG_DIR = REPO_ROOT / "results" / "eda" / "figures"

# Colours: one fixed role each, used the same way in every figure and the map.
TRAIN = "#2a78d6"   # blue   - train split
TEST = "#eb6834"    # orange - test split
GPS = "#2a78d6"     # blue   - GPS fixes
ROUTE = "#008300"   # green  - true route
ROAD = "#b4b2ab"    # grey   - road network
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SPLIT_COLOUR = {"train": TRAIN, "student_test": TEST}
SPLIT_LABEL = {"train": "train (80 trips)", "student_test": "test (10 trips)"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": "white",
})

_thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")


def _save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPO_ROOT))


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


# ---------------------------------------------------------------- 1. points
def points_per_trip(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = df.sort_values("n_points").reset_index(drop=True)
    ax.bar(order.index, order["n_points"], width=0.8,
           color=[SPLIT_COLOUR[s] for s in order["split"]])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(_thousands)
    ax.set_xticks([])
    ax.set_xlabel("trips, sorted by number of GPS points")
    ax.set_ylabel("GPS points per trip (log scale)")
    med = df["n_points"].median()
    ax.axhline(med, color=INK2, lw=1, ls="--")
    ax.text(1, med * 1.08, f"median {med:,.0f}", color=INK2, fontsize=9)
    ax.legend(handles=[Patch(color=c, label=SPLIT_LABEL[k]) for k, c in SPLIT_COLOUR.items()],
              loc="upper left")
    ax.set_title(f"Trips differ in length by {df['n_points'].max() / df['n_points'].min():.0f}x", pad=22)
    _subtitle(ax, f"{df['n_points'].min():,} to {df['n_points'].max():,} points per trip; "
                  f"{df['n_points'].sum():,} points in total")
    return _save(fig, "01_points_per_trip.png")


# ---------------------------------------------------------------- 2. time gaps
GAP_BINS = [(0, 1, "1 s"), (1, 2, "2 s"), (2, 5, "3-5 s"), (5, 10, "6-10 s"),
            (10, 60, "11-60 s"), (60, 600, "1-10 min"), (600, np.inf, "> 10 min")]


def time_gaps(details: list[TripDetails]) -> str:
    dt = np.concatenate([d.dt_s for d in details])
    counts = [((dt > lo) & (dt <= hi)).sum() for lo, hi, _ in GAP_BINS]
    labels = [lab for *_, lab in GAP_BINS]
    share = 100 * np.array(counts) / len(dt)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(labels, counts, color=TRAIN, width=0.7)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(_thousands)
    ax.set_ylabel("number of consecutive-fix pairs (log)")
    ax.set_xlabel("time between two consecutive GPS fixes")
    for b, c, p in zip(bars, counts, share):
        pct = "<0.01%" if p < 0.01 else (f"{p:.2f}%" if p < 1 else f"{p:.1f}%")
        ax.text(b.get_x() + b.get_width() / 2, c * 1.15, f"{c:,}\n{pct}",
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_ylim(top=max(counts) * 12)
    ax.set_title("GPS is sampled every second, with occasional gaps", pad=22)
    _subtitle(ax, f"{len(dt):,} intervals over all 90 trips; "
                  f"{(dt > 60).sum()} gaps longer than a minute (signal lost)")
    return _save(fig, "02_time_gaps.png")


# ------------------------------------------------- 3. GPS noise vs true route
OFF = "#1baf7a"   # aqua - the few trips that leave the mapped route


def distance_to_route(df: pd.DataFrame, details: list[TripDetails]) -> str:
    off_ids = set(df.loc[df["off_route_trip"], "trip_id"])
    groups = [
        ("test (10 trips)", TEST, [x for x in details if x.split == "student_test"]),
        (f"train, normal trips ({sum(x.split == 'train' and x.trip_id not in off_ids for x in details)})",
         TRAIN, [x for x in details if x.split == "train" and x.trip_id not in off_ids]),
        (f"train, off-route trips ({', '.join(sorted(t.lstrip('0') for t in off_ids))})", OFF,
         [x for x in details if x.trip_id in off_ids]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for label, colour, ds in groups:
        if not ds:
            continue
        d = np.sort(np.concatenate([x.dist_to_route_m for x in ds]))
        ax.plot(d, 100 * np.arange(1, len(d) + 1) / len(d), color=colour, lw=2, label=label)
    clean = np.concatenate([x.dist_to_route_m for x in details if x.trip_id not in off_ids])
    d_all = np.concatenate([x.dist_to_route_m for x in details])
    for r in (10, 20, CANDIDATE_RADIUS_M):
        ax.axvline(r, color=INK2, lw=1, ls=":")
        ax.text(r + 0.8, 4, f"{100 * np.mean(clean <= r):.2f}%\nwithin {r} m", fontsize=8.5, color=INK)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 101)
    ax.set_xlabel("distance from GPS fix to the nearest road piece of the TRUE route (m)")
    ax.set_ylabel("% of GPS fixes (cumulative)")
    ax.legend(loc="center right")
    ax.set_title(f"GPS noise is small: {100 * np.mean(clean <= CANDIDATE_RADIUS_M):.2f}% of fixes lie "
                 f"within {CANDIDATE_RADIUS_M} m of the true road", pad=22)
    _subtitle(ax, f"normal trips: median {np.median(clean):.1f} m, 95th pct {np.percentile(clean, 95):.1f} m. "
                  f"{len(off_ids)} train trips drive off the mapped route; with them only "
                  f"{100 * np.mean(d_all <= CANDIDATE_RADIUS_M):.1f}% are within {CANDIDATE_RADIUS_M} m")
    return _save(fig, "03_gps_to_route_distance.png")


# --------------------------------------------- 4. candidates (ambiguity)
def candidates_per_point(details: list[TripDetails]) -> str:
    n = np.concatenate([d.n_candidate_roads for d in details])
    nearest_ok = np.concatenate([np.isclose(d.dist_to_road_m, d.dist_to_route_m) for d in details])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    top = int(np.percentile(n, 99.5))
    bins = np.arange(0, top + 2) - 0.5
    ax.hist(np.clip(n, 0, top), bins=bins, color=TRAIN, rwidth=0.85)
    ax.yaxis.set_major_formatter(_thousands)
    ax.set_xlabel(f"road pieces within {CANDIDATE_RADIUS_M} m of a GPS fix "
                  f"(two-way street counted once; last bar = {top} or more)")
    ax.set_ylabel("GPS fixes")
    med = np.median(n)
    ax.axvline(med, color=INK2, lw=1, ls="--")
    ax.text(med + 0.3, ax.get_ylim()[1] * 0.9, f"median {med:.0f}", color=INK2, fontsize=9)
    ax.set_title("Each GPS fix has several candidate roads - matching is a real choice", pad=22)
    _subtitle(ax, f"mean {n.mean():.1f} road pieces per fix ({100 * np.mean(n == 0):.1f}% have none - mostly "
                  f"trip 54's unmapped area); nearest road = true route for only {100 * nearest_ok.mean():.1f}% of fixes")
    return _save(fig, "04_candidates_per_point.png")


# --------------------------------------------- 10. per-trip quality
def route_coverage_per_trip(df: pd.DataFrame) -> str:
    order = df.sort_values("pct_within_50m_of_route").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(order.index, order["pct_within_50m_of_route"], width=0.8,
           color=[OFF if o else SPLIT_COLOUR[s] for s, o in zip(order["split"], order["off_route_trip"])])
    flagged = order[order["off_route_trip"]]
    # one label per flagged bar, stacked to the right so they never overlap
    # flagged bars are sorted low -> high, so labels go bottom -> top: no crossing lines
    for y, (i, r) in zip(np.linspace(18, 78, max(len(flagged), 1)), flagged.iterrows()):
        ax.annotate(f"trip {int(r.trip_id)}: {r.pct_within_50m_of_route:.0f}%",
                    xy=(i, r.pct_within_50m_of_route), xytext=(len(flagged) + 5, y),
                    fontsize=8.5, color=INK, va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID),
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xticks([])
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("trips, sorted")
    ax.set_ylabel(f"% of fixes within {CANDIDATE_RADIUS_M} m of the true route")
    ax.legend(handles=[Patch(color=c, label=lab) for lab, c in
                       [("train", TRAIN), ("test", TEST), ("flagged trip", OFF)]], loc="lower right")
    ax.set_title(f"{int(df['off_route_trip'].sum())} train trips have long stretches far from the true route",
                 pad=22)
    _subtitle(ax, "54: 5 h of laps in an unmapped 1 km area | 33: circling an area off the route | "
                  "2: loop on unmapped roads | 43: GPS 50-70 m beside the labelled motorway")
    return _save(fig, "10_route_coverage_per_trip.png")


# --------------------------------------------- 5. road network size
def roads_vs_points(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for split, colour in SPLIT_COLOUR.items():
        g = df[df["split"] == split]
        ax.scatter(g["n_points"], g["n_arcs"], s=40, color=colour, edgecolor="white",
                   linewidth=1.5, label=SPLIT_LABEL[split], zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(_thousands)
    ax.yaxis.set_major_formatter(_thousands)
    ax.set_xlabel("GPS points in trip (log)")
    ax.set_ylabel("road pieces (arcs) in trip's map (log)")
    for _, r in df.nlargest(3, "n_arcs").iterrows():
        ax.annotate(f"trip {int(r.trip_id)}", (r.n_points, r.n_arcs), xytext=(6, -3),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.legend(loc="lower right")
    ax.set_title(f"Road maps per trip range from {df['n_arcs'].min():,} to {df['n_arcs'].max():,} arcs", pad=22)
    _subtitle(ax, f"{df['n_arcs'].sum():,} arcs in total; true route uses only "
                  f"{df['n_route_arcs'].sum():,} of them ({100 * df['n_route_arcs'].sum() / df['n_arcs'].sum():.2f}%)")
    return _save(fig, "05_roads_vs_points.png")


# --------------------------------------------- 6. speed
def speed_distribution(details: list[TripDetails]) -> str:
    v = np.concatenate([d.win_speed_kmh for d in details])
    v = v[np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(np.clip(v, 0, 200), bins=np.arange(0, 205, 5), color=TRAIN, rwidth=0.88)
    ax.yaxis.set_major_formatter(_thousands)
    ax.set_xlabel("speed averaged over 10 GPS steps (km/h; last bar = 200+)")
    ax.set_ylabel("windows")
    ax.set_title("Mix of city, rural and motorway driving", pad=22)
    _subtitle(ax, f"median {np.median(v):.0f} km/h; only {100 * np.mean(v < 15):.1f}% of windows below 15 km/h - "
                  "stops seem to have been filtered out of the tracks")
    return _save(fig, "06_speed.png")


# --------------------------------------------- 7. trip locations (world)
def trip_locations(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    for split, colour in SPLIT_COLOUR.items():
        g = df[df["split"] == split]
        ax.scatter(g["centre_lon"], g["centre_lat"], s=45, color=colour, edgecolor="white",
                   linewidth=1.5, label=SPLIT_LABEL[split], zorder=3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-60, 81, 20))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="lower left")
    ax.set_title("Trips come from all over the world", pad=22)
    _subtitle(ax, f"{df['utm_epsg'].nunique()} different UTM zones - so one national grid "
                  "(e.g. EPSG:27700, UK only) cannot be used for metres")
    return _save(fig, "07_trip_locations.png")


# --------------------------------------------- 8. gallery of trips
def _draw_trip(ax, trip_id: str, title: str, window_m: float | None = None) -> None:
    trip = load_trip(trip_id)
    t, nodes, arcs = trip.track, trip.nodes, trip.arcs.to_numpy()
    proj = LocalProjection(t["lon"].to_numpy(), t["lat"].to_numpy())
    px, py = proj.to_metres(t["lon"].to_numpy(), t["lat"].to_numpy())
    nx, ny = proj.to_metres(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    ox, oy = px.min(), py.min()
    px, py, nx, ny = px - ox, py - oy, nx - ox, ny - oy
    if window_m is None:
        margin = 0.06 * max(px.max(), py.max())
        x0, x1, y0, y1 = -margin, px.max() + margin, -margin, py.max() + margin
    else:  # zoom on the middle of the trip
        cx, cy = px[len(px) // 2], py[len(py) // 2]
        x0, x1, y0, y1 = cx - window_m / 2, cx + window_m / 2, cy - window_m / 2, cy + window_m / 2
    seg = np.stack([np.column_stack([nx[arcs[:, 0]], ny[arcs[:, 0]]]),
                    np.column_stack([nx[arcs[:, 1]], ny[arcs[:, 1]]])], axis=1)
    inside = ((seg[:, :, 0].max(1) >= x0) & (seg[:, :, 0].min(1) <= x1)
              & (seg[:, :, 1].max(1) >= y0) & (seg[:, :, 1].min(1) <= y1))
    ax.add_collection(LineCollection(seg[inside], colors=ROAD, linewidths=0.6, zorder=1))
    ax.add_collection(LineCollection(seg[trip.route.to_numpy()], colors=ROUTE,
                                     linewidths=3.0 if window_m else 2.2, zorder=2))
    ax.scatter(px, py, s=10 if window_m else 2, color=GPS, zorder=3, linewidths=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRID)
    span = x1 - x0
    bar = 10 ** np.floor(np.log10(span / 4))
    bar = bar * 5 if span / bar > 15 else bar
    ax.plot([x0 + 0.05 * span, x0 + 0.05 * span + bar], [y0 + 0.05 * (y1 - y0)] * 2, color=INK, lw=2, zorder=4)
    ax.text(x0 + 0.05 * span, y0 + 0.07 * (y1 - y0), f"{bar:,.0f} m" if bar < 1000 else f"{bar / 1000:,.0f} km",
            fontsize=8, color=INK, zorder=4)
    ax.set_title(title, fontsize=9.5)


def trip_gallery(df: pd.DataFrame, trip_ids: list[str]) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, tid in zip(axes.flat, trip_ids):
        r = df.set_index("trip_id").loc[tid]
        _draw_trip(ax, tid, f"trip {tid} ({'test' if r.split == 'student_test' else 'train'}) - "
                            f"{r.n_points:,} pts, {r.route_km:.1f} km")
    _legend(fig)
    fig.suptitle("Whole trips: GPS fixes (blue) follow the true route (green) through the road map (grey)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return _save(fig, "08_trip_gallery.png")


def zoom_gallery(df: pd.DataFrame, trip_ids: list[str], window_m: float = 400) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9))
    for ax, tid in zip(axes.flat, trip_ids):
        r = df.set_index("trip_id").loc[tid]
        _draw_trip(ax, tid, f"trip {tid} - {window_m:.0f} m window, "
                            f"median noise {r.dist_to_route_median_m:.1f} m", window_m=window_m)
    _legend(fig)
    fig.suptitle("Close up: the noisy fixes scatter around the true road, with other roads close by",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return _save(fig, "09_zoom_gallery.png")


def _legend(fig) -> None:
    handles = [plt.Line2D([], [], marker="o", ls="", color=GPS, label="GPS fix"),
               plt.Line2D([], [], color=ROUTE, lw=2.5, label="true route"),
               plt.Line2D([], [], color=ROAD, lw=1.2, label="road network")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)


def pick_gallery_trips(df: pd.DataFrame, n: int = 6) -> list[str]:
    """Trips at evenly spaced percentiles of route length (short to long)."""
    ordered = df.sort_values("route_km").reset_index(drop=True)
    idx = np.linspace(0, len(ordered) - 1, n + 2)[1:-1].round().astype(int)
    return ordered.loc[idx, "trip_id"].tolist()


def make_all(df: pd.DataFrame, details: list[TripDetails]) -> list[str]:
    gallery = pick_gallery_trips(df)
    return [
        points_per_trip(df),
        time_gaps(details),
        distance_to_route(df, details),
        candidates_per_point(details),
        roads_vs_points(df),
        speed_distribution(details),
        trip_locations(df),
        trip_gallery(df, gallery),
        zoom_gallery(df, gallery[:3]),
        route_coverage_per_trip(df),
    ]
