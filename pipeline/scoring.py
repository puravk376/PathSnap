"""
PathSnap - server-side scoring against the reference route.

This module is the ONLY place that reads .route files at prediction time. It is called after
the prediction is finished, to compute accuracy. The UI only ever receives the metrics,
never the reference route itself.
"""
import os
import sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from data_processing.build_candidates import TO_METRES, point_to_segments, align_points_to_route  # noqa: E402


def reference_point_arcs(trip_dir, track, nodes, arcs):
    """Reference arc for every GPS point (same monotonic alignment used to build training labels)
    plus the distance from the point to it (reliability)."""
    tid = os.path.basename(os.path.normpath(trip_dir))
    route = pd.read_csv(os.path.join(trip_dir, f"{tid}.route"), header=None)[0].to_numpy()
    gx, gy = TO_METRES.transform(track.lon.to_numpy(), track.lat.to_numpy())
    nx, ny = TO_METRES.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    U, V = arcs.u.to_numpy(), arcs.v.to_numpy()
    r = route
    D = np.vstack([point_to_segments(gx[i], gy[i], nx[U[r]], ny[U[r]], nx[V[r]], ny[V[r]])[0] for i in range(len(gx))])
    pos = align_points_to_route(D)
    return route, route[pos], D[np.arange(len(gx)), pos]


def continuity(pred, U, V):
    a = pred[~np.isnan(pred)].astype(int)
    if len(a) < 2:
        return float("nan")
    prev, cur = a[:-1], a[1:]
    return float(((prev == cur) | (V[prev] == U[cur])).mean())


def score(pred, route, ref_arc, ref_dist, U, V, radius=50.0):
    """pred: array of predicted arc per GPS point (NaN where no prediction)."""
    has = ~np.isnan(pred)
    correct = has & (np.nan_to_num(pred, nan=-1).astype(int) == ref_arc)
    reliable = ref_dist <= radius
    pred_set = set(pred[has].astype(int))
    ref_set = set(route.tolist())
    tp = len(pred_set & ref_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(ref_set) if ref_set else 0.0
    return dict(
        point_accuracy=float(correct.mean()),
        point_accuracy_reliable=float(correct[reliable].mean()) if reliable.any() else float("nan"),
        coverage=float(has.mean()),
        route_continuity=continuity(pred, U, V),
        route_arc_precision=prec, route_arc_recall=rec,
        route_arc_f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        n_points=int(len(pred)),
    )
