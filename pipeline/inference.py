"""
PathSnap - label-free prediction pipeline (used by the API and the evaluation scripts)

predict_trip(trip_dir, ...) reads ONLY .track, .nodes and .arcs. It never opens .route,
so reference labels cannot leak into the predictions.

Steps: GPS + nodes -> metres -> candidate arcs within radius -> features -> classifier
probability per candidate -> classifier-only route (Top-1) + HMM/Viterbi route.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from data_processing.build_candidates import TO_METRES, point_to_segments  # noqa: E402
from features.build_features import trip_features  # noqa: E402
from hmm.viterbi import viterbi_trip, DEFAULT_HMM  # noqa: E402


def read_ws(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def load_inputs(trip_dir):
    tid = os.path.basename(os.path.normpath(trip_dir))
    track = read_ws(os.path.join(trip_dir, f"{tid}.track"), ["lon", "lat", "t"])
    nodes = read_ws(os.path.join(trip_dir, f"{tid}.nodes"), ["lon", "lat"])
    arcs = read_ws(os.path.join(trip_dir, f"{tid}.arcs"), ["u", "v"])
    return tid, track, nodes, arcs


def generate_candidates(tid, track, nodes, arcs, radius):
    gx, gy = TO_METRES.transform(track.lon.to_numpy(), track.lat.to_numpy())
    nx, ny = TO_METRES.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    U, V = arcs.u.to_numpy(), arcs.v.to_numpy()
    ax, ay, bx, by = nx[U], ny[U], nx[V], ny[V]
    # speed-up: only arcs whose bounding box comes within `radius` of the trajectory's bounding box
    near = ((np.maximum(ax, bx) >= gx.min() - radius) & (np.minimum(ax, bx) <= gx.max() + radius) &
            (np.maximum(ay, by) >= gy.min() - radius) & (np.minimum(ay, by) <= gy.max() + radius))
    idx = np.flatnonzero(near)
    rows = []
    for i in range(len(gx)):
        d, t = point_to_segments(gx[i], gy[i], ax[idx], ay[idx], bx[idx], by[idx])
        c = np.flatnonzero(d <= radius)
        c = c[np.argsort(d[c])]
        for rank, k in enumerate(c):
            rows.append((tid, i, track.t.iat[i], gx[i], gy[i], int(idx[k]), d[k], t[k], rank, len(c)))
    cdf = pd.DataFrame(rows, columns=["trip_id", "observation_id", "t", "x", "y", "arc_id",
                                      "dist_m", "proj_t", "dist_rank", "n_cands"])
    return cdf, (gx, gy, nx, ny, U, V)


class Predictor:
    def __init__(self, artifacts_dir=os.path.join(REPO, "artifacts")):
        self.model = joblib.load(os.path.join(artifacts_dir, "model.joblib"))
        with open(os.path.join(artifacts_dir, "model_config.json")) as f:
            self.cfg = json.load(f)
        hmm_path = os.path.join(artifacts_dir, "hmm_config.json")
        self.hmm = dict(DEFAULT_HMM)
        if os.path.isfile(hmm_path):
            with open(hmm_path) as f:
                self.hmm.update(json.load(f)["params"])
        self.features = self.cfg["features"]
        self.radius = float(self.cfg.get("candidate_radius_m", 50.0))

    def predict_trip(self, trip_dir, use_hmm=True):
        timing = {}
        t0 = time.perf_counter()
        tid, track, nodes, arcs = load_inputs(trip_dir)
        cdf, (gx, gy, nx, ny, U, V) = generate_candidates(tid, track, nodes, arcs, self.radius)
        timing["candidates_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        feats = trip_features(cdf, trip_dir)
        timing["features_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = self.model.predict_proba(feats[self.features])[:, 1]
        feats["proba"] = proba
        timing["classifier_s"] = time.perf_counter() - t0

        best = feats.loc[feats.groupby("observation_id")["proba"].idxmax(), ["observation_id", "arc_id", "proba"]]
        clf_arc = pd.Series(best.arc_id.to_numpy(), index=best.observation_id.to_numpy())

        hmm_arc = None
        if use_hmm and len(feats):
            t0 = time.perf_counter()
            length = np.hypot(nx[V] - nx[U], ny[V] - ny[U])
            hmm_arc = viterbi_trip(feats, proba, np.c_[gx, gy], track.t.to_numpy(), U, V, length,
                                   len(nodes), self.hmm)
            timing["hmm_s"] = time.perf_counter() - t0

        n = len(track)
        out = pd.DataFrame({"observation_id": np.arange(n), "lon": track.lon, "lat": track.lat, "t": track.t})
        out["clf_arc"] = clf_arc.reindex(out.observation_id).to_numpy()
        out["hmm_arc"] = hmm_arc.reindex(out.observation_id).to_numpy() if hmm_arc is not None else np.nan
        return dict(trip_id=tid, points=out, candidates=feats, nodes=nodes, arcs=arcs, timing=timing)
