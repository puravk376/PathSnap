"""
PathSnap - HMM/Viterbi decoder tuning and evaluation

1. Decoder tuning (training trips only, no leakage):
   * hold out 12 training trips, fit the tuned classifier on the other 68 trips,
     predict the 12 held-out trips (out-of-fold probabilities, like unseen data)
   * grid search over beta x emission_weight, choose the best point accuracy
   * save artifacts/hmm_config.json
2. Student-test evaluation with the final model (artifacts/model.joblib):
   classifier-only (Top-1) vs classifier + HMM, per trip and overall:
   point accuracy, route continuity, route-arc precision/recall/F1
3. Folium maps for student_test trips: GPS, network, reference route, classifier route, HMM route

Outputs: results/hmm/*.csv, *.png, results/maps/*.html, artifacts/hmm_config.json

Usage:
  python hmm/evaluate_hmm.py results/features_train.parquet data/train data/student_test
"""
import os
import sys
import json
import time
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "modeling"))

from common import load_train, sample_observations, SEED  # noqa: E402
from hmm.viterbi import viterbi_trip, DEFAULT_HMM  # noqa: E402
from pipeline.inference import Predictor, load_inputs  # noqa: E402
from pipeline.scoring import reference_point_arcs, score  # noqa: E402
from visualization.maps import route_map, nearby_network  # noqa: E402
from data_processing.build_candidates import TO_METRES  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

OUT, MAPS, ART = os.path.join("results", "hmm"), os.path.join("results", "maps"), "artifacts"
GRID = dict(beta=[5.0, 10.0, 20.0, 40.0, 80.0, 160.0], emission_weight=[1.0, 2.0, 4.0])


def geometry(trip_dir):
    tid, track, nodes, arcs = load_inputs(trip_dir)
    gx, gy = TO_METRES.transform(track.lon.to_numpy(), track.lat.to_numpy())
    nx, ny = TO_METRES.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    U, V = arcs.u.to_numpy(), arcs.v.to_numpy()
    return dict(xy=np.c_[gx, gy], t=track.t.to_numpy(), U=U, V=V,
                length=np.hypot(nx[V] - nx[U], ny[V] - ny[U]), n_nodes=len(nodes))


def tune_decoder(train_path, train_dir, cfg):
    full = load_train(train_path, reliable_only=False)
    rel = full.groupby("trip_id").agg(rel=("reliable", "mean"), n=("observation_id", "nunique"))
    pool = rel[(rel.rel > 0.97) & (rel.n < 4000)].index.to_numpy()
    rng = np.random.default_rng(SEED)
    hold = sorted(rng.choice(pool, size=min(12, len(pool)), replace=False))
    fit_df = sample_observations(full[(~full.trip_id.isin(hold)) & (full.reliable == 1)], 0.2)
    feats = cfg["features"]
    model = HistGradientBoostingClassifier(**cfg["params"]).fit(fit_df[feats], fit_df["label"])
    print(f"decoder tuning: {len(hold)} held-out training trips, classifier fit on {fit_df.trip_id.nunique()} other trips")

    trips = []
    for tid in hold:
        c = full[full.trip_id == tid].sort_values(["observation_id", "dist_rank"]).reset_index(drop=True)
        p = model.predict_proba(c[feats])[:, 1]
        truth = c[c.label == 1].set_index("observation_id")["arc_id"]
        rel_obs = c[c.reliable == 1]["observation_id"].unique()
        g = geometry(os.path.join(train_dir, tid))
        clf = c.assign(p=p).loc[lambda d: d.groupby("observation_id")["p"].idxmax()].set_index("observation_id")["arc_id"]
        trips.append((tid, c, p, truth, rel_obs, g, clf))

    def acc(pred, truth, rel_obs):
        pr = pred.reindex(rel_obs)
        tr = truth.reindex(rel_obs)
        return float((pr.to_numpy() == tr.to_numpy()).mean())

    clf_acc = np.mean([acc(clf, tr, ro) for _, _, _, tr, ro, _, clf in trips])
    rows = []
    for beta, w in itertools.product(GRID["beta"], GRID["emission_weight"]):
        params = {**DEFAULT_HMM, "beta": beta, "emission_weight": w}
        t0 = time.perf_counter()
        accs = [acc(viterbi_trip(c, p, g["xy"], g["t"], g["U"], g["V"], g["length"], g["n_nodes"], params), tr, ro)
                for _, c, p, tr, ro, g, _ in trips]
        rows.append(dict(beta=beta, emission_weight=w, hmm_point_accuracy=float(np.mean(accs)),
                         classifier_point_accuracy=clf_acc, seconds=time.perf_counter() - t0))
        print(f"  beta={beta:5.1f} w={w:.1f}: HMM acc {rows[-1]['hmm_point_accuracy']:.4f} (classifier {clf_acc:.4f})", flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(OUT, "hmm_grid_search.csv"), index=False)
    best = grid.loc[grid.hmm_point_accuracy.idxmax()]
    params = {**DEFAULT_HMM, "beta": float(best.beta), "emission_weight": float(best.emission_weight)}
    with open(os.path.join(ART, "hmm_config.json"), "w") as f:
        json.dump(dict(params=params, tuned_on="12 held-out training trips (out-of-fold classifier probabilities)",
                       held_out_trips=list(map(str, hold)), best_accuracy=float(best.hmm_point_accuracy),
                       classifier_only_accuracy=clf_acc), f, indent=2)
    piv = grid.pivot(index="emission_weight", columns="beta", values="hmm_point_accuracy")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for (r, c), v in np.ndenumerate(piv.values):
        ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=8)
    ax.set(xlabel="beta (transition tolerance, m)", ylabel="emission weight",
           title=f"HMM grid search (held-out train trips); classifier-only = {clf_acc:.3f}")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(OUT, "hmm_grid_search.png"), dpi=150); plt.close(fig)
    print(f"best decoder params: beta={params['beta']}, emission_weight={params['emission_weight']}")


def evaluate_student_test(test_dir, n_maps=3):
    pred = Predictor(ART)
    rows, cache = [], {}
    for tid in sorted(os.listdir(test_dir)):
        d = os.path.join(test_dir, tid)
        if not os.path.isdir(d):
            continue
        t0 = time.perf_counter()
        r = pred.predict_trip(d)
        total = time.perf_counter() - t0
        pts = r["points"]
        route, ref_arc, ref_dist = reference_point_arcs(d, pts, r["nodes"], r["arcs"])
        U, V = r["arcs"].u.to_numpy(), r["arcs"].v.to_numpy()
        for method, col in [("Classifier only", "clf_arc"), ("Classifier + HMM", "hmm_arc")]:
            s = score(pts[col].to_numpy(dtype=float), route, ref_arc, ref_dist, U, V)
            s.update(trip_id=tid, method=method, total_time_s=total)
            rows.append(s)
        cache[tid] = (r, route)
        print(f"  trip {tid}: classifier {rows[-2]['point_accuracy']:.3f} -> HMM {rows[-1]['point_accuracy']:.3f} "
              f"| continuity {rows[-2]['route_continuity']:.3f} -> {rows[-1]['route_continuity']:.3f} ({total:.1f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "student_test_hmm_per_trip.csv"), index=False)
    w = res.n_points
    overall = res.groupby("method").apply(lambda g: pd.Series({
        "point_accuracy (point-weighted)": np.average(g.point_accuracy, weights=g.n_points),
        "point_accuracy (reliable points)": np.average(g.point_accuracy_reliable, weights=g.n_points),
        "route_continuity": g.route_continuity.mean(),
        "route_arc_precision": g.route_arc_precision.mean(),
        "route_arc_recall": g.route_arc_recall.mean(),
        "route_arc_f1": g.route_arc_f1.mean(),
        "mean_time_per_trip_s": g.total_time_s.mean()}), include_groups=False)
    overall.to_csv(os.path.join(OUT, "student_test_hmm_summary.csv"))
    print("\n", overall.round(4).T.to_string())

    piv = res.pivot(index="trip_id", columns="method", values="point_accuracy")[["Classifier only", "Classifier + HMM"]]
    ax = piv.plot.bar(figsize=(10, 4), color=["#ef6c00", "#c62828"], width=0.8)
    ax.set(title="Point accuracy per student_test trip: classifier-only vs classifier + HMM", ylabel="point accuracy", ylim=(0, 1))
    plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(os.path.join(OUT, "hmm_vs_classifier_per_trip.png"), dpi=150); plt.close()
    m = overall[["point_accuracy (point-weighted)", "route_continuity", "route_arc_f1"]].T[["Classifier only", "Classifier + HMM"]]
    ax = m.plot.bar(figsize=(8, 4), color=["#ef6c00", "#c62828"]); ax.set_ylim(0, 1.05)
    ax.set(title="Student-test: classifier-only vs classifier + HMM"); ax.legend(loc="lower right"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "hmm_vs_classifier_summary.png"), dpi=150); plt.close()

    # maps: best, median and worst HMM trips
    order = res[res.method == "Classifier + HMM"].sort_values("point_accuracy").trip_id.tolist()
    picks = list(dict.fromkeys([order[-1], order[len(order) // 2], order[0]]))[:n_maps]
    for tid in picks:
        r, route = cache[tid]
        mp = route_map(r["points"], r["nodes"], r["arcs"], f"student_test trip {tid}",
                       reference_arcs=route, clf_arcs=r["points"].clf_arc, hmm_arcs=r["points"].hmm_arc,
                       network_arcs=nearby_network(r["candidates"]))
        mp.save(os.path.join(MAPS, f"student_test_{tid}.html"))
    print(f"maps saved for trips {picks} in {MAPS}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_features"); ap.add_argument("train_dir"); ap.add_argument("test_dir")
    ap.add_argument("--skip-tuning", action="store_true")
    args = ap.parse_args()
    for d in (OUT, MAPS, ART):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(ART, "model_config.json")) as f:
        cfg = json.load(f)
    if not args.skip_tuning:
        tune_decoder(args.train_features, args.train_dir, cfg)
    evaluate_student_test(args.test_dir)


if __name__ == "__main__":
    main()
