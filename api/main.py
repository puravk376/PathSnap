"""
PathSnap - FastAPI backend (runs on LAPTOP 1)

Holds the trained classifier, the HMM/Viterbi decoder, the road networks and the evaluator
trips. The UI (laptop 2) only sends a trip ID; everything else happens here.

Endpoints
  GET  /health            -> status + loaded model info
  GET  /trips             -> the evaluator trip IDs available on this server
  POST /predict           -> body {"trip_id": "00000015"}; runs the full pipeline
                             (candidates -> features -> classifier -> HMM/Viterbi),
                             then scores the prediction against the server-side reference.
                             Returns GPS points, predicted route geometry and metrics.
                             The reference route is NEVER returned.

Run (from the repo root):
  set PATHSNAP_EVAL_DIR=data\\evaluator_test        (Windows)   export PATHSNAP_EVAL_DIR=data/evaluator_test  (Mac/Linux)
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import os
import re
import sys
import time
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pipeline.inference import Predictor  # noqa: E402
from pipeline.scoring import reference_point_arcs, score  # noqa: E402
from visualization.maps import ordered_unique  # noqa: E402

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(REPO, "data", "evaluator_test"))
ARTIFACTS = os.environ.get("PATHSNAP_ARTIFACTS", os.path.join(REPO, "artifacts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = Predictor(ARTIFACTS)
log.info("model loaded: %s | features: %d | HMM params: %s", predictor.cfg["model"], len(predictor.features), predictor.hmm)
log.info("evaluator trips folder: %s", EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str


def list_trips():
    if not os.path.isdir(EVAL_DIR):
        return []
    return sorted(d for d in os.listdir(EVAL_DIR) if os.path.isdir(os.path.join(EVAL_DIR, d)))


def lines(arc_ids, nodes, arcs):
    la, lo, U, V = nodes.lat.to_numpy(), nodes.lon.to_numpy(), arcs.u.to_numpy(), arcs.v.to_numpy()
    return [[[float(la[U[a]]), float(lo[U[a]])], [float(la[V[a]]), float(lo[V[a]])]] for a in arc_ids]


def clean(d):
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("REQUEST %s %s from %s -> %d (%.2fs)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code, time.perf_counter() - t0)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.cfg["model"], "n_features": len(predictor.features),
            "hmm": predictor.hmm, "n_trips": len(list_trips())}


@app.get("/trips")
def trips():
    return {"trips": list_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if not re.fullmatch(r"\d{1,8}", tid) or tid.zfill(8) not in list_trips():
        raise HTTPException(404, f"trip {tid} not found on server")
    tid = tid.zfill(8)
    trip_dir = os.path.join(EVAL_DIR, tid)
    log.info("PREDICT trip %s: running candidates -> features -> classifier -> HMM", tid)
    t0 = time.perf_counter()
    r = predictor.predict_trip(trip_dir)
    pts, nodes, arcs = r["points"], r["nodes"], r["arcs"]
    predict_s = time.perf_counter() - t0

    # ---- server-side scoring (reference stays on this laptop)
    route, ref_arc, ref_dist = reference_point_arcs(trip_dir, pts, nodes, arcs)
    U, V = arcs.u.to_numpy(), arcs.v.to_numpy()
    m_clf = score(pts.clf_arc.to_numpy(dtype=float), route, ref_arc, ref_dist, U, V)
    m_hmm = score(pts.hmm_arc.to_numpy(dtype=float), route, ref_arc, ref_dist, U, V)
    log.info("RESULT trip %s: %d points | classifier acc %.4f | HMM acc %.4f | continuity %.4f -> %.4f | %.2fs",
             tid, len(pts), m_clf["point_accuracy"], m_hmm["point_accuracy"],
             m_clf["route_continuity"], m_hmm["route_continuity"], predict_s)

    hmm_seq, clf_seq = ordered_unique(pts.hmm_arc), ordered_unique(pts.clf_arc)
    network = np.unique(r["candidates"]["arc_id"].to_numpy())[:4000]
    return {
        "trip_id": tid,
        "n_points": int(len(pts)),
        "gps": pts[["lat", "lon", "t"]].round(6).values.tolist(),
        "predicted_route": lines(hmm_seq, nodes, arcs),
        "classifier_route": lines(clf_seq, nodes, arcs),
        "network": lines(network, nodes, arcs),
        "predicted_arc_per_point": [None if a != a else int(a) for a in pts.hmm_arc],
        "metrics": {"classifier_only": clean(m_clf), "classifier_hmm": clean(m_hmm)},
        "timing_s": {**{k: round(v, 3) for k, v in r["timing"].items()}, "total": round(predict_s, 3)},
    }
