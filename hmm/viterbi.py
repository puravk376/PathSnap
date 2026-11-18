"""
PathSnap - HMM + Viterbi route reconstruction

Hidden states : at GPS point i, the hidden state is WHICH candidate arc the car is on
                (the candidate set from build_candidates, so state k <-> arc_id is preserved).
Emission      : log P(obs_i | arc) = w * log(classifier probability for that candidate)
Transition    : from candidate a at point i to candidate b at point j (next point with candidates)
                  route_d = distance driven along the road network from a's projected position
                            to b's projected position (directed shortest path between arcs)
                  gps_d   = straight-line distance between the two GPS points
                  log P(b | a) = -|route_d - gps_d| / (beta * max(1, dt))
                A car moving between adjacent GPS points drives about as far as the points are apart,
                so routes that need a long detour (or are impossible) are penalised.
                Impossible moves (no path within the search limit) get a large fixed penalty
                instead of -inf, so the decoder can still "restart" after a GPS jump.
Gaps          : points with no candidates are skipped (the chain links the points either side);
                beta is scaled by the time gap, so long gaps tolerate bigger detours.
Decoding      : standard Viterbi recurrence in log space
                  delta_j(b) = max_a [ delta_i(a) + log T(a->b) ] + log E_j(b)
                with back-pointers, then backtracking from the best final state.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DEFAULT_HMM = dict(beta=10.0, emission_weight=1.0, sp_limit_m=2000.0, impossible_logp=-50.0,
                   backward_tolerance_m=5.0)


def road_graph(U, V, length, n_nodes):
    """Directed graph u->v weighted by arc length (duplicate arcs keep the shortest)."""
    e = pd.DataFrame({"u": U, "v": V, "w": length}).groupby(["u", "v"], as_index=False)["w"].min()
    return csr_matrix((e.w.to_numpy() + 1e-3, (e.u.to_numpy(), e.v.to_numpy())), shape=(n_nodes, n_nodes))


def shortest_paths(graph, sources, targets, limit, chunk=200):
    """Network distance from each source node to each target node (inf if > limit)."""
    sources, targets = np.asarray(sources), np.asarray(targets)
    out = np.full((len(sources), len(targets)), np.inf, dtype=np.float32)
    for s in range(0, len(sources), chunk):
        d = dijkstra(graph, directed=True, indices=sources[s:s + chunk], limit=limit)
        out[s:s + chunk] = d[:, targets]
    return out


def viterbi_trip(cands, proba, gps_xy, gps_t, U, V, length, n_nodes, params=None):
    """
    cands : DataFrame with observation_id, arc_id, proj_t for ONE trip (rows aligned with proba)
    returns: Series observation_id -> decoded arc_id (only for points that have candidates)
    """
    p = {**DEFAULT_HMM, **(params or {})}
    df = cands[["observation_id", "arc_id", "proj_t"]].copy()
    df["logE"] = p["emission_weight"] * np.log(np.clip(proba, 1e-6, 1.0))
    df = df.sort_values(["observation_id"], kind="stable")
    obs_ids = df["observation_id"].unique()
    groups = [g for _, g in df.groupby("observation_id", sort=True)]

    arcs_used = df["arc_id"].unique()
    src_nodes, tgt_nodes = np.unique(V[arcs_used]), np.unique(U[arcs_used])
    sp = shortest_paths(road_graph(U, V, length, n_nodes), src_nodes, tgt_nodes, p["sp_limit_m"])
    src_ix = {n: i for i, n in enumerate(src_nodes)}
    tgt_ix = {n: i for i, n in enumerate(tgt_nodes)}

    first = groups[0]
    delta = first["logE"].to_numpy()
    backs = []
    for k in range(1, len(groups)):
        a_g, b_g = groups[k - 1], groups[k]
        i, j = a_g["observation_id"].iat[0], b_g["observation_id"].iat[0]
        a, ta = a_g["arc_id"].to_numpy(), a_g["proj_t"].to_numpy()
        b, tb = b_g["arc_id"].to_numpy(), b_g["proj_t"].to_numpy()
        La, Lb = length[a], length[b]
        net = sp[np.array([src_ix[n] for n in V[a]])[:, None], np.array([tgt_ix[n] for n in U[b]])[None, :]]
        route_d = (1 - ta)[:, None] * La[:, None] + net + (tb * Lb)[None, :]
        same = a[:, None] == b[None, :]
        along = (tb[None, :] - ta[:, None]) * La[:, None]                # metres moved along the same arc
        same_d = np.where(along >= -p["backward_tolerance_m"], np.abs(along), np.inf)
        route_d = np.where(same, np.minimum(same_d, route_d), route_d)
        gps_d = float(np.hypot(*(gps_xy[j] - gps_xy[i])))
        dt = max(1.0, float(gps_t[j] - gps_t[i]))
        logT = -np.abs(route_d - gps_d) / (p["beta"] * dt)
        logT = np.where(np.isfinite(logT), logT, p["impossible_logp"])
        score = delta[:, None] + logT
        back = np.argmax(score, axis=0)
        delta = score[back, np.arange(len(b))] + b_g["logE"].to_numpy()
        backs.append(back)

    path = [int(np.argmax(delta))]
    for back in reversed(backs):
        path.append(int(back[path[-1]]))
    path.reverse()
    decoded = [g["arc_id"].iat[s] for g, s in zip(groups, path)]
    return pd.Series(decoded, index=obs_ids, name="hmm_arc")
