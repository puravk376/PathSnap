"""
PathSnap - Folium map builder (used by the evaluation scripts and the UI).

route_map(...) draws, as separate toggleable layers:
  * raw GPS points           (blue dots)
  * nearby road network      (grey lines)
  * reference route          (green, only where allowed: train/student_test, never in the evaluator UI)
  * classifier-only route    (orange)
  * HMM/Viterbi route        (red)
with a legend, layer control and a zoom fitted to the trajectory.
"""
import numpy as np
import folium


def _arc_lines(arc_ids, nodes, arcs):
    lines = []
    for a in arc_ids:
        u, v = arcs.u.iat[a], arcs.v.iat[a]
        lines.append([(nodes.lat.iat[u], nodes.lon.iat[u]), (nodes.lat.iat[v], nodes.lon.iat[v])])
    return lines


def ordered_unique(arcs_seq):
    """Arc sequence with consecutive duplicates removed (one entry per arc visit)."""
    out = []
    for a in arcs_seq:
        if a == a and (not out or out[-1] != a):   # a == a filters NaN
            out.append(int(a))
    return out


def route_map(points, nodes, arcs, title, reference_arcs=None, clf_arcs=None, hmm_arcs=None,
              network_arcs=None, point_step=2):
    lat, lon = points["lat"].to_numpy(), points["lon"].to_numpy()
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=15, tiles="OpenStreetMap")
    legend = []
    if network_arcs is not None:
        g = folium.FeatureGroup(name="Road network (nearby)")
        for line in _arc_lines(network_arcs, nodes, arcs):
            folium.PolyLine(line, color="#757575", weight=1, opacity=0.5).add_to(g)
        g.add_to(m); legend.append(("#757575", "&#9473;", "Road network"))
    if reference_arcs is not None:
        g = folium.FeatureGroup(name="Reference route")
        for line in _arc_lines(reference_arcs, nodes, arcs):
            folium.PolyLine(line, color="#2e7d32", weight=8, opacity=0.55).add_to(g)
        g.add_to(m); legend.append(("#2e7d32", "&#9473;", "Reference route"))
    if clf_arcs is not None:
        g = folium.FeatureGroup(name="Classifier-only route", show=hmm_arcs is None)
        for line in _arc_lines(ordered_unique(clf_arcs), nodes, arcs):
            folium.PolyLine(line, color="#ef6c00", weight=4, opacity=0.9).add_to(g)
        g.add_to(m); legend.append(("#ef6c00", "&#9473;", "Classifier-only route"))
    if hmm_arcs is not None:
        g = folium.FeatureGroup(name="HMM/Viterbi route")
        for line in _arc_lines(ordered_unique(hmm_arcs), nodes, arcs):
            folium.PolyLine(line, color="#c62828", weight=4, opacity=0.95).add_to(g)
        g.add_to(m); legend.append(("#c62828", "&#9473;", "HMM/Viterbi predicted route"))
    g = folium.FeatureGroup(name="Raw GPS points")
    sub = points.iloc[::point_step]
    for la, lo, t in sub[["lat", "lon", "t"]].itertuples(index=False):
        folium.CircleMarker((la, lo), radius=2, color="#1565c0", fill=True, fill_opacity=0.9,
                            tooltip=f"t={t:.0f}s").add_to(g)
    g.add_to(m); legend.append(("#1565c0", "&#9679;", f"GPS points (every {point_step})"))
    folium.Marker((lat[0], lon[0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="End", icon=folium.Icon(color="black")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    items = "".join(f"<span style='color:{c}'>{s}</span> {lbl}<br>" for c, s, lbl in legend)
    m.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px 10px;"
        f"border:1px solid #999;border-radius:4px;font-size:13px'><b>{title}</b><br>{items}</div>"))
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    return m


def nearby_network(candidates, limit=4000):
    """Arcs that appeared as candidates = the relevant road network around the trajectory."""
    arcs = np.unique(candidates["arc_id"].to_numpy())
    return arcs[:limit]
