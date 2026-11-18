"""
PathSnap - Evaluator UI (runs on LAPTOP 2)

Talks ONLY to the API on laptop 1: lists the evaluator trips, sends the selected trip ID,
and shows the predicted route on a Folium map with the evaluation results returned by the API.
It has no model, no data and no reference routes.

Run:
  streamlit run ui/app.py
then enter laptop 1's address, e.g. http://192.168.1.10:8000
"""
import os
import requests
import pandas as pd
import folium
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PathSnap", page_icon="🛰️", layout="wide")
st.title("PathSnap: map-matching and route reconstruction")
st.caption("Pick an evaluation trip. The API on laptop 1 runs the full pipeline and returns the predicted route.")

with st.sidebar:
    st.header("API connection")
    api = st.text_input("API address (laptop 1)", os.environ.get("PATHSNAP_API", "http://localhost:8000")).rstrip("/")
    if st.button("Connect / refresh trips"):
        try:
            h = requests.get(f"{api}/health", timeout=10).json()
            st.session_state["trips"] = requests.get(f"{api}/trips", timeout=10).json()["trips"]
            st.success(f"Connected: {h['model']}, {h['n_trips']} trips")
        except Exception as e:
            st.error(f"Could not reach API: {e}")

trips = st.session_state.get("trips", [])
if not trips:
    st.info("Enter the API address in the sidebar and click **Connect / refresh trips**.")
    st.stop()

st.subheader("Evaluation trips")
trip = st.radio("Select a trip", trips, horizontal=True)

if st.button("Run prediction", type="primary"):
    with st.spinner(f"Sending trip {trip} to the API..."):
        try:
            resp = requests.post(f"{api}/predict", json={"trip_id": trip}, timeout=600)
            resp.raise_for_status()
            st.session_state["result"] = resp.json()
        except Exception as e:
            st.error(f"Prediction failed: {e}")

res = st.session_state.get("result")
if res:
    mc, mh = res["metrics"]["classifier_only"], res["metrics"]["classifier_hmm"]
    st.subheader(f"Results for trip {res['trip_id']} ({res['n_points']} GPS points)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Point accuracy (HMM)", f"{mh['point_accuracy']*100:.1f}%",
              f"{(mh['point_accuracy']-mc['point_accuracy'])*100:+.1f} pts vs classifier only")
    c2.metric("Route continuity (HMM)", f"{mh['route_continuity']*100:.1f}%",
              f"{(mh['route_continuity']-mc['route_continuity'])*100:+.1f} pts")
    c3.metric("Route arc F1 (HMM)", f"{mh['route_arc_f1']*100:.1f}%")
    c4.metric("Server time", f"{res['timing_s']['total']:.1f} s")

    table = pd.DataFrame({"Classifier only": mc, "Classifier + HMM": mh}).drop(index=["n_points"])
    table.index = ["Point accuracy", "Point accuracy (reliable reference)", "Coverage", "Route continuity",
                   "Route arc precision", "Route arc recall", "Route arc F1"]
    st.dataframe(table.style.format("{:.4f}"))

    gps = res["gps"]
    lats, lons = [g[0] for g in gps], [g[1] for g in gps]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=15, tiles="OpenStreetMap")
    net = folium.FeatureGroup(name="Road network (nearby)")
    for ln in res["network"]:
        folium.PolyLine(ln, color="#757575", weight=1, opacity=0.5).add_to(net)
    net.add_to(m)
    clf = folium.FeatureGroup(name="Classifier-only route", show=False)
    for ln in res["classifier_route"]:
        folium.PolyLine(ln, color="#ef6c00", weight=4, opacity=0.9).add_to(clf)
    clf.add_to(m)
    hmm = folium.FeatureGroup(name="Predicted route (HMM/Viterbi)")
    for ln in res["predicted_route"]:
        folium.PolyLine(ln, color="#c62828", weight=5, opacity=0.9).add_to(hmm)
    hmm.add_to(m)
    pts = folium.FeatureGroup(name="Raw GPS points")
    for la, lo, t in gps[::2]:
        folium.CircleMarker((la, lo), radius=2, color="#1565c0", fill=True, fill_opacity=0.9, tooltip=f"t={t:.0f}s").add_to(pts)
    pts.add_to(m)
    folium.Marker((lats[0], lons[0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((lats[-1], lons[-1]), tooltip="End", icon=folium.Icon(color="black")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px 10px;"
        "border:1px solid #999;border-radius:4px;font-size:13px'>"
        "<span style='color:#1565c0'>&#9679;</span> GPS points<br>"
        "<span style='color:#c62828'>&#9473;</span> Predicted route (HMM)<br>"
        "<span style='color:#ef6c00'>&#9473;</span> Classifier-only route<br>"
        "<span style='color:#757575'>&#9473;</span> Road network</div>"))
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    html = m.get_root().render()
    try:
        components.html(html, height=620)          # Streamlit <= 1.x
    except Exception:
        st.iframe(html, height=620)                # newer Streamlit versions
    with st.expander("Server timing breakdown"):
        st.json(res["timing_s"])
