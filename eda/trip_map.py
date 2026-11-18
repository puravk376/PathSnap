"""Interactive Folium map of one trip: blue GPS dots, grey roads, green true route.

    python -m eda.trip_map 00000045            # -> results/eda/maps/trip_00000045.html
    python -m eda.trip_map 00000045 00000007   # several at once
    python -m eda.trip_map --overview          # world map of all 90 trips

Open the .html file in any browser (needs internet for the background tiles).
Layers can be switched on/off in the top-right control; hover a blue dot to see
its index, timestamp and distance to the true route.
"""
from __future__ import annotations

import argparse
import html

import folium
import numpy as np
import pandas as pd
from branca.element import Element

from eda.geo import LocalProjection, point_to_segments_distance
from eda.loader import REPO_ROOT, list_trips, load_trip

MAP_DIR = REPO_ROOT / "results" / "eda" / "maps"

GPS = "#2a78d6"     # blue
ROUTE = "#008300"   # green
ROAD = "#8f8d86"    # grey (a bit darker than in the PNGs so it shows on light tiles)
OFF = "#e34948"     # red ring: fix far from the true route
ROAD_MARGIN_M = 300  # draw roads only this far around the trip, keeps the file small
MAX_DOTS = 6000      # thin very long tracks so the page stays responsive
FAR_M = 20           # highlight fixes further than this from the true route


def _legend_html(trip, stats: dict) -> str:
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td style='text-align:right'><b>{html.escape(v)}</b></td></tr>"
                   for k, v in stats.items())
    swatch = ("<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
              "<span style='{s}'></span>{label}</div>")
    return f"""
<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:rgba(255,255,255,.95);
            padding:12px 14px;border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.25);
            font:13px/1.35 system-ui,Segoe UI,Arial,sans-serif;color:#0b0b0b;min-width:230px">
  <div style="font-weight:700;font-size:14px;margin-bottom:6px">PathSnap - trip {trip.trip_id}
    <span style="font-weight:400;color:#52514e">({'test' if trip.split == 'student_test' else 'train'})</span></div>
  {swatch.format(s=f'width:10px;height:10px;border-radius:50%;background:{GPS};display:inline-block', label='GPS fix (observed)')}
  {swatch.format(s=f'width:22px;height:5px;background:{ROUTE};display:inline-block', label='true route (ground truth)')}
  {swatch.format(s=f'width:22px;height:2px;background:{ROAD};display:inline-block', label='road network')}
  {swatch.format(s=f'width:9px;height:9px;border-radius:50%;border:2px solid {OFF};display:inline-block', label=f'fix &gt; {FAR_M} m from route')}
  <table style="margin-top:6px;border-top:1px solid #e4e3df;padding-top:4px;width:100%;font-size:12px">{rows}</table>
</div>"""


def _add_basemaps(m: folium.Map) -> None:
    # Esri light grey needs no API key and lets the grey/green/blue layers stand out.
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ", name="Light grey basemap", max_zoom=16,
        max_native_zoom=16).add_to(m)
    folium.TileLayer("openstreetmap", name="OpenStreetMap", max_zoom=19, show=False).add_to(m)


def _points_layer(idx, lon, lat, time, d_route, marker) -> folium.GeoJson:
    """All fixes as ONE GeoJSON layer (much smaller html than one marker per fix)."""
    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(float(lon[i]), 6), round(float(lat[i]), 6)]},
        "properties": {"fix": int(i), "t": f"{time[i]:.0f} s", "d": f"{d_route[i]:.1f} m"},
    } for i in idx]
    return folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        marker=marker,
        tooltip=folium.GeoJsonTooltip(["fix", "t", "d"], aliases=["GPS fix", "time", "distance to true route"]),
    )


def trip_map(trip_id: str, out_dir=MAP_DIR) -> str:
    trip = load_trip(trip_id)
    t = trip.track
    lon, lat, time = t["lon"].to_numpy(), t["lat"].to_numpy(), t["time"].to_numpy()

    # --- metres, for clipping and for distances ---------------------------
    proj = LocalProjection(lon, lat)
    px, py = proj.to_metres(lon, lat)
    nx, ny = proj.to_metres(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy())
    a = trip.arcs.to_numpy()
    seg_m = np.stack([np.column_stack([nx[a[:, 0]], ny[a[:, 0]]]),
                      np.column_stack([nx[a[:, 1]], ny[a[:, 1]]])], axis=1)
    route_ids = np.unique(trip.route.to_numpy())
    d_route = point_to_segments_distance(px, py, seg_m[route_ids])

    # roads near the trip, each two-way street drawn once
    x0, x1 = px.min() - ROAD_MARGIN_M, px.max() + ROAD_MARGIN_M
    y0, y1 = py.min() - ROAD_MARGIN_M, py.max() + ROAD_MARGIN_M
    near = ((seg_m[:, :, 0].max(1) >= x0) & (seg_m[:, :, 0].min(1) <= x1)
            & (seg_m[:, :, 1].max(1) >= y0) & (seg_m[:, :, 1].min(1) <= y1))
    near_ids = np.flatnonzero(near)
    _, first = np.unique(np.sort(a[near_ids], axis=1), axis=0, return_index=True)
    road_ids = near_ids[np.sort(first)]
    ll = trip.arc_coords  # [arc, end, (lon, lat)]
    road_lines = np.round(ll[road_ids][:, :, ::-1], 6).tolist()  # [[lat, lon], [lat, lon]] per road

    # --- map -------------------------------------------------------------------
    m = folium.Map(tiles=None, control_scale=True, prefer_canvas=True)
    _add_basemaps(m)

    roads = folium.FeatureGroup(name=f"Road network ({len(road_ids):,} pieces shown)")
    folium.PolyLine(road_lines, color=ROAD, weight=1.5, opacity=0.9).add_to(roads)
    roads.add_to(m)

    route = folium.FeatureGroup(name=f"True route ({len(trip.route):,} arcs)")
    for piece in trip.route_polyline():
        folium.PolyLine(piece[:, ::-1].tolist(), color=ROUTE, weight=7, opacity=0.8,
                        tooltip="true route").add_to(route)
    route.add_to(m)

    stride = max(1, int(np.ceil(len(t) / MAX_DOTS)))
    dots = folium.FeatureGroup(name="GPS fixes" + (f" (every {stride}th)" if stride > 1 else ""))
    _points_layer(np.arange(0, len(t), stride), lon, lat, time, d_route,
                  folium.CircleMarker(radius=2.5, color=GPS, weight=0, fill=True,
                                      fill_color=GPS, fill_opacity=0.75)).add_to(dots)
    dots.add_to(m)

    far_idx = np.flatnonzero(d_route > FAR_M)
    far = folium.FeatureGroup(name=f"Fixes > {FAR_M} m from route ({len(far_idx):,})")
    if len(far_idx):
        _points_layer(far_idx[:: max(1, int(np.ceil(len(far_idx) / MAX_DOTS)))], lon, lat, time, d_route,
                      folium.CircleMarker(radius=7, color=OFF, weight=2, fill=False)).add_to(far)
    far.add_to(m)

    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    dt = np.diff(time)
    stats = {
        "GPS fixes": f"{len(t):,}",
        "duration": f"{(time[-1] - time[0]) / 60:.1f} min",
        "max time gap": f"{dt.max():.0f} s" if len(dt) else "-",
        "route arcs": f"{len(trip.route):,}",
        "roads in map file": f"{len(a):,}",
        "median distance to route": f"{np.median(d_route):.1f} m",
        "95th pct distance": f"{np.percentile(d_route, 95):.1f} m",
    }
    m.get_root().html.add_child(Element(_legend_html(trip, stats)))
    m.get_root().header.add_child(Element(f"<title>PathSnap trip {trip.trip_id}</title>"))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trip_{trip.trip_id}.html"
    m.save(str(path))
    return str(path.relative_to(REPO_ROOT))


def overview_map(summary: pd.DataFrame | None = None, out_dir=MAP_DIR) -> str:
    """World map with one marker per trip (click for its numbers)."""
    if summary is None:
        summary = pd.read_csv(REPO_ROOT / "results" / "eda" / "trip_summary.csv", dtype={"trip_id": str})
    m = folium.Map(location=[30, 10], zoom_start=2, tiles=None, world_copy_jump=True)
    _add_basemaps(m)
    colours = {"train": GPS, "student_test": "#eb6834"}
    for split, g in summary.groupby("split"):
        fg = folium.FeatureGroup(name=f"{'test' if split == 'student_test' else 'train'} ({len(g)} trips)")
        for r in g.itertuples():
            folium.CircleMarker(
                [r.centre_lat, r.centre_lon], radius=7, color="white", weight=1.5, fill=True,
                fill_color=colours[split], fill_opacity=0.95,
                tooltip=f"trip {r.trip_id} ({split})",
                popup=folium.Popup(
                    f"<b>trip {r.trip_id}</b> ({split})<br>{r.n_points:,} GPS fixes, "
                    f"{r.duration_min:.0f} min<br>{r.n_arcs:,} road arcs, route {r.route_km:.1f} km<br>"
                    f"median GPS noise {r.dist_to_route_median_m:.1f} m", max_width=260),
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().header.add_child(Element("<title>PathSnap - all trips</title>"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "overview_all_trips.html"
    m.save(str(path))
    return str(path.relative_to(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trip_ids", nargs="*", help="trip ids, e.g. 00000045")
    ap.add_argument("--overview", action="store_true", help="also build the world overview map")
    ap.add_argument("--all-test", action="store_true", help="map every test trip")
    args = ap.parse_args()
    ids = list(args.trip_ids)
    if args.all_test:
        ids += list_trips("student_test")["trip_id"].tolist()
    if not ids and not args.overview:
        ap.error("give at least one trip id, --all-test or --overview")
    for tid in ids:
        print("wrote", trip_map(tid))
    if args.overview:
        print("wrote", overview_map())


if __name__ == "__main__":
    main()
