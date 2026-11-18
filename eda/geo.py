"""Small geometry helpers for EDA.

Projection: trips are spread over the whole world (Europe, Russia, USA,
Japan, Vietnam, Brazil, South Africa, ...), so a single national grid such as
EPSG:27700 (British National Grid, valid only for Great Britain) would distort
distances badly. We project each trip into its own local UTM zone, chosen by
pyproj from the trip's bounding box. UTM keeps scale error under ~0.1%, so
distances in metres are trustworthy.
"""
from __future__ import annotations

import numpy as np
import shapely
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def utm_crs_for(lon: np.ndarray, lat: np.ndarray) -> CRS:
    """Best-fitting WGS84 / UTM zone for the given points."""
    aoi = AreaOfInterest(float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat)))
    infos = query_utm_crs_info(datum_name="WGS 84", area_of_interest=aoi)
    if not infos:
        raise ValueError("no UTM zone found for area")
    # several zones can overlap the box; pick the one containing the centre
    centre_zone = int((np.mean(lon) + 180) // 6) + 1
    hemi = "N" if np.mean(lat) >= 0 else "S"
    for info in infos:
        if info.name.endswith(f"zone {centre_zone}{hemi}"):
            return CRS.from_epsg(info.code)
    return CRS.from_epsg(infos[0].code)


class LocalProjection:
    """lon/lat <-> metres in the trip's UTM zone."""

    def __init__(self, lon: np.ndarray, lat: np.ndarray):
        self.crs = utm_crs_for(lon, lat)
        self._fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    def to_metres(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        return self._fwd.transform(np.asarray(lon), np.asarray(lat))


def point_to_segments_distance(px, py, segments_xy: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest of the given straight segments.

    segments_xy: (n_seg, 2, 2) array in metres. Uses a shapely STRtree, so
    cost is ~O((n_points + n_segments) log n_segments), not n_points*n_segments.
    """
    lines = shapely.linestrings(segments_xy)
    tree = shapely.STRtree(lines)
    pts = shapely.points(np.asarray(px), np.asarray(py))
    idx, dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
    out = np.full(len(pts), np.nan)
    out[idx[0]] = dist
    return out
