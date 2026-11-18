"""Load PathSnap trips (GPS track, road network, true route) into pandas.

Dataset layout (one folder per trip)::

    data/
      train/<trip_id>/<trip_id>.{track,nodes,arcs,route}         (80 trips)
      student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}  (10 trips)

File formats (whitespace separated, no header) - verified on all 90 trips:

* ``.track``  ``lon lat time``  one GPS fix per line, time in seconds
* ``.nodes``  ``lon lat``       road-graph vertex; node id = 0-based line number
* ``.arcs``   ``from to``       directed road piece between two node ids;
                                arc id = 0-based line number
* ``.route``  ``arc_id``        the true route, as arc ids in driving order

NOTE: the files store LONGITUDE FIRST, then latitude (some trips have
|first column| > 90, e.g. Vietnam at 106.7 E). Every loader here returns
explicit ``lon`` / ``lat`` columns so nobody has to remember the order.

This module is shared: Person B (candidates/labels) can import it directly::

    from eda.loader import load_trip
    trip = load_trip("00000001")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "student_test")


def data_dir() -> Path:
    """Dataset root: $PATHSNAP_DATA if set, otherwise <repo>/data."""
    return Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT / "data"))


def _read(path: Path, names: list[str], dtypes: dict) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtypes, engine="c")


def read_track(path: Path) -> pd.DataFrame:
    df = _read(path, ["lon", "lat", "time"], {"lon": float, "lat": float, "time": float})
    df.index.name = "point_id"
    return df


def read_nodes(path: Path) -> pd.DataFrame:
    df = _read(path, ["lon", "lat"], {"lon": float, "lat": float})
    df.index.name = "node_id"
    return df


def read_arcs(path: Path) -> pd.DataFrame:
    df = _read(path, ["from_node", "to_node"], {"from_node": np.int64, "to_node": np.int64})
    df.index.name = "arc_id"
    return df


def read_route(path: Path) -> pd.Series:
    df = _read(path, ["arc_id"], {"arc_id": np.int64})
    return df["arc_id"].rename("arc_id")


@dataclass
class Trip:
    trip_id: str
    split: str
    track: pd.DataFrame          # point_id -> lon, lat, time
    nodes: pd.DataFrame          # node_id  -> lon, lat
    arcs: pd.DataFrame           # arc_id   -> from_node, to_node
    route: pd.Series             # ordered arc ids of the true route
    folder: Path = field(repr=False, default=None)

    @cached_property
    def arc_coords(self) -> np.ndarray:
        """(n_arcs, 2, 2) array: [arc, end(0=from,1=to), (lon, lat)]."""
        xy = self.nodes[["lon", "lat"]].to_numpy()
        a = self.arcs.to_numpy()
        return np.stack([xy[a[:, 0]], xy[a[:, 1]]], axis=1)

    @cached_property
    def route_coords(self) -> np.ndarray:
        """(n_route_arcs, 2, 2) coordinates of the true-route arcs, in order."""
        return self.arc_coords[self.route.to_numpy()]

    def route_polyline(self) -> list[np.ndarray]:
        """True route as lon/lat polylines. A new piece starts wherever two
        consecutive route arcs do not share a node (rare in this dataset)."""
        a = self.arcs.to_numpy()[self.route.to_numpy()]
        xy = self.nodes[["lon", "lat"]].to_numpy()
        pieces, current = [], [a[0, 0], a[0, 1]]
        for prev, cur in zip(a[:-1], a[1:]):
            if cur[0] == prev[1]:
                current.append(cur[1])
            else:
                pieces.append(xy[current])
                current = [cur[0], cur[1]]
        pieces.append(xy[current])
        return pieces


def trip_folder(trip_id: str, split: str | None = None) -> tuple[Path, str]:
    root = data_dir()
    for s in ([split] if split else SPLITS):
        folder = root / s / trip_id
        if folder.is_dir():
            return folder, s
    raise FileNotFoundError(f"trip {trip_id!r} not found under {root} (splits: {SPLITS})")


def load_trip(trip_id: str, split: str | None = None) -> Trip:
    folder, split = trip_folder(trip_id, split)
    return Trip(
        trip_id=trip_id,
        split=split,
        track=read_track(folder / f"{trip_id}.track"),
        nodes=read_nodes(folder / f"{trip_id}.nodes"),
        arcs=read_arcs(folder / f"{trip_id}.arcs"),
        route=read_route(folder / f"{trip_id}.route"),
        folder=folder,
    )


def list_trips(split: str | None = None) -> pd.DataFrame:
    """All trip ids with their split, sorted (train first, then test)."""
    root = data_dir()
    rows = []
    for s in ([split] if split else SPLITS):
        d = root / s
        if not d.is_dir():
            raise FileNotFoundError(f"missing split folder {d}. Put the dataset in {root} "
                                    f"or set PATHSNAP_DATA.")
        rows += [{"trip_id": p.name, "split": s} for p in sorted(d.iterdir()) if p.is_dir()]
    return pd.DataFrame(rows)


def iter_trips(split: str | None = None):
    for row in list_trips(split).itertuples():
        yield load_trip(row.trip_id, row.split)
