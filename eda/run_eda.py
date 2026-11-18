"""Run the whole EDA in one go.

    python -m eda.run_eda

Writes everything into results/eda/:
    trip_summary.csv / .md, split_summary.csv   - the tables
    figures/*.png                                - the graphs
    maps/trip_<id>.html, maps/overview_all_trips.html - the Folium maps
"""
from __future__ import annotations

import argparse
import time

from eda import plots, trip_map
from eda.trip_summary import OUT_DIR, run, save_tables

# Trip shown at the evaluation: a 24-minute drive through Horsens (Denmark).
# Dense town-centre streets (~20 candidate roads per fix), clear turns, no time
# gaps, and simply snapping to the nearest road is wrong for 15% of the fixes.
SHOWCASE_TRIP = "00000061"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--showcase", default=SHOWCASE_TRIP, help="trip id for the main Folium map")
    ap.add_argument("--no-maps", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("1/3  analysing all trips ...")
    summary, details = run()
    save_tables(summary)
    print(f"     tables -> {OUT_DIR}")

    print("2/3  drawing figures ...")
    for p in plots.make_all(summary, details):
        print("     ", p)

    if not args.no_maps:
        print("3/3  building Folium maps ...")
        ids = [args.showcase] + plots.pick_gallery_trips(summary)[:3]
        ids += summary.loc[summary["off_route_trip"], "trip_id"].tolist()  # the data problems
        ids = list(dict.fromkeys(ids))  # unique, keep order
        for tid in ids:
            print("     ", trip_map.trip_map(tid))
        print("     ", trip_map.overview_map(summary))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
