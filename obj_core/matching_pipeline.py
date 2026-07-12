"""Driver: match a truth object series against a forecast object series over
many timesteps, writing self-contained match-result files.

A genuinely separate process from identification/tracking (Step 4), connected
only via Step 4's object files on disk -- reads truth_files/forecast_files
(paths written by run_object_id_series or produced any other way that
conforms to object_io.py's schema), never re-identifies anything itself.
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from ..time_utils import nearest_within_tolerance
from .identify import precompute_grid_geometry
from .match_io import MatchResult, write_match_file
from .matching import match_objects_one_timestep
from .object_io import iter_object_slices, read_object_file


@dataclass
class MatchingSummary:
    output_paths: list[str]
    skipped_forecast_times: list[datetime]  # no truth time within tolerance


def run_matching_series(
    truth_files: list[str],
    forecast_files: list[str],
    max_boundary_disp_km: float,
    max_centroid_disp_km: float,
    ti_threshold: float,
    output_dir: str,
    max_time_offset_minutes: float = 5.0,
) -> MatchingSummary:
    """Match every forecast (member, time) slice against the nearest truth
    slice within `max_time_offset_minutes`, writing one match file per
    distinct forecast valid_time (all members for that time in one file, via
    write_match_file's automatic member-dimension handling).

    truth_files/forecast_files: paths to Step 4 object files, any of the four
    shapes -- unpacked into individual (member, time) slices via
    iter_object_slices() regardless of how they happened to be grouped on disk.

    A forecast valid_time with no truth valid_time within tolerance is skipped
    (not silently dropped) and reported in the returned MatchingSummary.
    """
    if not truth_files:
        raise ValueError("run_matching_series: truth_files must be non-empty")
    if not forecast_files:
        raise ValueError("run_matching_series: forecast_files must be non-empty")

    os.makedirs(output_dir, exist_ok=True)

    # collect every truth (time, labels, objects) slice, keyed by valid_time
    truth_by_time: dict[datetime, tuple[np.ndarray, list]] = {}
    grid_lat2d = grid_lon2d = None
    for f in truth_files:
        contents = read_object_file(f)
        if grid_lat2d is None:
            grid_lat2d, grid_lon2d = contents.lat2d, contents.lon2d
        elif contents.lat2d.shape != grid_lat2d.shape:
            raise ValueError(f"'{f}': grid shape {contents.lat2d.shape} does not match earlier truth files' {grid_lat2d.shape}")
        for member_id, vt, labels2d, objects in iter_object_slices(contents):
            truth_by_time[vt] = (labels2d, objects)

    truth_times = sorted(truth_by_time.keys())

    # collect every forecast (member, time) slice, grouped by valid_time
    forecast_by_time: dict[datetime, list[tuple[str | None, np.ndarray, list]]] = defaultdict(list)
    for f in forecast_files:
        contents = read_object_file(f)
        if contents.lat2d.shape != grid_lat2d.shape:
            raise ValueError(f"'{f}': grid shape {contents.lat2d.shape} does not match truth files' {grid_lat2d.shape}")
        for member_id, vt, labels2d, objects in iter_object_slices(contents):
            forecast_by_time[vt].append((member_id, labels2d, objects))

    grid_geometry = precompute_grid_geometry(grid_lat2d, grid_lon2d)

    output_paths = []
    skipped_forecast_times = []

    for forecast_time in sorted(forecast_by_time.keys()):
        nearest_truth_time = nearest_within_tolerance(forecast_time, truth_times, max_time_offset_minutes)
        if nearest_truth_time is None:
            skipped_forecast_times.append(forecast_time)
            continue

        truth_labels, truth_objects = truth_by_time[nearest_truth_time]

        results = []
        for member_id, forecast_labels, forecast_objects in forecast_by_time[forecast_time]:
            records = match_objects_one_timestep(
                truth_objects, truth_labels, forecast_objects, forecast_labels, grid_geometry,
                max_boundary_disp_km, max_centroid_disp_km, ti_threshold,
            )
            results.append(MatchResult(records=records, valid_time=forecast_time, member_id=member_id))

        out_path = os.path.join(output_dir, f"match_{forecast_time:%Y%m%d_%H%M%S}.nc")
        write_match_file(
            out_path, results,
            truth_source_files=[f for f in truth_files],
            forecast_source_files=[f for f in forecast_files],
            max_boundary_disp_km=max_boundary_disp_km, max_centroid_disp_km=max_centroid_disp_km,
            ti_threshold=ti_threshold, max_time_offset_minutes=max_time_offset_minutes,
        )
        output_paths.append(out_path)

    return MatchingSummary(output_paths=output_paths, skipped_forecast_times=skipped_forecast_times)
