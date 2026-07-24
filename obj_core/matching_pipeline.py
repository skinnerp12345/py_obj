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
from typing import Literal

import numpy as np

from ..time_utils import nearest_within_tolerance
from .identify import precompute_grid_geometry
from .match_io import MatchResult, write_match_file
from .matching import match_objects_one_timestep
from .object_io import (
    iter_object_slices,
    iter_object_slices_lazy,
    read_grid_shape,
    read_init_time_only,
    read_object_file,
)

FileGrouping = Literal["per_time", "init_snapshot"]


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
    file_grouping: FileGrouping = "per_time",
) -> MatchingSummary:
    """Match every forecast (member, time) slice against the nearest truth
    slice within `max_time_offset_minutes`.

    truth_files: paths to Step 4 object files, any of the four shapes --
    unpacked into individual (member, time) slices via iter_object_slices()
    regardless of how they happened to be grouped on disk. Held fully in
    memory for the whole run (truth files are expected to stay small -- e.g.
    one hour of a single truth grid).

    forecast_files: same four shapes, but read via iter_object_slices_lazy()
    and matched immediately as each slice is read (rather than first
    collecting every slice's full label array into memory) -- a large
    ensemble/long-forecast case file (file_grouping="init_snapshot") can be
    multiple GB once its `labels` array is decompressed, and only ever needs
    one (member, time) slice in memory at a time for matching. Only the
    small resulting match records are accumulated until a given forecast
    valid_time's records (across every member) are complete, at which point
    that valid_time's file is written -- see object_io.iter_object_slices_lazy
    for the measured memory numbers this avoids.

    file_grouping: "per_time" (default) writes one match file per distinct
    forecast valid_time (all members for that time in one file, via
    write_match_file's automatic member-dimension handling) -- unchanged
    behavior from before this parameter existed. "init_snapshot" instead
    accumulates every (member, time) result from this whole call into ONE
    file (both member and time dimensions, mirroring object files'
    file_grouping="init_snapshot" shape exactly) -- named after the forecast
    case's own init_time, which every forecast_files entry must therefore
    share (raises if any lack an init_time or if more than one distinct
    init_time is present, since the output filename couldn't otherwise be
    unambiguous).

    A forecast valid_time with no truth valid_time within tolerance is skipped
    (not silently dropped) and reported in the returned MatchingSummary.
    """
    if not truth_files:
        raise ValueError("run_matching_series: truth_files must be non-empty")
    if not forecast_files:
        raise ValueError("run_matching_series: forecast_files must be non-empty")

    os.makedirs(output_dir, exist_ok=True)

    # collect every truth (time, labels, objects) slice, keyed by valid_time --
    # also record which specific file each valid_time actually came from, so
    # output files can report an accurate (not merely "every truth file ever
    # passed in") source-file count.
    truth_by_time: dict[datetime, tuple[np.ndarray, list]] = {}
    truth_file_for_time: dict[datetime, str] = {}
    grid_lat2d = grid_lon2d = None
    for f in truth_files:
        contents = read_object_file(f)
        if grid_lat2d is None:
            grid_lat2d, grid_lon2d = contents.lat2d, contents.lon2d
        elif contents.lat2d.shape != grid_lat2d.shape:
            raise ValueError(f"'{f}': grid shape {contents.lat2d.shape} does not match earlier truth files' {grid_lat2d.shape}")
        for member_id, vt, labels2d, objects in iter_object_slices(contents):
            truth_by_time[vt] = (labels2d, objects)
            truth_file_for_time[vt] = f

    truth_times = sorted(truth_by_time.keys())
    grid_geometry = precompute_grid_geometry(grid_lat2d, grid_lon2d)

    # match each forecast (member, time) slice immediately as it is read --
    # only the resulting (small) match records are accumulated per
    # valid_time, never a full label array, so peak memory is bounded by one
    # slice/member-block at a time regardless of how many forecast files or
    # how large any one of them is.
    results_by_time: dict[datetime, list[MatchResult]] = defaultdict(list)
    forecast_files_for_time: dict[datetime, set[str]] = defaultdict(set)
    # keyed by the FORECAST's own valid_time (not the truth's) -- the two
    # differ whenever there's real jitter within max_time_offset_minutes, so
    # this must be recorded directly at match time rather than reconstructed
    # afterward by reusing truth_file_for_time with a forecast-time key (a
    # real bug caught by running this against actual jittered MRMS data: a
    # forecast valid at 00:00:00 matched against a truth file whose own real
    # valid_time was 00:00:41, so truth_file_for_time had no entry for
    # 00:00:00 at all).
    truth_file_used_for_forecast_time: dict[datetime, str] = {}
    skipped_forecast_times: list[datetime] = []
    skipped_forecast_times_seen: set[datetime] = set()

    for f in forecast_files:
        forecast_grid_shape = read_grid_shape(f)
        if forecast_grid_shape != grid_lat2d.shape:
            raise ValueError(f"'{f}': grid shape {forecast_grid_shape} does not match truth files' {grid_lat2d.shape}")

        for member_id, vt, forecast_labels, forecast_objects in iter_object_slices_lazy(f):
            nearest_truth_time = nearest_within_tolerance(vt, truth_times, max_time_offset_minutes)
            if nearest_truth_time is None:
                if vt not in skipped_forecast_times_seen:
                    skipped_forecast_times.append(vt)
                    skipped_forecast_times_seen.add(vt)
                continue

            truth_labels, truth_objects = truth_by_time[nearest_truth_time]
            records = match_objects_one_timestep(
                truth_objects, truth_labels, forecast_objects, forecast_labels, grid_geometry,
                max_boundary_disp_km, max_centroid_disp_km, ti_threshold,
            )
            results_by_time[vt].append(MatchResult(records=records, valid_time=vt, member_id=member_id))
            forecast_files_for_time[vt].add(f)
            truth_file_used_for_forecast_time[vt] = truth_file_for_time[nearest_truth_time]

    output_paths = []
    if file_grouping == "per_time":
        for forecast_time in sorted(results_by_time.keys()):
            n_truth_source_files = 1  # per_time: exactly one truth file feeds any one output time, by construction
            n_forecast_source_files = len(forecast_files_for_time[forecast_time])
            out_path = os.path.join(output_dir, f"match_{forecast_time:%Y%m%d_%H%M%S}.nc")
            write_match_file(
                out_path, results_by_time[forecast_time],
                n_truth_source_files=n_truth_source_files, n_forecast_source_files=n_forecast_source_files,
                max_boundary_disp_km=max_boundary_disp_km, max_centroid_disp_km=max_centroid_disp_km,
                ti_threshold=ti_threshold, max_time_offset_minutes=max_time_offset_minutes,
            )
            output_paths.append(out_path)
    elif file_grouping == "init_snapshot":
        if results_by_time:
            init_times = {read_init_time_only(f) for f in forecast_files}
            if None in init_times:
                raise ValueError("run_matching_series: file_grouping='init_snapshot' requires every forecast file to have an init_time")
            if len(init_times) > 1:
                raise ValueError(f"run_matching_series: file_grouping='init_snapshot' requires one shared init_time across forecast_files, found {sorted(init_times)}")
            init_time = init_times.pop()

            all_results = [r for results in results_by_time.values() for r in results]
            truth_files_used = {truth_file_used_for_forecast_time[vt] for vt in results_by_time}
            forecast_files_used = set().union(*forecast_files_for_time.values()) if forecast_files_for_time else set()

            out_path = os.path.join(output_dir, f"match_init_{init_time:%Y%m%d_%H%M%S}.nc")
            write_match_file(
                out_path, all_results,
                n_truth_source_files=len(truth_files_used), n_forecast_source_files=len(forecast_files_used),
                max_boundary_disp_km=max_boundary_disp_km, max_centroid_disp_km=max_centroid_disp_km,
                ti_threshold=ti_threshold, max_time_offset_minutes=max_time_offset_minutes,
            )
            output_paths.append(out_path)
    else:
        raise ValueError(f"run_matching_series: unknown file_grouping '{file_grouping}'")

    return MatchingSummary(output_paths=output_paths, skipped_forecast_times=skipped_forecast_times)
