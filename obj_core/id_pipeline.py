"""Driver: run object identification (+ optional in-time tracking) over a
manifest of input files, writing self-contained object files.

Fully generic over truth vs. forecast series -- nothing here knows or cares
which the manifest represents; that distinction only matters to the (separate)
matching process that later reads whichever two output directories the user
points it at.

The caller resolves its own model/obs-specific file-naming convention into a
plain manifest (list[SeriesEntry]) before calling this; parsing a particular
model's ensemble-naming scheme is not this pipeline's job.
"""

import os
from collections import defaultdict
from datetime import datetime
from typing import Callable, Literal

import numpy as np

from .identify import GridGeometry, StormObject, identify_objects, precompute_grid_geometry
from .object_io import IdentificationResult, SeriesEntry, write_object_file
from .tracking import track_objects_incremental

FileGrouping = Literal["single", "member_series", "ensemble_snapshot", "full"]


def _default_output_name(grouping: FileGrouping, member_id: str | None, valid_time: datetime | None) -> str:
    if grouping == "single":
        member_part = member_id if member_id is not None else "obs"
        return f"obj_{member_part}_{valid_time:%Y%m%d_%H%M%S}.nc"
    if grouping == "member_series":
        member_part = member_id if member_id is not None else "obs"
        return f"obj_{member_part}_series.nc"
    if grouping == "ensemble_snapshot":
        return f"obj_ensemble_{valid_time:%Y%m%d_%H%M%S}.nc"
    if grouping == "full":
        return "obj_full.nc"
    raise ValueError(f"Unknown file_grouping '{grouping}'")


def run_object_id_series(
    manifest: list[SeriesEntry],
    loader_fn: Callable[[SeriesEntry], object],  # -> anything with .lat2d/.lon2d/.data, e.g. GriddedField
    thresh_1: float,
    thresh_2: float,
    area_thresh_km2: float,
    output_dir: str,
    file_grouping: FileGrouping = "single",
    track_in_time: bool = False,
    track_bound_disp_km: float = 0.0,
    mask: np.ndarray | None = None,
    init_time: datetime | None = None,
    linear_eccentricity_thresh: float = 0.8,
    linear_length_thresh_km: float = 200.0,
    mixed_eccentricity_thresh: float = 0.75,
    mixed_length_thresh_km: float = 100.0,
) -> list[str]:
    """Identify (and optionally track) objects for every entry in `manifest`,
    writing output object file(s) under `output_dir` per `file_grouping`.

    loader_fn(entry) -> an object with .lat2d, .lon2d, .data attributes (e.g.
    the GriddedField Step 1/2 loaders already return). Receives the whole
    SeriesEntry (not just its filepath) so a loader can use entry.extra_dim_index
    to extract one slice from a file shared by multiple manifest entries
    (e.g. an ensemble stacked as an array dimension inside one file).

    linear_eccentricity_thresh/linear_length_thresh_km/mixed_eccentricity_thresh/
    mixed_length_thresh_km: passed straight through to identify_objects() for
    the three-way linear/mixed/cellular classification (see
    StormObject.is_linear there).

    Tracking, when enabled, only ever links objects across consecutive times
    *within the same member* -- age/track_id along a fixed member's own
    timeline. Different members (or entries with no member_id, i.e. an obs
    series) are tracked independently of one another. track_id values are
    unique across the whole call, not just within one member, so consolidated
    files (e.g. "full") never have colliding ids across members.
    """
    if not manifest:
        raise ValueError("run_object_id_series: manifest must be non-empty")

    os.makedirs(output_dir, exist_ok=True)

    by_member: dict[str | None, list[SeriesEntry]] = defaultdict(list)
    for entry in manifest:
        by_member[entry.member_id].append(entry)
    for entries in by_member.values():
        entries.sort(key=lambda e: e.valid_time)

    grid_geometry: GridGeometry | None = None
    all_results: list[IdentificationResult] = []
    next_track_id = 1

    for member_id, entries in by_member.items():
        prev_objects: list[StormObject] | None = None
        prev_labels: np.ndarray | None = None
        prev_time: datetime | None = None

        for entry in entries:
            field = loader_fn(entry)

            if grid_geometry is None:
                grid_geometry = precompute_grid_geometry(field.lat2d, field.lon2d)

            data2d = field.data
            if mask is not None:
                data2d = np.where(mask, 0.0, data2d)

            labels, objects = identify_objects(
                data2d, grid_geometry, thresh_1, thresh_2, area_thresh_km2,
                linear_eccentricity_thresh=linear_eccentricity_thresh,
                linear_length_thresh_km=linear_length_thresh_km,
                mixed_eccentricity_thresh=mixed_eccentricity_thresh,
                mixed_length_thresh_km=mixed_length_thresh_km,
            )

            if track_in_time:
                objects, next_track_id = track_objects_incremental(
                    prev_objects, prev_labels, prev_time,
                    objects, labels, entry.valid_time,
                    grid_geometry, next_track_id, track_bound_disp_km=track_bound_disp_km,
                )
                prev_objects, prev_labels, prev_time = objects, labels, entry.valid_time

            all_results.append(
                IdentificationResult(
                    labels=labels, objects=objects, valid_time=entry.valid_time, member_id=member_id,
                )
            )

    output_paths = []

    if file_grouping == "single":
        for r in all_results:
            path = os.path.join(output_dir, _default_output_name("single", r.member_id, r.valid_time))
            write_object_file(
                path, init_time, grid_geometry.lat2d, grid_geometry.lon2d, [r],
                source_files=[e.filepath for e in manifest if e.member_id == r.member_id and e.valid_time == r.valid_time],
                thresh_1=thresh_1, thresh_2=thresh_2, area_thresh_km2=area_thresh_km2,
                tracked=track_in_time, track_bound_disp_km=track_bound_disp_km if track_in_time else None,
            )
            output_paths.append(path)

    elif file_grouping == "member_series":
        by_member_results: dict[str | None, list[IdentificationResult]] = defaultdict(list)
        for r in all_results:
            by_member_results[r.member_id].append(r)
        for member_id, results in by_member_results.items():
            path = os.path.join(output_dir, _default_output_name("member_series", member_id, None))
            write_object_file(
                path, init_time, grid_geometry.lat2d, grid_geometry.lon2d, results,
                source_files=[e.filepath for e in manifest if e.member_id == member_id],
                thresh_1=thresh_1, thresh_2=thresh_2, area_thresh_km2=area_thresh_km2,
                tracked=track_in_time, track_bound_disp_km=track_bound_disp_km if track_in_time else None,
            )
            output_paths.append(path)

    elif file_grouping == "ensemble_snapshot":
        by_time_results: dict[datetime, list[IdentificationResult]] = defaultdict(list)
        for r in all_results:
            by_time_results[r.valid_time].append(r)
        for valid_time, results in by_time_results.items():
            path = os.path.join(output_dir, _default_output_name("ensemble_snapshot", None, valid_time))
            write_object_file(
                path, init_time, grid_geometry.lat2d, grid_geometry.lon2d, results,
                source_files=[e.filepath for e in manifest if e.valid_time == valid_time],
                thresh_1=thresh_1, thresh_2=thresh_2, area_thresh_km2=area_thresh_km2,
                tracked=track_in_time, track_bound_disp_km=track_bound_disp_km if track_in_time else None,
            )
            output_paths.append(path)

    elif file_grouping == "full":
        path = os.path.join(output_dir, _default_output_name("full", None, None))
        write_object_file(
            path, init_time, grid_geometry.lat2d, grid_geometry.lon2d, all_results,
            source_files=[e.filepath for e in manifest],
            thresh_1=thresh_1, thresh_2=thresh_2, area_thresh_km2=area_thresh_km2,
            tracked=track_in_time, track_bound_disp_km=track_bound_disp_km if track_in_time else None,
        )
        output_paths.append(path)

    else:
        raise ValueError(f"Unknown file_grouping '{file_grouping}'")

    return output_paths
