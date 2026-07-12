"""Incremental, source-agnostic object-in-time tracking (storm age + track id).

Generalizes python_base/object_matcher_base.py's MRMS-only age-linkage
algorithm (an exact 0-distance boundary-overlap check between consecutive
timesteps) to *any* object series, not just obs -- there is nothing here that
assumes truth vs. forecast.

Deliberately incremental/streaming: only the immediately preceding timestep's
objects/labels are needed, not the whole history, so a caller's loop over a
long series stays memory-light. The caller threads `prev_objects`,
`prev_labels`, `prev_time`, and `next_track_id` forward call to call (see
id_pipeline.py), exactly mirroring the legacy per-file loop structure.
"""

from dataclasses import replace
from datetime import datetime

import numpy as np

from .geometry import boundary_dist_km, object_coords_km
from .identify import GridGeometry, StormObject


def track_objects_incremental(
    prev_objects: list[StormObject] | None,
    prev_labels: np.ndarray | None,
    prev_time: datetime | None,
    curr_objects: list[StormObject],
    curr_labels: np.ndarray,
    curr_time: datetime,
    grid_geometry: GridGeometry,
    next_track_id: int,
    track_bound_disp_km: float = 0.0,
) -> tuple[list[StormObject], int]:
    """Annotate curr_objects with age_seconds/track_id given the previous
    timestep's (already-tracked) objects, and return the updated next_track_id
    counter for the caller to pass into the following call.

    track_bound_disp_km=0.0 (default) matches the legacy behavior exactly: only
    objects that actually touch/overlap are linked, not objects merely within
    some buffered search radius. This case is handled by a fast direct label
    intersection (O(grid size) per object, no pairwise distance computation)
    rather than the general buffered case below, which is a real performance
    requirement, not just an optimization: an early real-data test (MPAS, tens
    of objects per timestep, some spanning hundreds to thousands of pixels for
    an MCS) timed out after 2+ minutes using a naive all-pairs cdist-based
    boundary distance for every (curr, prev) object pair -- cdist scales with
    the NUMBER OF PIXELS in each object, not the number of objects, so it blows
    up badly for large storm objects. Direct label-array intersection avoids
    this entirely for the (default, legacy-matching) exact-overlap case.

    If prev_objects is None (first timestep of a series), every curr object
    starts a brand-new track at age 0.
    """
    if prev_objects is None or prev_labels is None or prev_time is None:
        tracked = []
        for obj in curr_objects:
            tracked.append(_with_tracking(obj, age_seconds=0.0, track_id=next_track_id))
            next_track_id += 1
        return tracked, next_track_id

    dt_seconds = (curr_time - prev_time).total_seconds()
    prev_by_id = {p.id: p for p in prev_objects}

    tracked = []
    for obj in curr_objects:
        if track_bound_disp_km == 0.0:
            overlapping_prev_ids = _exact_overlap_prev_ids(obj.id, curr_labels, prev_labels)
        else:
            overlapping_prev_ids = _buffered_overlap_prev_ids(
                obj.id, curr_labels, prev_labels, prev_objects, grid_geometry, track_bound_disp_km
            )

        best_age = None
        best_track_id = None
        for prev_id in overlapping_prev_ids:
            prev_obj = prev_by_id.get(prev_id)
            if prev_obj is None:
                continue
            candidate_age = (prev_obj.age_seconds or 0.0) + dt_seconds
            # matches legacy: among all overlapping prior objects (e.g. a
            # merger), the OLDEST wins -- both its age and its track_id
            if best_age is None or candidate_age > best_age:
                best_age = candidate_age
                best_track_id = prev_obj.track_id

        if best_age is None:
            tracked.append(_with_tracking(obj, age_seconds=0.0, track_id=next_track_id))
            next_track_id += 1
        else:
            tracked.append(_with_tracking(obj, age_seconds=best_age, track_id=best_track_id))

    return tracked, next_track_id


def _exact_overlap_prev_ids(curr_id: int, curr_labels: np.ndarray, prev_labels: np.ndarray) -> list[int]:
    """Prior-timestep object ids whose pixels directly intersect curr_id's
    pixels -- O(grid size), no per-pixel pairwise distance computation."""
    overlapping = np.unique(prev_labels[curr_labels == curr_id])
    return [int(i) for i in overlapping if i != 0]


def _buffered_overlap_prev_ids(
    curr_id: int,
    curr_labels: np.ndarray,
    prev_labels: np.ndarray,
    prev_objects: list[StormObject],
    grid_geometry: GridGeometry,
    track_bound_disp_km: float,
) -> list[int]:
    """General case: prior objects within track_bound_disp_km (not just
    touching). Falls back to a per-pixel boundary distance (Step 2's
    boundary_dist_km), which is more expensive -- only used when a non-zero
    buffer is explicitly requested."""
    curr_coords_km = object_coords_km(curr_id, curr_labels, grid_geometry.x2d, grid_geometry.y2d)
    result = []
    for prev_obj in prev_objects:
        prev_coords_km = object_coords_km(prev_obj.id, prev_labels, grid_geometry.x2d, grid_geometry.y2d)
        if boundary_dist_km(curr_coords_km, prev_coords_km) <= track_bound_disp_km:
            result.append(prev_obj.id)
    return result


def _with_tracking(obj: StormObject, age_seconds: float, track_id: int) -> StormObject:
    return replace(obj, age_seconds=age_seconds, track_id=track_id)
