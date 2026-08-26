"""Incremental, source-agnostic object-in-time tracking (storm age + track id).

An age-linkage algorithm (an exact 0-distance boundary-overlap check between
consecutive timesteps) that applies to *any* object series, not just
observations -- there is nothing here that assumes truth vs. forecast.

Deliberately incremental/streaming: only the immediately preceding timestep's
objects/labels are needed, not the whole history, so a caller's loop over a
long series stays memory-light. The caller threads `prev_objects`,
`prev_labels`, `prev_time`, and `next_track_id` forward call to call (see
id_pipeline.py).
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
    """Annotate curr_objects with age_seconds/track_id/branch_id given the
    previous timestep's (already-tracked) objects, and return the updated
    next_track_id counter for the caller to pass into the following call.

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

    track_id vs. branch_id (v2): track_id is unchanged from v1 -- shared by
    every descendant of one convective-initiation event, forever. branch_id
    is new: it starts equal to track_id when an object is first created,
    stays inherited unchanged across ordinary one-parent-to-one-child
    continuation, but forks to a brand-new value for EVERY child (no
    exceptions -- no child is treated as "the" continuation) the moment a
    split is detected (2+ current objects resolving to the SAME one previous
    object). track_id/next_track_id share one counter (a brand-new track's
    branch_id starts equal to its track_id, so there is no risk of the two id
    spaces colliding). Mergers (multiple previous objects -> one current
    object) keep the existing "oldest wins" rule below unchanged -- age_seconds/
    track_id/branch_id are all inherited from that single winning (oldest)
    parent.
    """
    if prev_objects is None or prev_labels is None or prev_time is None:
        tracked = []
        for obj in curr_objects:
            tracked.append(_with_tracking(obj, age_seconds=0.0, track_id=next_track_id, branch_id=next_track_id))
            next_track_id += 1
        return tracked, next_track_id

    dt_seconds = (curr_time - prev_time).total_seconds()
    prev_by_id = {p.id: p for p in prev_objects}

    # Phase A: resolve each curr object's single winning prior object (the
    # existing "oldest wins" rule, unchanged) -- now also recording WHICH
    # prev object id won (not just its track_id), since two different prev
    # objects can share one track_id after an earlier split, and split
    # detection in Phase B needs to group by the specific prev object, not
    # merely by track_id.
    resolved = []  # list of (curr_obj, best_age, best_track_id, best_branch_id, best_prev_id)
    for obj in curr_objects:
        if track_bound_disp_km == 0.0:
            overlapping_prev_ids = _exact_overlap_prev_ids(obj.id, curr_labels, prev_labels)
        else:
            overlapping_prev_ids = _buffered_overlap_prev_ids(
                obj.id, curr_labels, prev_labels, prev_objects, grid_geometry, track_bound_disp_km
            )

        best_age = None
        best_track_id = None
        best_branch_id = None
        best_prev_id = None
        for prev_id in overlapping_prev_ids:
            prev_obj = prev_by_id.get(prev_id)
            if prev_obj is None:
                continue
            candidate_age = (prev_obj.age_seconds or 0.0) + dt_seconds
            # matches legacy: among all overlapping prior objects (e.g. a
            # merger), the OLDEST wins -- its age, track_id, and branch_id
            if best_age is None or candidate_age > best_age:
                best_age = candidate_age
                best_track_id = prev_obj.track_id
                best_branch_id = prev_obj.branch_id
                best_prev_id = prev_obj.id

        resolved.append((obj, best_age, best_track_id, best_branch_id, best_prev_id))

    # Phase B: a split is exactly "2+ curr objects resolved to the same one
    # prev object id" -- count how many curr objects each prev id attracted.
    prev_id_counts: dict[int, int] = {}
    for _, _, _, _, best_prev_id in resolved:
        if best_prev_id is not None:
            prev_id_counts[best_prev_id] = prev_id_counts.get(best_prev_id, 0) + 1

    tracked = []
    for obj, best_age, best_track_id, best_branch_id, best_prev_id in resolved:
        if best_age is None:
            # no qualifying candidate at all -- a brand-new track (age 0),
            # same as CI: mint one id, used for both track_id and branch_id.
            tracked.append(_with_tracking(obj, age_seconds=0.0, track_id=next_track_id, branch_id=next_track_id))
            next_track_id += 1
        elif prev_id_counts[best_prev_id] > 1:
            # split: track_id inherited (lineage preserved), branch_id freshly minted
            tracked.append(_with_tracking(obj, age_seconds=best_age, track_id=best_track_id, branch_id=next_track_id))
            next_track_id += 1
        else:
            # ordinary continuation (including a merger's single winning
            # parent): both ids inherited unchanged
            tracked.append(_with_tracking(obj, age_seconds=best_age, track_id=best_track_id, branch_id=best_branch_id))

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


def _with_tracking(obj: StormObject, age_seconds: float, track_id: int, branch_id: int) -> StormObject:
    return replace(obj, age_seconds=age_seconds, track_id=track_id, branch_id=branch_id)
