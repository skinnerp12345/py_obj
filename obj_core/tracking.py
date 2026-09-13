"""Incremental, source-agnostic object-in-time tracking (storm age, track id,
split id).

An age-linkage algorithm (an exact 0-distance boundary-overlap check between
consecutive timesteps) that applies to *any* object series, not just
observations -- there is nothing here that assumes truth vs. forecast.

Deliberately incremental/streaming: only the immediately preceding timestep's
objects/labels are needed, not the whole history, so a caller's loop over a
long series stays memory-light. The caller threads `prev_objects`,
`prev_labels`, `prev_time`, `next_track_id`, and `next_split_n` forward call
to call (see id_pipeline.py).
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
    next_split_n: dict[int, int],
    track_bound_disp_km: float = 0.0,
) -> tuple[list[StormObject], int, dict[int, int]]:
    """Annotate curr_objects with age_seconds/track_id/split_id given the
    previous timestep's (already-tracked) objects, and return the updated
    next_track_id/next_split_n state for the caller to pass into the
    following call.

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

    track_id vs. split_id: track_id is shared by every descendant of one
    convective-initiation event, forever, and never changes once minted.
    split_id is a string "<track_id>:n" that starts at "<track_id>:0" for
    every object (even one that never splits) and identifies one continuous
    physical thread: it is inherited UNCHANGED across ordinary
    one-parent-to-one-child continuation, and across a merger (multiple
    previous objects -> one current object, "oldest wins" as before -- the
    merged object simply inherits the winning parent's ids, no new number
    minted). The moment a split is detected (2+ current objects resolving to
    the SAME one previous object), the child with the GREATEST PIXEL OVERLAP
    against that shared previous object keeps the inherited split_id
    unchanged (it is "the" continuation from here on); every other child
    gets a freshly minted "<track_id>:<n>", where n is drawn from a
    per-track_id counter (`next_split_n`) that only advances for
    non-winning children, never resets, and is shared across every branch
    already living under that track_id -- so if track 5 has already
    produced 2 new branches from earlier splits (its next available n is 3),
    the very next split anywhere in track 5's lineage mints "5:3", regardless
    of which of track 5's current branches does the splitting.
    """
    if prev_objects is None or prev_labels is None or prev_time is None:
        tracked = []
        for obj in curr_objects:
            new_track_id = next_track_id
            next_track_id += 1
            next_split_n[new_track_id] = 1
            tracked.append(_with_tracking(obj, age_seconds=0.0, track_id=new_track_id, split_id=f"{new_track_id}:0"))
        return tracked, next_track_id, next_split_n

    dt_seconds = (curr_time - prev_time).total_seconds()
    prev_by_id = {p.id: p for p in prev_objects}

    # Phase A: resolve each curr object's single winning prior object (the
    # existing "oldest wins" rule, unchanged -- this decides WHOSE lineage a
    # continuing/merged object inherits, a different question from split-
    # child ranking in Phase B below) -- now also recording the pixel
    # overlap count and boundary distance between the curr object and
    # specifically its winning prior object, since Phase B needs to rank
    # split candidates by overlap, not by age.
    resolved = []  # (curr_obj, best_age, best_track_id, best_split_id, best_prev_id, best_overlap, best_dist_km)
    for obj in curr_objects:
        if track_bound_disp_km == 0.0:
            candidates = _exact_overlap_prev_ids(obj.id, curr_labels, prev_labels)
        else:
            candidates = _buffered_overlap_prev_ids(
                obj.id, curr_labels, prev_labels, prev_objects, grid_geometry, track_bound_disp_km
            )

        best_age = None
        best_track_id = None
        best_split_id = None
        best_prev_id = None
        best_overlap = None
        best_dist_km = None
        for prev_id, overlap_count, dist_km in candidates:
            prev_obj = prev_by_id.get(prev_id)
            if prev_obj is None:
                continue
            candidate_age = (prev_obj.age_seconds or 0.0) + dt_seconds
            # matches legacy: among all overlapping prior objects (e.g. a
            # merger), the OLDEST wins -- its age, track_id, and split_id
            if best_age is None or candidate_age > best_age:
                best_age = candidate_age
                best_track_id = prev_obj.track_id
                best_split_id = prev_obj.split_id
                best_prev_id = prev_obj.id
                best_overlap = overlap_count
                best_dist_km = dist_km

        resolved.append((obj, best_age, best_track_id, best_split_id, best_prev_id, best_overlap, best_dist_km))

    # Phase B: a split is exactly "2+ curr objects resolved to the same one
    # prev object id" -- group by that shared prev id, then within any group
    # of size 2+, the member with the greatest overlap against the shared
    # prev object is the winner (nearest-boundary-distance tiebreak, for the
    # buffered path where overlap can legitimately be 0 for every candidate).
    groups: dict[int, list[int]] = {}
    for i, entry in enumerate(resolved):
        best_prev_id = entry[4]
        if best_prev_id is not None:
            groups.setdefault(best_prev_id, []).append(i)

    group_winner: dict[int, int] = {
        prev_id: max(indices, key=lambda i: (resolved[i][5], -resolved[i][6]))
        for prev_id, indices in groups.items()
        if len(indices) > 1
    }

    tracked = []
    for i, (obj, best_age, best_track_id, best_split_id, best_prev_id, _, _) in enumerate(resolved):
        if best_age is None:
            # no qualifying candidate at all -- a brand-new track (age 0),
            # same as CI: mint one id for track_id, split_id starts at ":0".
            new_track_id = next_track_id
            next_track_id += 1
            next_split_n[new_track_id] = 1
            tracked.append(_with_tracking(obj, age_seconds=0.0, track_id=new_track_id, split_id=f"{new_track_id}:0"))
        elif best_prev_id in group_winner and i != group_winner[best_prev_id]:
            # split, and NOT the greatest-overlap child -- mint the next
            # available number for this track.
            n = next_split_n[best_track_id]
            next_split_n[best_track_id] = n + 1
            tracked.append(_with_tracking(obj, age_seconds=best_age, track_id=best_track_id, split_id=f"{best_track_id}:{n}"))
        else:
            # ordinary continuation, a merger's single winning parent, or
            # the greatest-overlap child of a split -- split_id unchanged.
            tracked.append(_with_tracking(obj, age_seconds=best_age, track_id=best_track_id, split_id=best_split_id))

    return tracked, next_track_id, next_split_n


def _exact_overlap_prev_ids(curr_id: int, curr_labels: np.ndarray, prev_labels: np.ndarray) -> list[tuple[int, int, float]]:
    """Prior-timestep object ids whose pixels directly intersect curr_id's
    pixels, with the overlap pixel count -- O(grid size), no per-pixel
    pairwise distance computation. Boundary distance is always 0.0 here
    (these ids were found via direct intersection); kept as a 3rd tuple
    element only so callers can treat both overlap paths uniformly."""
    prev_vals = prev_labels[curr_labels == curr_id]
    ids, counts = np.unique(prev_vals, return_counts=True)
    return [(int(i), int(c), 0.0) for i, c in zip(ids, counts) if i != 0]


def _buffered_overlap_prev_ids(
    curr_id: int,
    curr_labels: np.ndarray,
    prev_labels: np.ndarray,
    prev_objects: list[StormObject],
    grid_geometry: GridGeometry,
    track_bound_disp_km: float,
) -> list[tuple[int, int, float]]:
    """General case: prior objects within track_bound_disp_km (not just
    touching), with each candidate's real pixel overlap count (which can
    legitimately be 0 -- this is a proximity search, not an intersection
    test) and boundary distance in km. Falls back to a per-pixel boundary
    distance (Step 2's boundary_dist_km), which is more expensive -- only
    used when a non-zero buffer is explicitly requested."""
    curr_mask = curr_labels == curr_id
    curr_coords_km = object_coords_km(curr_id, curr_labels, grid_geometry.x2d, grid_geometry.y2d)
    result = []
    for prev_obj in prev_objects:
        prev_coords_km = object_coords_km(prev_obj.id, prev_labels, grid_geometry.x2d, grid_geometry.y2d)
        dist_km = boundary_dist_km(curr_coords_km, prev_coords_km)
        if dist_km <= track_bound_disp_km:
            overlap_count = int(np.count_nonzero(curr_mask & (prev_labels == prev_obj.id)))
            result.append((prev_obj.id, overlap_count, dist_km))
    return result


def _with_tracking(obj: StormObject, age_seconds: float, track_id: int, split_id: str) -> StormObject:
    return replace(obj, age_seconds=age_seconds, track_id=track_id, split_id=split_id)
