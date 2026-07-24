"""Object matching: classify truth vs. forecast objects into hit/miss/
false_alarm/truth_extra/forecast_extra via a Total Interest (TI) score.

Ports python_base/object_matcher_base.py's calc_ti/calc_ti_area_ratio (lines
274-325) unchanged in formula, fed km-space geometry (Step 2's
centroid_dist_km/boundary_dist_km) instead of pixel-index distances
pre-divided by a scalar dx. Operates on the unified StormObject for both
sides -- the legacy functions awkwardly expect a dict for MRMS and a
regionprops object for WoFS; StormObject already eliminates that asymmetry.

The MATCHING ALGORITHM itself (this module's match_objects_one_timestep) is
NOT a port of the legacy's iterative reassignment loop (lines 330-438), which
resolves contested duplicate claims via up to 5 rounds of "each unmatched
object picks its best still-available partner, repeat." Per the user's
request, this is replaced with a single global greedy assignment over all
candidate pairs: compute the full pairwise spatial TI matrix once, define the
candidate population as pairs exceeding ti_threshold, weight by area ratio,
sort all candidates by weighted TI descending, and walk the sorted list once
confirming a match whenever both objects are still free. This resolves every
contested duplicate directly (the highest-scoring pair for any contested
object is always considered first in the single sorted pass) with no
iteration cap needed, and is a standard approach for this kind of bipartite
object-matching problem.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import boundary_dist_km, centroid_dist_km, object_coords_km
from .identify import GridGeometry, StormObject


def total_interest(
    cent1_km: tuple[float, float],
    coords1_km: np.ndarray,
    cent2_km: tuple[float, float],
    coords2_km: np.ndarray,
    max_centroid_km: float,
    max_boundary_km: float,
) -> float:
    """Spatial Total Interest -- centroid + boundary displacement only, no area
    ratio. Direct km-space replacement for obj_cbook.py's calc_ti."""
    cent_dist = centroid_dist_km(cent1_km, cent2_km)
    bound_dist = boundary_dist_km(coords1_km, coords2_km)
    cent_ti = max(0.0, (max_centroid_km - cent_dist) / max_centroid_km)
    bound_ti = max(0.0, (max_boundary_km - bound_dist) / max_boundary_km)
    return 0.5 * (cent_ti + bound_ti)


def total_interest_area_ratio(ti: float, area1_km2: float, area2_km2: float) -> float:
    """Area-ratio-weighted Total Interest -- direct km-space replacement for
    obj_cbook.py's calc_ti_area_ratio. area_ratio = smaller/larger area, so it
    can never exceed 1."""
    area_ratio = min(area1_km2, area2_km2) / max(area1_km2, area2_km2)
    return ti * area_ratio


_MATCH_RECORD_SIDE_FIELDS = [
    "area_km2", "max_intensity", "mean_intensity", "is_linear", "centroid_lat", "centroid_lon",
    "solidity", "major_axis_length", "minor_axis_length", "eccentricity",
]


@dataclass
class MatchRecord:
    category: str  # "hit" | "miss" | "false_alarm" | "truth_extra" | "forecast_extra"
    truth_id: int  # -1 if no truth object applies to this row
    forecast_id: int  # -1 if no forecast object applies to this row
    ti_score: float  # area-weighted TI; 0.0 for miss/false_alarm (no candidate existed at all)

    truth_area_km2: float | None = None
    truth_max_intensity: float | None = None
    truth_mean_intensity: float | None = None
    truth_is_linear: int | None = None  # 0=cellular, 1=mixed, 2=linear; None if no truth object
    truth_centroid_lat: float | None = None
    truth_centroid_lon: float | None = None
    truth_solidity: float | None = None
    truth_major_axis_length: float | None = None
    truth_minor_axis_length: float | None = None
    truth_eccentricity: float | None = None

    forecast_area_km2: float | None = None
    forecast_max_intensity: float | None = None
    forecast_mean_intensity: float | None = None
    forecast_is_linear: int | None = None  # 0=cellular, 1=mixed, 2=linear; None if no forecast object
    forecast_centroid_lat: float | None = None
    forecast_centroid_lon: float | None = None
    forecast_solidity: float | None = None
    forecast_major_axis_length: float | None = None
    forecast_minor_axis_length: float | None = None
    forecast_eccentricity: float | None = None


def _record_from_objects(
    category: str, truth_obj: StormObject | None, forecast_obj: StormObject | None, ti_score: float
) -> MatchRecord:
    kwargs = {"category": category, "ti_score": ti_score, "truth_id": -1, "forecast_id": -1}
    if truth_obj is not None:
        kwargs["truth_id"] = truth_obj.id
        for f in _MATCH_RECORD_SIDE_FIELDS:
            kwargs[f"truth_{f}"] = getattr(truth_obj, f)
    if forecast_obj is not None:
        kwargs["forecast_id"] = forecast_obj.id
        for f in _MATCH_RECORD_SIDE_FIELDS:
            kwargs[f"forecast_{f}"] = getattr(forecast_obj, f)
    return MatchRecord(**kwargs)


def match_objects_one_timestep(
    truth_objects: list[StormObject],
    truth_labels: np.ndarray,
    forecast_objects: list[StormObject],
    forecast_labels: np.ndarray,
    grid_geometry: GridGeometry,
    max_boundary_disp_km: float,
    max_centroid_disp_km: float,
    ti_threshold: float,
) -> list[MatchRecord]:
    """Match truth vs. forecast objects at one aligned valid_time via a global
    greedy assignment over all candidate pairs (see module docstring for why
    this replaces the legacy iterative-reassignment algorithm).

    Returns one MatchRecord per object (both matched and unmatched, on either
    side) -- never drops an object silently.
    """
    if not truth_objects and not forecast_objects:
        return []

    # precompute each object's own pixel coords in km-space ONCE, not per pair
    # -- boundary_dist_km's cost scales with pixel count, so this avoids
    # recomputing np.where(labels==id) redundantly across O(n_truth*n_forecast) pairs
    truth_coords = {o.id: object_coords_km(o.id, truth_labels, grid_geometry.x2d, grid_geometry.y2d) for o in truth_objects}
    forecast_coords = {o.id: object_coords_km(o.id, forecast_labels, grid_geometry.x2d, grid_geometry.y2d) for o in forecast_objects}

    # full pairwise spatial TI -> candidate population (pairs with spatial TI > threshold),
    # weighted by area ratio; also track each object's own best candidate (for extras)
    candidates: list[tuple[float, StormObject, StormObject]] = []
    truth_best: dict[int, tuple[float, StormObject]] = {}
    forecast_best: dict[int, tuple[float, StormObject]] = {}

    for t in truth_objects:
        t_cent = (t.centroid_x_km, t.centroid_y_km)
        for f in forecast_objects:
            f_cent = (f.centroid_x_km, f.centroid_y_km)
            spatial_ti = total_interest(
                t_cent, truth_coords[t.id], f_cent, forecast_coords[f.id],
                max_centroid_disp_km, max_boundary_disp_km,
            )
            if spatial_ti <= ti_threshold:
                continue
            weighted_ti = total_interest_area_ratio(spatial_ti, t.area_km2, f.area_km2)
            candidates.append((weighted_ti, t, f))
            if t.id not in truth_best or weighted_ti > truth_best[t.id][0]:
                truth_best[t.id] = (weighted_ti, f)
            if f.id not in forecast_best or weighted_ti > forecast_best[f.id][0]:
                forecast_best[f.id] = (weighted_ti, t)

    # global greedy assignment: sort all candidates by weighted TI descending,
    # confirm a match whenever both objects are still free -- one pass, no
    # iteration cap, resolves every contested duplicate directly
    candidates.sort(key=lambda c: c[0], reverse=True)
    matched_truth_ids: set[int] = set()
    matched_forecast_ids: set[int] = set()
    records: list[MatchRecord] = []

    for weighted_ti, t, f in candidates:
        if t.id in matched_truth_ids or f.id in matched_forecast_ids:
            continue
        matched_truth_ids.add(t.id)
        matched_forecast_ids.add(f.id)
        records.append(_record_from_objects("hit", t, f, weighted_ti))

    # classify every remaining (unmatched) object -- "extra" if it had a viable
    # candidate that lost the greedy competition, "miss"/"false_alarm" if it
    # never had a qualifying candidate at all
    for t in truth_objects:
        if t.id in matched_truth_ids:
            continue
        if t.id in truth_best:
            ti, partner = truth_best[t.id]
            records.append(_record_from_objects("truth_extra", t, partner, ti))
        else:
            records.append(_record_from_objects("miss", t, None, 0.0))

    for f in forecast_objects:
        if f.id in matched_forecast_ids:
            continue
        if f.id in forecast_best:
            ti, partner = forecast_best[f.id]
            records.append(_record_from_objects("forecast_extra", partner, f, ti))
        else:
            records.append(_record_from_objects("false_alarm", None, f, 0.0))

    return records
