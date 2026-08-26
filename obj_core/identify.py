"""Single-timestep thunderstorm object identification.

Object identification is a two-stage threshold + area filter:
find_initial_objects (skimage.measure.label + regionprops, already
grid-agnostic) and apply_maxint_thresh (a dBZ threshold, unit-independent),
plus a physical-area filter that filters on true physical area (via Step 2's
pixel_area_km2) instead of a raw pixel count pre-divided by a scalar dx**2 --
the same class of anisotropy fix described in geometry.py's own docstring.
Standalone: no knowledge of tracking, matching, masking, or where the field
came from.
"""

from dataclasses import dataclass, replace

import numpy as np
import skimage
from skimage.measure import regionprops

from .geometry import build_projected_coords, pixel_area_km2 as _pixel_area_km2, principal_axis_km


@dataclass
class GridGeometry:
    lat2d: np.ndarray
    lon2d: np.ndarray
    x2d: np.ndarray  # km, projected (Step 2's build_projected_coords)
    y2d: np.ndarray  # km
    pixel_area_km2: np.ndarray  # km^2 per grid cell


def precompute_grid_geometry(lat2d: np.ndarray, lon2d: np.ndarray) -> GridGeometry:
    """Compute once per fixed grid, reuse across every timestep of a series on
    that grid -- same pattern as the regrid weight cache (Step 1) and the CONUS
    boundary/KDTree (Step 3)."""
    x2d, y2d = build_projected_coords(lat2d, lon2d)
    area = _pixel_area_km2(x2d, y2d)
    return GridGeometry(lat2d=lat2d, lon2d=lon2d, x2d=x2d, y2d=y2d, pixel_area_km2=area)


@dataclass
class StormObject:
    id: int
    area_px: int
    area_km2: float
    max_intensity: float
    mean_intensity: float
    major_axis_length: float
    minor_axis_length: float
    eccentricity: float
    orientation: float
    solidity: float
    centroid_rowcol: tuple[float, float]
    centroid_lat: float
    centroid_lon: float
    centroid_x_km: float
    centroid_y_km: float
    is_linear: int  # 0=cellular, 1=mixed, 2=linear (see identify_objects())
    # Only populated when tracking is requested (see tracking.py); None otherwise.
    age_seconds: float | None = None
    track_id: int | None = None
    branch_id: int | None = None  # see tracking.py -- forks to a new id on a split, unchanged on simple continuation
    # Only populated when storm_mode_classification is requested (see
    # identify_objects()); None otherwise. Groups objects merged into one
    # "system" for storm-mode classification -- every object sharing a
    # system_id was reclassified (is_linear) from their MERGED footprint,
    # not individually.
    system_id: int | None = None


def find_initial_objects(var: np.ndarray, thresh: float):
    """Label every connected region at/above thresh (already grid-agnostic:
    skimage.measure.label + regionprops operate on plain pixel arrays)."""
    obj_init = np.where(var >= thresh, var, 0.0)
    obj_int = (var >= thresh).astype(int)
    obj_labels = skimage.measure.label(obj_int).astype(int)
    obj_props = regionprops(obj_labels, obj_init)
    return obj_labels, obj_props


def apply_maxint_thresh(props: list, thresh: float) -> list:
    """Keep only objects whose peak value exceeds thresh (a dBZ threshold,
    unit-independent)."""
    return [p for p in props if p.max_intensity > thresh]


def _classify_shape(
    major_axis_length_km: float,
    eccentricity_km: float,
    linear_eccentricity_thresh: float,
    linear_length_thresh_km: float,
    mixed_eccentricity_thresh: float,
    mixed_length_thresh_km: float,
) -> int:
    """Shared linear/mixed/cellular decision (0/1/2), used both per-object
    (the v1 default path) and per-merged-system (storm_mode_classification,
    v2) -- one copy of the strict-first two-tier threshold logic rather than
    two. See identify_objects()'s docstring for the tier definitions."""
    if eccentricity_km > linear_eccentricity_thresh and major_axis_length_km > linear_length_thresh_km:
        return 2
    elif eccentricity_km > mixed_eccentricity_thresh and major_axis_length_km > mixed_length_thresh_km:
        return 1
    return 0


def _merge_into_systems(
    objects: list[StormObject],
    object_pixels: dict[int, tuple[np.ndarray, np.ndarray]],
    data2d: np.ndarray,
    grid_geometry: GridGeometry,
    system_boundary_thresh: float,
    linear_eccentricity_thresh: float,
    linear_length_thresh_km: float,
    mixed_eccentricity_thresh: float,
    mixed_length_thresh_km: float,
) -> list[StormObject]:
    """storm_mode_classification's merge step -- groups `objects` by shared
    connectivity at system_boundary_thresh (a third, lower threshold than
    thresh_1/thresh_2), reclassifies each group's is_linear from the UNION of
    its constituent objects' own already-identified pixels (not the full
    system-boundary connected component -- see identify_objects()'s
    docstring), and assigns every member of a group the same system_id (the
    minimum constituent object id). A group of one object is a no-op for
    is_linear (the union of one object's own pixels is itself) but still
    gets a system_id.
    """
    system_labels, _ = find_initial_objects(data2d, system_boundary_thresh)

    # Which system-boundary connected region each object's pixels
    # predominantly fall under. Every retained object already exceeds
    # thresh_1 >= system_boundary_thresh by construction, so this is expected
    # to always find a positive label; the `obj.id` fallback (a singleton
    # system of just itself) only matters if that ordering is violated.
    object_system_label: dict[int, int] = {}
    for obj in objects:
        rows, cols = object_pixels[obj.id]
        labels_here = system_labels[rows, cols]
        nonzero = labels_here[labels_here != 0]
        object_system_label[obj.id] = int(np.bincount(nonzero).argmax()) if nonzero.size else obj.id

    groups: dict[int, list[StormObject]] = {}
    for obj in objects:
        groups.setdefault(object_system_label[obj.id], []).append(obj)

    updated_by_id: dict[int, StormObject] = {}
    for members in groups.values():
        system_id = min(o.id for o in members)
        if len(members) == 1:
            updated_by_id[members[0].id] = replace(members[0], system_id=system_id)
            continue

        rows = np.concatenate([object_pixels[o.id][0] for o in members])
        cols = np.concatenate([object_pixels[o.id][1] for o in members])
        x_km = grid_geometry.x2d[rows, cols]
        y_km = grid_geometry.y2d[rows, cols]
        major_axis_length_km, eccentricity_km = principal_axis_km(x_km, y_km)
        system_is_linear = _classify_shape(
            major_axis_length_km, eccentricity_km,
            linear_eccentricity_thresh, linear_length_thresh_km,
            mixed_eccentricity_thresh, mixed_length_thresh_km,
        )
        for obj in members:
            updated_by_id[obj.id] = replace(obj, is_linear=system_is_linear, system_id=system_id)

    return [updated_by_id[o.id] for o in objects]  # preserve original discovery order


def identify_objects(
    data2d: np.ndarray,
    grid_geometry: GridGeometry,
    thresh_1: float,
    thresh_2: float,
    area_thresh_km2: float,
    linear_eccentricity_thresh: float = 0.8,
    linear_length_thresh_km: float = 200.0,
    mixed_eccentricity_thresh: float = 0.75,
    mixed_length_thresh_km: float = 100.0,
    storm_mode_classification: bool = False,
    system_boundary_thresh: float | None = None,
) -> tuple[np.ndarray, list[StormObject]]:
    """Identify objects in one gridded field.

    Returns (labels, objects): `labels` is a 2D int array (0=background, else
    object id) containing ONLY the retained objects (dropped objects' pixels are
    zeroed out, retained objects keep their original skimage label id, matching
    each StormObject.id) -- this labels array, together with the grid, is
    sufficient to recover every retained object's full pixel membership later
    (np.where(labels == id)) without storing per-object coordinate lists.

    Every retained object is also classified into one of three shape
    categories (StormObject.is_linear) via major axis length + eccentricity,
    computed correctly in physical km-space (geometry.principal_axis_km)
    rather than the pixel-index-based major_axis_length/eccentricity
    regionprops fields, which would suffer the same latitude-dependent
    anisotropy distortion already fixed elsewhere in this library for
    area/distance. Two independent threshold tiers, checked strict-first
    (see _classify_shape()):
      - is_linear=2 ("linear"): eccentricity > linear_eccentricity_thresh AND
        major_axis_length_km > linear_length_thresh_km (defaults 0.8/200km).
      - is_linear=1 ("mixed"): only checked if the linear tier fails;
        eccentricity > mixed_eccentricity_thresh AND major_axis_length_km >
        mixed_length_thresh_km (defaults 0.75/100km).
      - is_linear=0 ("cellular"): meets neither tier.
    All four thresholds are tunable like every other threshold in this
    library, not hardcoded.

    storm_mode_classification (default False, v1-identical when off -- no new
    code below runs at all): objects sharing one connected region at a THIRD,
    lower threshold (system_boundary_thresh, required whenever this is True)
    are merged into a "system" for classification purposes -- fixes spurious
    loss of temporal continuity when a linear system's leading edge fragments
    into several individually-too-small/round objects at thresh_1/thresh_2.
    The merged shape used for classification is the UNION of the constituent
    objects' own already-identified pixel footprints (not the full
    system-boundary-threshold connected component, which would include weak
    connective-tissue pixels no individual object ever claimed) -- the
    system-boundary threshold is used only to decide which objects belong
    together, not to redefine any object's own pixels. The resulting
    system-level classification is written to `is_linear` on every
    constituent object (superseding each one's own individually-computed
    value), and every constituent object gets the same new `system_id`
    (the minimum constituent object id -- deterministic, no extra counter
    needed). A system of exactly one object trivially reproduces that
    object's own individual classification.
    """
    labels, props = find_initial_objects(data2d, thresh_1)
    props = apply_maxint_thresh(props, thresh_2)

    ny, nx = grid_geometry.lat2d.shape
    objects: list[StormObject] = []
    keep_ids: list[int] = []
    object_pixels: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for prop in props:
        rows = prop.coords[:, 0]
        cols = prop.coords[:, 1]
        area_km2 = float(grid_geometry.pixel_area_km2[rows, cols].sum())
        if area_km2 <= area_thresh_km2:
            continue

        keep_ids.append(prop.label)
        object_pixels[int(prop.label)] = (rows, cols)
        cr, cc = prop.centroid
        ri = min(max(int(round(cr)), 0), ny - 1)
        ci = min(max(int(round(cc)), 0), nx - 1)

        x_km = grid_geometry.x2d[rows, cols]
        y_km = grid_geometry.y2d[rows, cols]
        major_axis_length_km, eccentricity_km = principal_axis_km(x_km, y_km)
        is_linear = _classify_shape(
            major_axis_length_km, eccentricity_km,
            linear_eccentricity_thresh, linear_length_thresh_km,
            mixed_eccentricity_thresh, mixed_length_thresh_km,
        )

        objects.append(
            StormObject(
                id=int(prop.label),
                area_px=int(prop.area),
                area_km2=area_km2,
                max_intensity=float(prop.max_intensity),
                mean_intensity=float(prop.mean_intensity),
                major_axis_length=float(prop.major_axis_length),
                minor_axis_length=float(prop.minor_axis_length),
                eccentricity=float(prop.eccentricity),
                orientation=float(prop.orientation),
                solidity=float(prop.solidity),
                centroid_rowcol=(float(cr), float(cc)),
                centroid_lat=float(grid_geometry.lat2d[ri, ci]),
                centroid_lon=float(grid_geometry.lon2d[ri, ci]),
                centroid_x_km=float(grid_geometry.x2d[ri, ci]),
                centroid_y_km=float(grid_geometry.y2d[ri, ci]),
                is_linear=is_linear,
            )
        )

    if storm_mode_classification and objects:
        objects = _merge_into_systems(
            objects, object_pixels, data2d, grid_geometry, system_boundary_thresh,
            linear_eccentricity_thresh, linear_length_thresh_km,
            mixed_eccentricity_thresh, mixed_length_thresh_km,
        )

    if keep_ids:
        clean_labels = np.where(np.isin(labels, keep_ids), labels, 0)
    else:
        clean_labels = np.zeros_like(labels)

    return clean_labels, objects
