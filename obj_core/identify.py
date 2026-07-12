"""Single-timestep thunderstorm object identification.

Thin wrap of python_base/obj_cbook.py's find_initial_objects (unchanged --
skimage.measure.label + regionprops, already grid-agnostic) and
apply_maxint_thresh (unchanged -- a dBZ threshold, unit-independent), plus a
*replacement* for apply_area_thresh that filters on true physical area (via
Step 2's pixel_area_km2) instead of a raw pixel count pre-divided by a scalar
dx**2. Standalone: no knowledge of tracking, matching, masking, or where the
field came from.
"""

from dataclasses import dataclass

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


def find_initial_objects(var: np.ndarray, thresh: float):
    """Verbatim port of python_base/obj_cbook.py's find_initial_objects (already
    grid-agnostic; not modified)."""
    obj_init = np.where(var >= thresh, var, 0.0)
    obj_int = (var >= thresh).astype(int)
    obj_labels = skimage.measure.label(obj_int).astype(int)
    obj_props = regionprops(obj_labels, obj_init)
    return obj_labels, obj_props


def apply_maxint_thresh(props: list, thresh: float) -> list:
    """Verbatim port of python_base/obj_cbook.py's apply_maxint_thresh (a dBZ
    threshold, unit-independent; not modified)."""
    return [p for p in props if p.max_intensity > thresh]


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
    area/distance. Two independent threshold tiers, checked strict-first:
      - is_linear=2 ("linear"): eccentricity > linear_eccentricity_thresh AND
        major_axis_length_km > linear_length_thresh_km (defaults 0.8/200km).
      - is_linear=1 ("mixed"): only checked if the linear tier fails;
        eccentricity > mixed_eccentricity_thresh AND major_axis_length_km >
        mixed_length_thresh_km (defaults 0.75/100km).
      - is_linear=0 ("cellular"): meets neither tier.
    All four thresholds are tunable like every other threshold in this
    library, not hardcoded.
    """
    labels, props = find_initial_objects(data2d, thresh_1)
    props = apply_maxint_thresh(props, thresh_2)

    ny, nx = grid_geometry.lat2d.shape
    objects: list[StormObject] = []
    keep_ids: list[int] = []

    for prop in props:
        rows = prop.coords[:, 0]
        cols = prop.coords[:, 1]
        area_km2 = float(grid_geometry.pixel_area_km2[rows, cols].sum())
        if area_km2 <= area_thresh_km2:
            continue

        keep_ids.append(prop.label)
        cr, cc = prop.centroid
        ri = min(max(int(round(cr)), 0), ny - 1)
        ci = min(max(int(round(cc)), 0), nx - 1)

        x_km = grid_geometry.x2d[rows, cols]
        y_km = grid_geometry.y2d[rows, cols]
        major_axis_length_km, eccentricity_km = principal_axis_km(x_km, y_km)
        if eccentricity_km > linear_eccentricity_thresh and major_axis_length_km > linear_length_thresh_km:
            is_linear = 2
        elif eccentricity_km > mixed_eccentricity_thresh and major_axis_length_km > mixed_length_thresh_km:
            is_linear = 1
        else:
            is_linear = 0

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

    if keep_ids:
        clean_labels = np.where(np.isin(labels, keep_ids), labels, 0)
    else:
        clean_labels = np.zeros_like(labels)

    return clean_labels, objects
