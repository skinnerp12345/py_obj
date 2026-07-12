"""Physical (km-space) geometry helpers.

Fixes a real latent bug in python_base/obj_cbook.py: `calc_centroid_dist`/
`calc_boundary_dist` operate directly on `regionprops` pixel-index coordinates,
relying on the caller pre-dividing the paper's 40 km / 108 km^2 thresholds by a
single scalar `dx` to convert to "pixel units". That is only correct on a
uniform, isotropic grid. MRMS's native 0.01-degree lat/lon grid is not:
km-per-gridpoint-in-longitude shrinks with cos(latitude), so it differs by
roughly 35% between 20N and 55N.

The fix: project lat/lon once per grid onto a physical x/y plane (km), and do
all distance/area math there instead of in pixel-index space. `calc_ti`/
`calc_ti_area_ratio` in obj_cbook.py are NOT changed here -- their control flow
already just consumes whatever centroid/coords/area values they're handed, so a
later step (object matching) reuses them unchanged by converting each object's
centroid/coords/area to km-space with these primitives before calling them.
"""

import numpy as np
import pyproj
from scipy.spatial import distance

# Spherical earth radius, matching the convention already used (but unused/dead)
# in python_base/obj_cbook.py's convert_lat_lon -- consistent with WRF's own
# spherical-datum assumption for the domains this method was developed against.
EARTH_RADIUS_M = 6370000.0


def build_projected_coords(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    true_lat_1: float | None = None,
    true_lat_2: float | None = None,
    cen_lat: float | None = None,
    cen_lon: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lat/lon onto a Lambert Conformal Conic plane, returning (x2d, y2d) in km.

    If true_lat_1/true_lat_2/cen_lat/cen_lon aren't given, they are derived
    automatically from THIS INPUT's own lat/lon range (standard parallels placed
    symmetrically at +/- lat_range/4 around the domain's central latitude). This is
    a documented approximation -- we only ever have a lat/lon array here, not the
    source model's actual projection metadata. Callers who know their model's real
    projection parameters (e.g. from a WRF namelist) should pass them explicitly.

    IMPORTANT: to compare distances between two different point sets (e.g. a model
    grid and a separately-loaded boundary polygon's vertices), pass the SAME
    true_lat_1/true_lat_2/cen_lat/cen_lon to both calls. Auto-deriving them
    independently per input -- as this function did before this parameter existed --
    silently produces two different projections whose x/y values are not
    comparable, making any cross-array distance calculation meaningless. This was
    a real bug caught while building CONUS masking (Step 3): projecting grid points
    and boundary vertices separately gave nonsensical distances (off by ~700 km).
    """
    lat_min, lat_max = float(np.min(lat2d)), float(np.max(lat2d))
    # lon2d may be in 0-360 (MRMS/grib convention) or -180-180; normalize to
    # -180-180 before computing a center longitude, to avoid wraparound errors.
    lon_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)

    if cen_lat is None:
        cen_lat = 0.5 * (lat_min + lat_max)
    if cen_lon is None:
        cen_lon = 0.5 * (float(np.min(lon_norm)) + float(np.max(lon_norm)))

    if true_lat_1 is None or true_lat_2 is None:
        lat_range = lat_max - lat_min
        true_lat_1 = cen_lat - lat_range / 4.0
        true_lat_2 = cen_lat + lat_range / 4.0

    proj = pyproj.Proj(
        proj="lcc", lat_1=true_lat_1, lat_2=true_lat_2, lat_0=cen_lat, lon_0=cen_lon,
        a=EARTH_RADIUS_M, b=EARTH_RADIUS_M,
    )
    x_m, y_m = proj(lon_norm, lat2d)
    return np.asarray(x_m) / 1000.0, np.asarray(y_m) / 1000.0


def pixel_area_km2(x2d: np.ndarray, y2d: np.ndarray) -> np.ndarray:
    """Physical area per grid cell (km^2) from projected coordinates.

    Uses the magnitude of the cross product of the two local grid-axis basis
    vectors (central-difference estimated), i.e. a parallelogram-area
    approximation per cell -- correct even if the projected grid is sheared or
    rotated relative to the row/column axes (as a lat/lon grid generally is once
    projected through an LCC transform), not just a naive dx*dy.
    """
    dx_dcol_x = np.gradient(x2d, axis=1)
    dx_dcol_y = np.gradient(y2d, axis=1)
    dx_drow_x = np.gradient(x2d, axis=0)
    dx_drow_y = np.gradient(y2d, axis=0)
    return np.abs(dx_dcol_x * dx_drow_y - dx_dcol_y * dx_drow_x)


def centroid_dist_km(cent1_xy: tuple[float, float], cent2_xy: tuple[float, float]) -> float:
    """Euclidean distance (km) between two (x, y) km-space centroids.

    Direct km-space replacement for obj_cbook.py's calc_centroid_dist, which
    operates on raw (row, col) pixel indices instead.
    """
    return float(np.hypot(cent2_xy[0] - cent1_xy[0], cent2_xy[1] - cent1_xy[1]))


def boundary_dist_km(coords1_xy: np.ndarray, coords2_xy: np.ndarray) -> float:
    """Minimum pairwise distance (km) between two objects' boundary point sets,
    each given as an (N, 2) array of (x, y) km-space coordinates.

    Direct km-space replacement for obj_cbook.py's calc_boundary_dist, which
    operates on raw (row, col) pixel indices instead.
    """
    return float(np.min(distance.cdist(coords1_xy, coords2_xy)))


def object_coords_km(obj_id: int, labels: np.ndarray, x2d: np.ndarray, y2d: np.ndarray) -> np.ndarray:
    """An object's full pixel set, projected to km-space -- (N, 2) array of
    (x, y) km coordinates. Shared by tracking.py and matching.py so both
    consume the identical object-to-coordinates mapping, not two copies of it.
    """
    rows, cols = np.where(labels == obj_id)
    return np.column_stack([x2d[rows, cols], y2d[rows, cols]])


def principal_axis_km(x_km: np.ndarray, y_km: np.ndarray) -> tuple[float, float]:
    """Major axis length (km) and eccentricity of an object, computed from its
    own projected (x, y) km-space pixel coordinates.

    Uses the same covariance/eigenvalue definition skimage.measure.regionprops
    itself uses for major_axis_length/eccentricity (central second moments of
    the pixel coordinate point set -> eigenvalues l1>=l2 of the covariance
    matrix -> major_axis_length=4*sqrt(l1), eccentricity=sqrt(1-l2/l1)) -- just
    fed physical coordinates instead of raw pixel row/col indices, so it
    inherits none of the anisotropy distortion a plain
    `regionprops.major_axis_length * grid_spacing` conversion would have (the
    same km-per-gridpoint-varies-with-latitude issue documented above for
    area/distance).
    """
    points = np.column_stack([x_km, y_km])
    cov = np.cov(points, rowvar=False, ddof=0)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    l2, l1 = float(eigvals[0]), float(eigvals[1])
    major_axis_length_km = 4.0 * np.sqrt(max(l1, 0.0))
    eccentricity_km = float(np.sqrt(max(1.0 - l2 / l1, 0.0))) if l1 > 0 else 0.0
    return major_axis_length_km, eccentricity_km
