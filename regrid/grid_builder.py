"""Construct a target GridSpec from a southwest-corner point + regular km
spacing + dimensions, instead of reading one out of an existing model file.

Everything downstream of a GridSpec (cell-corner estimation, conservative
regridding, weight caching) is already completely agnostic to how the grid's
lat2d/lon2d arrays were produced -- grid_hash() (used for the weight-cache
key) hashes only array content, never a file path. So a computed grid plugs
into the existing pipeline with zero changes needed there; this module's job
is only to produce the lat2d/lon2d arrays.
"""

import numpy as np
import pyproj

from .grid_spec import GridSpec

# Spherical earth radius -- duplicated from obj_core.geometry's own constant
# of the same name/value rather than imported, to avoid a real circular
# import: obj_core/__init__.py -> manifest.py -> `from python_obj.regrid
# import ...` -> (this module, if it imported obj_core back) -> obj_core
# again. regrid/ and obj_core/ are siblings with a one-directional
# dependency (obj_core depends on regrid, never the reverse); importing
# obj_core from here would introduce a new, backwards coupling just to
# reuse one float constant.
_EARTH_RADIUS_M = 6370000.0

_KM_PER_DEG_LAT = 111.0  # rough spherical approximation, only used to derive
                         # LCC projection parameters (standard parallels/center)
                         # before the grid itself is built -- not used for the
                         # actual grid point positions, which come from the
                         # exact LCC forward/inverse projection below.


def build_corner_spacing_grid(
    sw_lat: float,
    sw_lon: float,
    dx_km: float,
    nx: int,
    ny: int,
    dy_km: float | None = None,
    true_lat_1: float | None = None,
    true_lat_2: float | None = None,
    cen_lat: float | None = None,
    cen_lon: float | None = None,
) -> GridSpec:
    """Build an (ny, nx) GridSpec of regularly-spaced points on an LCC plane,
    starting at (sw_lat, sw_lon) as the (row=0, col=0) grid point (a cell
    CENTER, matching how every other grid in this library is represented --
    corners are a separate, optional field, estimated automatically via
    estimate_cell_corners() when absent, exactly as for a file-loaded grid).

    dy_km defaults to dx_km (square cells) if not given.

    true_lat_1/true_lat_2/cen_lat/cen_lon: same escape hatch as
    obj_core.geometry.build_projected_coords -- if not given, they're derived
    automatically from a rough spherical estimate of this grid's own extent
    (since, unlike build_projected_coords, there's no existing lat/lon array
    to derive them from directly -- the grid doesn't exist yet). Pass them
    explicitly to match a specific model's real projection, or to guarantee
    two separately-built grids share one projection (see
    build_projected_coords's own docstring for why that matters when
    comparing distances across two point sets).
    """
    if dy_km is None:
        dy_km = dx_km
    if nx < 1 or ny < 1:
        raise ValueError(f"build_corner_spacing_grid: nx/ny must be >= 1, got nx={nx}, ny={ny}")

    if cen_lat is None or cen_lon is None or true_lat_1 is None or true_lat_2 is None:
        # rough spherical estimate of the domain's NE corner, used only to
        # pick sensible LCC parameters -- the actual grid point positions
        # come from the exact projection built below, not this estimate.
        lat_span_deg = (ny * dy_km) / _KM_PER_DEG_LAT
        mid_lat_est = sw_lat + lat_span_deg / 2.0
        lon_span_deg = (nx * dx_km) / (_KM_PER_DEG_LAT * max(np.cos(np.radians(mid_lat_est)), 1e-6))
        ne_lat_est = sw_lat + lat_span_deg
        ne_lon_est = sw_lon + lon_span_deg

        if cen_lat is None:
            cen_lat = 0.5 * (sw_lat + ne_lat_est)
        if cen_lon is None:
            cen_lon = 0.5 * (sw_lon + ne_lon_est)
        if true_lat_1 is None or true_lat_2 is None:
            true_lat_1 = cen_lat - lat_span_deg / 4.0
            true_lat_2 = cen_lat + lat_span_deg / 4.0

    proj = pyproj.Proj(
        proj="lcc", lat_1=true_lat_1, lat_2=true_lat_2, lat_0=cen_lat, lon_0=cen_lon,
        a=_EARTH_RADIUS_M, b=_EARTH_RADIUS_M,
    )

    x0_m, y0_m = proj(sw_lon, sw_lat)
    x_m = x0_m + np.arange(nx) * dx_km * 1000.0
    y_m = y0_m + np.arange(ny) * dy_km * 1000.0
    x2d_m, y2d_m = np.meshgrid(x_m, y_m)  # shape (ny, nx)

    lon2d, lat2d = proj(x2d_m, y2d_m, inverse=True)
    return GridSpec(lat2d=np.asarray(lat2d), lon2d=np.asarray(lon2d))
