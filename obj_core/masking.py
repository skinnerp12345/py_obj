"""CONUS domain masking: exclude grid cells too far outside the US border to have
reliable MRMS radar coverage.

Rule (confirmed with the user, 2026-07-07): grid cells >100 km OUTSIDE the US
border are masked (excluded). Interior cells are never masked regardless of
distance from the border; cells outside but within the 100 km buffer are kept,
since near-border radar coverage can still reach a short distance beyond the
border. Two presets: the buffer rule alone over the full CONUS, and the same
rule plus everything west of a longitude cutoff (eastern ~2/3 CONUS, where
radar coverage is generally more reliable in less mountainous terrain).
"""

from functools import lru_cache

import numpy as np
import shapely
from matplotlib.path import Path
from scipy.spatial import cKDTree
from shapely.ops import unary_union

from .geometry import build_projected_coords

# States excluded from the "CONUS" (contiguous US) boundary.
_NON_CONUS_POSTAL_CODES = ("AK", "HI")

# ~5 km vertex spacing (in degrees) used to densify the boundary polygon before
# building the KDTree, so a nearest-*vertex* distance closely approximates the
# true nearest-*edge* distance (bounds the approximation error to roughly half
# this spacing, well under the 100 km decision threshold).
_DENSIFY_DEGREES = 0.05


@lru_cache(maxsize=4)
def load_conus_boundary(resolution: str = "50m"):
    """Load and cache the CONUS (48 states + DC) boundary as a single, densified
    shapely MultiPolygon, built from Natural Earth's admin_1_states_provinces_lakes
    shapefile (already used/cached by python_test_plot's cartopy-based tools).

    These per-state polygons are already clipped to the true coastline, so this
    one boundary simultaneously represents the land border and the coastline --
    no separate coastline dataset is needed.
    """
    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(
        resolution=resolution, category="cultural", name="admin_1_states_provinces_lakes"
    )
    reader = shpreader.Reader(path)
    conus_geoms = [
        rec.geometry
        for rec in reader.records()
        if rec.attributes.get("admin") == "United States of America"
        and rec.attributes.get("postal") not in _NON_CONUS_POSTAL_CODES
    ]
    if not conus_geoms:
        raise ValueError(
            f"No CONUS state polygons found in Natural Earth admin_1_states_provinces_lakes "
            f"({resolution}) -- check the shapefile's 'admin'/'postal' attributes."
        )

    conus_union = unary_union(conus_geoms)
    return shapely.segmentize(conus_union, max_segment_length=_DENSIFY_DEGREES)


def _boundary_vertices_lonlat(boundary) -> np.ndarray:
    """Flat (N, 2) array of (lon, lat) exterior-ring vertices from a Polygon or
    MultiPolygon (interior/hole rings are irrelevant for nearest-border distance
    -- see masking design notes on why lake holes don't need special handling)."""
    polys = list(boundary.geoms) if boundary.geom_type == "MultiPolygon" else [boundary]
    pts = np.concatenate([np.asarray(p.exterior.coords) for p in polys], axis=0)
    return pts  # columns: lon, lat


def _boundary_paths(boundary) -> list[Path]:
    polys = list(boundary.geoms) if boundary.geom_type == "MultiPolygon" else [boundary]
    return [Path(np.asarray(p.exterior.coords)) for p in polys]


def _projection_params_for_boundary(boundary) -> dict:
    """LCC projection parameters (true_lat_1, true_lat_2, cen_lat, cen_lon) derived
    from a boundary polygon's own extent -- cheap (a bounds lookup + arithmetic),
    not cached separately from `load_conus_boundary`'s own cache."""
    lon_min, lat_min, lon_max, lat_max = boundary.bounds
    cen_lat = 0.5 * (lat_min + lat_max)
    cen_lon = 0.5 * (lon_min + lon_max)
    lat_range = lat_max - lat_min
    return {
        "true_lat_1": cen_lat - lat_range / 4.0,
        "true_lat_2": cen_lat + lat_range / 4.0,
        "cen_lat": cen_lat,
        "cen_lon": cen_lon,
    }


def distance_to_boundary_km(lat2d: np.ndarray, lon2d: np.ndarray, boundary) -> tuple[np.ndarray, np.ndarray]:
    """Core, boundary-agnostic implementation: given ANY shapely (Multi)Polygon,
    compute, for every point in lat2d/lon2d:
    - inside: True if the point falls within `boundary`.
    - distance_km: physical distance (km) to the nearest boundary vertex,
      regardless of inside/outside.

    Not CONUS-specific -- takes the boundary directly, so it's independently
    testable against a synthetic polygon with known dimensions (see
    test_masking.py's exact-math check) without going through the real,
    irregular CONUS shapefile.
    """
    lon_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)

    # inside/outside: vectorized point-in-polygon test in plain lat/lon degrees
    points_lonlat = np.column_stack([lon_norm.ravel(), lat2d.ravel()])
    inside_flat = np.zeros(points_lonlat.shape[0], dtype=bool)
    for path in _boundary_paths(boundary):
        inside_flat |= path.contains_points(points_lonlat)
    inside = inside_flat.reshape(lat2d.shape)

    # distance to border: project grid points and boundary vertices into the
    # SAME km-space (fixed projection anchored to the boundary's own extent, not
    # independently re-derived per input -- see build_projected_coords docstring),
    # then one vectorized nearest-neighbor query.
    proj_kwargs = _projection_params_for_boundary(boundary)

    grid_x_km, grid_y_km = build_projected_coords(lat2d, lon_norm, **proj_kwargs)

    boundary_lonlat = _boundary_vertices_lonlat(boundary)
    boundary_x_km, boundary_y_km = build_projected_coords(
        boundary_lonlat[:, 1].reshape(1, -1), boundary_lonlat[:, 0].reshape(1, -1), **proj_kwargs
    )
    boundary_xy_km = np.column_stack([boundary_x_km.ravel(), boundary_y_km.ravel()])

    tree = cKDTree(boundary_xy_km)
    grid_xy_km = np.column_stack([grid_x_km.ravel(), grid_y_km.ravel()])
    distance_km_flat, _ = tree.query(grid_xy_km)
    distance_km = distance_km_flat.reshape(lat2d.shape)

    return inside, distance_km


def compute_distance_to_border_km(
    lat2d: np.ndarray, lon2d: np.ndarray, resolution: str = "50m", boundary=None
) -> tuple[np.ndarray, np.ndarray]:
    """CONUS-specific wrapper around distance_to_boundary_km: loads (and caches)
    the real CONUS boundary unless one is passed explicitly (for testing)."""
    if boundary is None:
        boundary = load_conus_boundary(resolution)
    return distance_to_boundary_km(lat2d, lon2d, boundary)


def conus_mask(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    boundary_buffer_km: float = 100.0,
    resolution: str = "50m",
    boundary=None,
) -> np.ndarray:
    """True where a grid cell should be masked (excluded): outside the CONUS
    border by more than `boundary_buffer_km`."""
    inside, distance_km = compute_distance_to_border_km(lat2d, lon2d, resolution=resolution, boundary=boundary)
    return (~inside) & (distance_km > boundary_buffer_km)


def conus_mask_east(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    boundary_buffer_km: float = 100.0,
    lon_cutoff: float = -105.0,
    resolution: str = "50m",
    boundary=None,
) -> np.ndarray:
    """Same rule as conus_mask(), plus mask everything west of lon_cutoff
    (eastern ~2/3 CONUS preset)."""
    lon_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    base_mask = conus_mask(
        lat2d, lon2d, boundary_buffer_km=boundary_buffer_km, resolution=resolution, boundary=boundary
    )
    return base_mask | (lon_norm < lon_cutoff)
