"""Grid representation, cell-corner estimation, bbox cropping, and coverage checks."""

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class GridSpec:
    lat2d: np.ndarray
    lon2d: np.ndarray
    lat_b: np.ndarray | None = None  # cell corners, shape (ny+1, nx+1)
    lon_b: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat2d.shape


@dataclass
class CoverageReport:
    n_target_cells: int
    n_outside_source_bbox: int
    frac_outside_source_bbox: float

    @property
    def fully_covered(self) -> bool:
        return self.n_outside_source_bbox == 0


def estimate_cell_corners(lat2d: np.ndarray, lon2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate (ny+1, nx+1) cell-corner lat/lon arrays from (ny, nx) cell-center arrays.

    Standard linear-extrapolation approach: interior corners are the average of the
    4 surrounding cell centers; edge/corner-of-domain corners are extrapolated linearly
    from the two nearest interior cell centers. This is the conventional approximation
    used when a grid only carries cell centers (no native corner/bounds variable), as is
    the case for both the MRMS grib2 fields and the interpolated MPAS test file.
    """
    ny, nx = lat2d.shape

    def _corners_1axis(field: np.ndarray) -> np.ndarray:
        # pad by extrapolating one extra row/col on each side, then average 2x2 blocks
        padded = np.empty((ny + 2, nx + 2), dtype=np.float64)
        padded[1:-1, 1:-1] = field
        # extrapolate rows
        padded[0, 1:-1] = 2 * field[0, :] - field[1, :]
        padded[-1, 1:-1] = 2 * field[-1, :] - field[-2, :]
        # extrapolate cols (using already-filled rows for corners of the padded array)
        padded[1:-1, 0] = 2 * field[:, 0] - field[:, 1]
        padded[1:-1, -1] = 2 * field[:, -1] - field[:, -2]
        # corners of the padded array (bilinear extrapolation of the 4 true corners)
        padded[0, 0] = 2 * padded[1, 0] - padded[2, 0]
        padded[0, -1] = 2 * padded[1, -1] - padded[2, -1]
        padded[-1, 0] = 2 * padded[-2, 0] - padded[-3, 0]
        padded[-1, -1] = 2 * padded[-2, -1] - padded[-3, -1]

        # corners = average of each 2x2 neighborhood in the padded array
        corners = 0.25 * (
            padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:]
        )
        return corners

    lat_b = _corners_1axis(lat2d)
    lon_b = _corners_1axis(lon2d)
    return lat_b, lon_b


def ensure_corners(grid: GridSpec) -> GridSpec:
    """Return a GridSpec guaranteed to have lat_b/lon_b, estimating them if absent."""
    if grid.lat_b is not None and grid.lon_b is not None:
        return grid
    lat_b, lon_b = estimate_cell_corners(grid.lat2d, grid.lon2d)
    return GridSpec(lat2d=grid.lat2d, lon2d=grid.lon2d, lat_b=lat_b, lon_b=lon_b)


def crop_to_bbox(
    grid: GridSpec,
    data: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    buffer_deg: float = 1.0,
) -> tuple[GridSpec, np.ndarray]:
    """Crop a (lat2d, lon2d, data) grid to a bounding box + buffer, by row/col mask.

    Uses the smallest rectangular index slice that contains all points within the
    padded bbox (not a ragged boolean mask), since ESMF/xesmf need a regular 2D array.
    """
    lon2d = grid.lon2d
    # normalize longitudes to the same convention as the bbox (MRMS uses 0-360; targets
    # are typically -180-180). Work in -180..180 internally for the comparison.
    lon2d_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)

    in_box = (
        (grid.lat2d >= lat_min - buffer_deg)
        & (grid.lat2d <= lat_max + buffer_deg)
        & (lon2d_norm >= lon_min - buffer_deg)
        & (lon2d_norm <= lon_max + buffer_deg)
    )
    if not np.any(in_box):
        raise ValueError(
            f"crop_to_bbox: no source grid points fall within lat=[{lat_min},{lat_max}] "
            f"lon=[{lon_min},{lon_max}] (+/-{buffer_deg} deg buffer); grids do not overlap."
        )

    rows = np.where(np.any(in_box, axis=1))[0]
    cols = np.where(np.any(in_box, axis=0))[0]
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    cropped_grid = GridSpec(
        lat2d=grid.lat2d[r0:r1, c0:c1],
        lon2d=grid.lon2d[r0:r1, c0:c1],
    )
    cropped_data = data[r0:r1, c0:c1]
    return cropped_grid, cropped_data


def grid_hash(grid: GridSpec, precision: int = 4) -> str:
    """Content hash of a grid's lat/lon arrays, for weight-cache keys.

    Rounds to `precision` decimal degrees before hashing so trivial floating-point
    noise doesn't defeat cache reuse, while still being effectively unique per
    distinct domain (WoFS/MPAS domains move most days, so cross-case cache hits are
    not expected in general -- this mainly speeds up repeated calls *within* one case).
    """
    lat_bytes = np.round(grid.lat2d, precision).tobytes()
    lon_bytes = np.round(grid.lon2d, precision).tobytes()
    shape_bytes = str(grid.lat2d.shape).encode()
    h = hashlib.sha256()
    h.update(shape_bytes)
    h.update(lat_bytes)
    h.update(lon_bytes)
    return h.hexdigest()[:16]


def check_coverage(tgt_grid: GridSpec, src_grid: GridSpec) -> CoverageReport:
    """Fraction of the target grid's points falling outside the source grid's bounding box.

    This is a coarse, whole-bbox check for "does the source even span the target domain",
    distinct from the finer per-cell "unmapped after regridding" check done in regridder.py
    (a target point can be inside the source bbox but still unmapped, e.g. in a small gap).
    """
    src_lon_norm = np.where(src_grid.lon2d > 180.0, src_grid.lon2d - 360.0, src_grid.lon2d)
    tgt_lon_norm = np.where(tgt_grid.lon2d > 180.0, tgt_grid.lon2d - 360.0, tgt_grid.lon2d)

    src_lat_min, src_lat_max = src_grid.lat2d.min(), src_grid.lat2d.max()
    src_lon_min, src_lon_max = src_lon_norm.min(), src_lon_norm.max()

    outside = (
        (tgt_grid.lat2d < src_lat_min)
        | (tgt_grid.lat2d > src_lat_max)
        | (tgt_lon_norm < src_lon_min)
        | (tgt_lon_norm > src_lon_max)
    )
    n_total = tgt_grid.lat2d.size
    n_outside = int(np.sum(outside))
    return CoverageReport(
        n_target_cells=n_total,
        n_outside_source_bbox=n_outside,
        frac_outside_source_bbox=n_outside / n_total,
    )
