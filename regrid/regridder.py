"""Build/cache ESMF conservative regridders (via xesmf) and apply them to fields.

Failure policy (see project memory `feedback-fail-loud-no-fallback`):
  - The regridding *method* failing (degenerate cells, zero spatial overlap, any
    ESMF/xesmf construction error) -> raise RegridError with diagnostics. No fallback
    to a different method.
  - The target domain legitimately extending beyond MRMS's observed coverage (some
    target cells have no contributing source cells) -> NOT an error: warn clearly and
    fill those cells with `fill_value`.
"""

import os
import warnings

import numpy as np

from .grid_spec import CoverageReport, GridSpec, check_coverage, ensure_corners, grid_hash


class RegridError(Exception):
    """Raised when the conservative regridding method itself fails to build/apply."""


def _grid_dict(grid: GridSpec) -> dict:
    # xesmf's ds_to_ESMFgrid() takes plain (n_lat, n_lon) C-ordered arrays and
    # transposes them itself (`lon.T`, `lat.T`) to produce the Fortran order ESMF's
    # backend actually wants -- so *we* must hand it C-contiguous arrays here, not
    # pre-transpose to Fortran order ourselves (that would be undone/doubled by its
    # own `.T`, which is what was silently causing the "not F_CONTIGUOUS" warning
    # and slow weight builds even after an earlier, incorrect asfortranarray fix).
    # Arrays produced by slicing (e.g. crop_to_bbox) are otherwise non-contiguous
    # views, which is what actually needs fixing here.
    return {
        "lat": np.ascontiguousarray(grid.lat2d),
        "lon": np.ascontiguousarray(grid.lon2d),
        "lat_b": np.ascontiguousarray(grid.lat_b),
        "lon_b": np.ascontiguousarray(grid.lon_b),
    }


def build_conservative_regridder(
    src_grid: GridSpec,
    tgt_grid: GridSpec,
    weight_cache_dir: str,
):
    """Build (or reuse a cached) ESMF conservative regridder from src_grid to tgt_grid.

    Cell corners are estimated automatically if not already present on either grid.
    Weight files are cached under `weight_cache_dir`, keyed by a content hash of both
    grids' lat/lon arrays (see grid_spec.grid_hash) -- this speeds up repeated calls
    within one case (same source/target grid pair reused across many timesteps), but
    is not expected to hit across different cases/domains, which move daily.

    Raises RegridError (no fallback) if xesmf/ESMF cannot construct the regridder,
    e.g. due to degenerate cells or zero spatial overlap between the two grids.
    """
    import xesmf

    try:
        src_grid = ensure_corners(src_grid)
        tgt_grid = ensure_corners(tgt_grid)
    except Exception as exc:
        raise RegridError(
            "Conservative regridding failed during cell-corner estimation (e.g. a grid "
            "too small/degenerate to estimate corners from, needs at least 2x2 cells). "
            f"src_grid shape={src_grid.shape}; tgt_grid shape={tgt_grid.shape}. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    os.makedirs(weight_cache_dir, exist_ok=True)
    key = f"conservative__{grid_hash(src_grid)}__{grid_hash(tgt_grid)}"
    weight_path = os.path.join(weight_cache_dir, key + ".nc")

    coverage = check_coverage(tgt_grid, src_grid)

    try:
        if os.path.exists(weight_path):
            regridder = xesmf.Regridder(
                _grid_dict(src_grid),
                _grid_dict(tgt_grid),
                method="conservative",
                weights=weight_path,
                reuse_weights=True,
                unmapped_to_nan=True,
            )
        else:
            regridder = xesmf.Regridder(
                _grid_dict(src_grid),
                _grid_dict(tgt_grid),
                method="conservative",
                filename=weight_path,
                unmapped_to_nan=True,
            )
    except Exception as exc:
        raise RegridError(
            "Conservative regridding failed to build. This is a hard failure of the "
            "regridding method itself (e.g. degenerate cells or zero spatial overlap) "
            "-- no fallback method is applied. Diagnostics: "
            f"src_grid shape={src_grid.shape}, lat=[{src_grid.lat2d.min():.3f},"
            f"{src_grid.lat2d.max():.3f}], lon=[{src_grid.lon2d.min():.3f},"
            f"{src_grid.lon2d.max():.3f}]; tgt_grid shape={tgt_grid.shape}, "
            f"lat=[{tgt_grid.lat2d.min():.3f},{tgt_grid.lat2d.max():.3f}], "
            f"lon=[{tgt_grid.lon2d.min():.3f},{tgt_grid.lon2d.max():.3f}]; "
            f"fraction of target outside source bbox={coverage.frac_outside_source_bbox:.3f}. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    return regridder


EARTH_RADIUS_KM = 6371.0
DEFAULT_MAX_CRESSMAN_NNZ = 1_500_000_000  # ~12 GB of float32 weights + int32/int64 indices


class CressmanRegridder:
    """Single-pass, fixed-radius Cressman operator: each target value is the
    weighted mean of the source points within `radius_km`, with weights
    (R^2 - r^2) / (R^2 + r^2). Holds the (unnormalized) weights as a sparse
    n_target x n_source matrix and is called like an xesmf regridder
    (`regridder(field2d) -> target2d`), so regrid_field() uses it unchanged.

    Calling it returns (W @ x) / (W @ 1): a normalized, linear operator. Targets
    with no source within the radius are NaN (as xesmf's unmapped_to_nan), and
    regrid_field's valid-fraction handling of missing sources gives exactly a
    Cressman average over the valid points only.
    """

    def __init__(self, weights, target_shape: tuple, radius_km: float):
        self.weights = weights.tocsr()
        self.target_shape = tuple(target_shape)
        self.radius_km = float(radius_km)
        self._row_sums = np.asarray(self.weights.sum(axis=1)).ravel().astype(np.float64)

    def __call__(self, field2d):
        x = np.asarray(field2d, dtype=np.float64).ravel()
        if x.size != self.weights.shape[1]:
            raise RegridError(
                f"CressmanRegridder: field has {x.size} points but the operator expects "
                f"{self.weights.shape[1]} source points"
            )
        num = self.weights @ x
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(self._row_sums > 0, num / self._row_sums, np.nan)
        return out.reshape(self.target_shape)


def _unit_xyz_km(lat2d, lon2d) -> np.ndarray:
    lat = np.radians(np.asarray(lat2d, dtype=np.float64).ravel())
    lon = np.radians(np.asarray(lon2d, dtype=np.float64).ravel())
    c = np.cos(lat)
    return EARTH_RADIUS_KM * np.column_stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)])


def build_cressman_regridder(
    src_grid: GridSpec,
    tgt_grid: GridSpec,
    radius_km: float,
    weight_cache_dir: str,
    chunk_size: int = 200_000,
    max_nnz: int = DEFAULT_MAX_CRESSMAN_NNZ,
) -> CressmanRegridder:
    """Build (or reuse a cached) single-pass Cressman operator from src_grid to
    tgt_grid with radius of influence `radius_km`.

    Points are placed on a sphere of radius EARTH_RADIUS_KM; distances are chord
    lengths, within 1e-6 (relative) of great-circle distance for radii <= 50 km.
    Pairs within the radius come from scipy cKDTree.sparse_distance_matrix, built
    in target chunks of `chunk_size` to bound memory. Weights are cached as a
    scipy .npz under `weight_cache_dir`, keyed by the radius and a content hash of
    both grids (as for the conservative weights).

    Raises RegridError (no fallback) for a non-positive radius, an estimated
    operator size above `max_nnz` nonzeros (checked before building), or no
    source point within the radius of any target point.
    """
    from scipy import sparse
    from scipy.spatial import cKDTree

    if not (radius_km is not None and np.isfinite(radius_km) and radius_km > 0):
        raise RegridError(f"Cressman radius must be a positive number of km, got {radius_km!r}")

    os.makedirs(weight_cache_dir, exist_ok=True)
    key = f"cressman_r{radius_km:g}km__{grid_hash(src_grid)}__{grid_hash(tgt_grid)}"
    weight_path = os.path.join(weight_cache_dir, key + ".npz")
    if os.path.exists(weight_path):
        return CressmanRegridder(sparse.load_npz(weight_path), tgt_grid.shape, radius_km)

    src_xyz = _unit_xyz_km(src_grid.lat2d, src_grid.lon2d)
    tgt_xyz = _unit_xyz_km(tgt_grid.lat2d, tgt_grid.lon2d)

    # source point spacing from neighboring points, to estimate the operator size up front
    ny, nx = src_grid.shape
    dy = np.median(np.linalg.norm(np.diff(src_xyz.reshape(ny, nx, 3), axis=0), axis=-1)) if ny > 1 else 1.0
    dx = np.median(np.linalg.norm(np.diff(src_xyz.reshape(ny, nx, 3), axis=1), axis=-1)) if nx > 1 else 1.0
    est_nnz = len(tgt_xyz) * np.pi * radius_km**2 / max(dx * dy, 1e-12)
    if est_nnz > max_nnz:
        raise RegridError(
            f"Cressman operator too large: ~{est_nnz:.3g} nonzero weights expected "
            f"({len(tgt_xyz):,} target points x ~{np.pi * radius_km**2 / (dx * dy):.0f} source points "
            f"within {radius_km:g} km) exceeds max_nnz={max_nnz:.3g}. Use a smaller radius or "
            f"target grid, or raise max_nnz if the memory is available."
        )

    src_tree = cKDTree(src_xyz)
    rows, cols, vals = [], [], []
    r2 = radius_km**2
    for start in range(0, len(tgt_xyz), chunk_size):
        chunk = tgt_xyz[start:start + chunk_size]
        dist = cKDTree(chunk).sparse_distance_matrix(src_tree, radius_km, output_type="coo_matrix")
        d2 = dist.data.astype(np.float64) ** 2
        keep = d2 < r2  # strictly inside the radius (the weight is 0 on it)
        rows.append(dist.row[keep] + start)
        cols.append(dist.col[keep])
        vals.append(((r2 - d2[keep]) / (r2 + d2[keep])).astype(np.float32))
    # (coincident points: sparse_distance_matrix keeps zero-distance pairs, so they get weight 1 above)
    weights = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(tgt_xyz), len(src_xyz)),
    ).tocsr()
    weights.sum_duplicates()
    if weights.nnz == 0:
        raise RegridError(
            f"Cressman regridding failed: no source point lies within {radius_km:g} km of any target point "
            f"(src lat [{src_grid.lat2d.min():.3f},{src_grid.lat2d.max():.3f}], lon "
            f"[{src_grid.lon2d.min():.3f},{src_grid.lon2d.max():.3f}]; tgt lat "
            f"[{tgt_grid.lat2d.min():.3f},{tgt_grid.lat2d.max():.3f}], lon "
            f"[{tgt_grid.lon2d.min():.3f},{tgt_grid.lon2d.max():.3f}])"
        )
    sparse.save_npz(weight_path, weights)
    return CressmanRegridder(weights, tgt_grid.shape, radius_km)


def regrid_field(
    regridder,
    data2d: np.ndarray,
    fill_value: float = 0.0,
    missing_value: float | None = None,
    min_valid_fraction: float = 0.5,
) -> tuple[np.ndarray, CoverageReport]:
    """Regrid one 2D field, padding any unmapped/no-coverage target cells.

    Two distinct sources of "no coverage" are both handled here as the same
    legitimate (non-error) outcome:

    1. Target cells with no *geometric* overlap with the source grid at all --
       because the regridder was built with unmapped_to_nan=True, ESMF leaves
       these as NaN instead of silently 0.
    2. Target cells that do geometrically overlap the source, but where the
       contributing source cells are themselves flagged `missing_value` (e.g.
       MRMS's -999 "no radar coverage" sentinel). Naively conservative-averaging
       -999 alongside real dBZ values would badly corrupt the result near
       coverage-gap boundaries, so when `missing_value` is given, this uses the
       standard valid-fraction trick: regrid `data*valid_mask` and `valid_mask`
       separately (through the *same* precomputed weights -- regridding is
       linear, so this needs no extra weight computation), then divide. Target
       cells where the regridded valid fraction is below `min_valid_fraction`
       are treated as unmapped too.

    Either way this is a legitimate coverage gap, not a method failure: warn
    with the affected fraction and fill with `fill_value`, rather than raising
    or returning unremarked NaNs.
    """
    data2d = np.ascontiguousarray(data2d, dtype=np.float64)

    if missing_value is not None:
        valid_mask = (data2d != missing_value) & np.isfinite(data2d)
        data_for_regrid = np.where(valid_mask, data2d, 0.0)

        regridded_data = np.asarray(regridder(data_for_regrid))
        regridded_valid_frac = np.asarray(regridder(valid_mask.astype(np.float64)))

        with np.errstate(invalid="ignore", divide="ignore"):
            regridded = regridded_data / regridded_valid_frac
        low_valid = ~(regridded_valid_frac >= min_valid_fraction)
        regridded = np.where(low_valid, np.nan, regridded)
    else:
        regridded = np.asarray(regridder(data2d))

    n_total = regridded.size
    unmapped = np.isnan(regridded)
    n_unmapped = int(np.sum(unmapped))

    report = CoverageReport(
        n_target_cells=n_total,
        n_outside_source_bbox=n_unmapped,
        frac_outside_source_bbox=n_unmapped / n_total,
    )

    if n_unmapped > 0:
        pct = 100.0 * n_unmapped / n_total
        warnings.warn(
            f"regrid_field: {n_unmapped} of {n_total} target cells ({pct:.2f}%) had no "
            f"valid source (MRMS) coverage; padding with fill_value={fill_value}.",
            stacklevel=2,
        )
        regridded = np.where(unmapped, fill_value, regridded)

    return regridded, report
