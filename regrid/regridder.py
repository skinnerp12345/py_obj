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
