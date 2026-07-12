"""Step 1 validation: MRMS-to-model-grid conservative regridding.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_regrid_step1.py -v -s

Uses the real, bundled python_obj/sample_data/ (a raw native-grid MRMS file +
a small, pre-cropped real MPAS target grid) -- both are already small, so
unlike when this test ran against the full multi-GB local test_mrms/test_mpas/
directories, no further sub-cropping for speed is needed here.
"""

import glob
import os
import time
import warnings

import numpy as np
import pytest

from python_obj.regrid import (
    RegridError,
    build_conservative_regridder,
    check_coverage,
    crop_to_bbox,
    load_mrms_grib2,
    load_target_grid,
    regrid_field,
)
from python_obj.regrid.grid_spec import GridSpec, estimate_cell_corners
from python_obj.regrid.io_mrms import MRMS_MISSING_VALUE, MRMS_NEAR_ZERO_VALUE, clip_near_zero_sentinel

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
MRMS_FILE = os.path.join(
    SAMPLE_DATA_DIR,
    "mpas_case/mrms/20230501/MRMS_MergedReflectivityQCComposite_00.50_20230501-010041.grib2.gz",
)
MPAS_FILE = os.path.join(SAMPLE_DATA_DIR, "mpas_case/mpas_mem1/interp_mpas_3km_2023050100_mem1_f001.nc")
WEIGHT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_weight_cache")


def _area_weighted_mean_latlon(
    data: np.ndarray, lat2d: np.ndarray, thresh: float | None = None,
    missing_value: float | None = None,
) -> tuple[float, float]:
    """Independent area-weighted domain mean / above-threshold area (km^2) for a
    regular lat/lon grid, using cos(lat) weighting -- deliberately not reusing any
    xesmf/ESMF internals, so this is a genuine independent check on conservation.
    Missing/no-coverage cells (e.g. MRMS's -999 flag) are excluded, not blended in."""
    valid = np.isfinite(data)
    if missing_value is not None:
        valid &= data != missing_value
    weights = np.where(valid, np.cos(np.deg2rad(lat2d)), 0.0)
    mean_val = float(np.sum(np.where(valid, data, 0.0) * weights) / np.sum(weights))
    if thresh is not None:
        # approximate km^2 per source cell at this latitude for a 0.01 deg grid
        dlat_km = 0.01 * 111.0
        dlon_km = 0.01 * 111.0 * np.cos(np.deg2rad(lat2d))
        cell_area = dlat_km * dlon_km
        above_area = float(np.sum(np.where(valid & (data > thresh), cell_area, 0.0)))
        return mean_val, above_area
    return mean_val, float("nan")


def _area_weighted_mean_projected(data: np.ndarray, dx_km: float, thresh: float | None = None) -> tuple[float, float]:
    """Independent domain mean / above-threshold area (km^2) for a roughly-uniform
    projected grid (equal-area cells), e.g. the 3-km MPAS grid. NaNs (unmapped/
    no-coverage cells) are excluded via nanmean/nansum, not blended in as 0."""
    mean_val = float(np.nanmean(data))
    if thresh is not None:
        cell_area = dx_km * dx_km
        above_area = float(np.nansum(data > thresh)) * cell_area
        return mean_val, above_area
    return mean_val, float("nan")


@pytest.fixture(scope="module")
def mrms_field():
    assert os.path.exists(MRMS_FILE), f"test MRMS file missing: {MRMS_FILE}"
    return load_mrms_grib2(MRMS_FILE)


@pytest.fixture(scope="module")
def mpas_grid():
    assert os.path.exists(MPAS_FILE), f"test MPAS file missing: {MPAS_FILE}"
    return load_target_grid(MPAS_FILE, "latitude", "longitude")


# --- Check 1: grib2 loader -------------------------------------------------

def test_load_mrms_grib2(mrms_field):
    assert mrms_field.data.shape == (3500, 7000)
    assert 19.5 < mrms_field.lat2d.min() < 20.5
    assert 54.5 < mrms_field.lat2d.max() < 55.5
    assert mrms_field.valid_time.strftime("%Y%m%d") == "20230501"
    assert mrms_field.valid_time.hour == 1
    assert np.any(mrms_field.data == mrms_field.missing_value), "expected -999 missing flag present in raw data"
    assert np.any(mrms_field.data == MRMS_NEAR_ZERO_VALUE), (
        "expected -99 'valid, near/below-zero reflectivity' sentinel present in raw data "
        "(confirmed by user: distinct from -999 'no coverage', must be floored not excluded)"
    )
    print(f"\n[check1] MRMS shape={mrms_field.data.shape} valid_time={mrms_field.valid_time} "
          f"data range=({mrms_field.data.min()}, {mrms_field.data.max()})")


def test_clip_near_zero_sentinel():
    data = np.array([-999.0, -99.0, 0.0, 15.0, 45.0])
    clipped = clip_near_zero_sentinel(data)
    np.testing.assert_array_equal(clipped, np.array([-999.0, 0.0, 0.0, 15.0, 45.0]))
    print("\n[check1b] clip_near_zero_sentinel: -99 -> 0.0, -999 and real values untouched")


# --- Check 2: netcdf target-grid loader ------------------------------------

def test_load_target_grid(mpas_grid):
    assert mpas_grid.shape == (250, 250)
    assert mpas_grid.lat2d.ndim == 2 and mpas_grid.lon2d.ndim == 2
    print(f"\n[check2] MPAS grid shape={mpas_grid.shape} "
          f"lat=({mpas_grid.lat2d.min():.2f},{mpas_grid.lat2d.max():.2f}) "
          f"lon=({mpas_grid.lon2d.min():.2f},{mpas_grid.lon2d.max():.2f})")


# --- Check 3: bbox cropping -------------------------------------------------

def test_crop_to_bbox(mrms_field, mpas_grid):
    src_grid = GridSpec(lat2d=mrms_field.lat2d, lon2d=mrms_field.lon2d)
    cropped_grid, cropped_data = crop_to_bbox(
        src_grid, mrms_field.data,
        lat_min=mpas_grid.lat2d.min(), lat_max=mpas_grid.lat2d.max(),
        lon_min=mpas_grid.lon2d.min(), lon_max=mpas_grid.lon2d.max(),
        buffer_deg=0.3,
    )
    assert cropped_grid.shape[0] < mrms_field.data.shape[0]
    n_full = mrms_field.data.shape[0] * mrms_field.data.shape[1]
    n_cropped = cropped_grid.shape[0] * cropped_grid.shape[1]
    assert n_cropped < n_full
    print(f"\n[check3] full={mrms_field.data.shape} -> cropped={cropped_grid.shape} "
          f"({100*n_cropped/n_full:.1f}% of full grid)")


# --- Fast build/regrid on a real, but small, regional subset ---------------

def _small_real_subset(mrms_field, mpas_grid):
    """A real (not synthetic) MRMS region that geometrically bounds the
    bundled sample's small MPAS target grid in full (Upper Midwest / Great
    Lakes, rows/cols computed to cover the MPAS grid's own lat/lon bbox plus
    margin) -- confirmed effectively 0% -999 "no coverage" cells (unlike an
    earlier candidate crop near the Atlantic coast, where ~45% of the region
    was legitimately outside MRMS's radar coverage), so no representativeness
    error from partial coverage contaminates the conservation check below.
    Check 7 further down separately exercises the "target legitimately
    extends past source bbox" case on purpose, using a deliberately extended
    target grid."""
    src_grid = GridSpec(
        lat2d=mrms_field.lat2d[570:1360, 4340:5470],
        lon2d=mrms_field.lon2d[570:1360, 4340:5470],
    )
    # -99 ("valid, near/below-zero reflectivity") must be floored to 0 before any
    # averaging/regridding; only -999 ("no coverage") stays a sentinel to exclude.
    src_data = clip_near_zero_sentinel(mrms_field.data[570:1360, 4340:5470])

    lat_min, lat_max = src_grid.lat2d.min(), src_grid.lat2d.max()
    lon_min, lon_max = src_grid.lon2d.min(), src_grid.lon2d.max()
    lon_min_signed = lon_min - 360.0 if lon_min > 180 else lon_min
    lon_max_signed = lon_max - 360.0 if lon_max > 180 else lon_max

    tgt_mask_rows = np.where(
        np.any(
            (mpas_grid.lat2d >= lat_min) & (mpas_grid.lat2d <= lat_max)
            & (mpas_grid.lon2d >= lon_min_signed) & (mpas_grid.lon2d <= lon_max_signed),
            axis=1,
        )
    )[0]
    tgt_mask_cols = np.where(
        np.any(
            (mpas_grid.lat2d >= lat_min) & (mpas_grid.lat2d <= lat_max)
            & (mpas_grid.lon2d >= lon_min_signed) & (mpas_grid.lon2d <= lon_max_signed),
            axis=0,
        )
    )[0]
    assert tgt_mask_rows.size > 0 and tgt_mask_cols.size > 0, "test subset regions do not overlap"
    r0, r1 = tgt_mask_rows.min(), tgt_mask_rows.max() + 1
    c0, c1 = tgt_mask_cols.min(), tgt_mask_cols.max() + 1
    tgt_grid = GridSpec(lat2d=mpas_grid.lat2d[r0:r1, c0:c1], lon2d=mpas_grid.lon2d[r0:r1, c0:c1])
    return src_grid, src_data, tgt_grid


# --- Check 4: conservative regrid + independent numeric conservation checks -

def test_build_and_regrid_small_subset(mrms_field, mpas_grid):
    src_grid, src_data, tgt_grid = _small_real_subset(mrms_field, mpas_grid)

    t0 = time.time()
    regridder = build_conservative_regridder(src_grid, tgt_grid, weight_cache_dir=WEIGHT_CACHE_DIR)
    build_time = time.time() - t0

    # use fill_value=nan here specifically so the conservation check below compares
    # only over the region actually covered by source data, not blended with padded
    # cells -- the fill_value=0.0 production default's warn+pad behavior for
    # legitimately uncovered cells is validated separately in check 7.
    out, report = regrid_field(regridder, src_data, fill_value=np.nan, missing_value=MRMS_MISSING_VALUE)

    src_mean, src_area = _area_weighted_mean_latlon(
        src_data, src_grid.lat2d, thresh=20.0, missing_value=MRMS_MISSING_VALUE
    )
    tgt_mean, tgt_area = _area_weighted_mean_projected(out, dx_km=3.0, thresh=20.0)

    rel_mean_diff = abs(tgt_mean - src_mean) / max(abs(src_mean), 1e-6)
    rel_area_diff = abs(tgt_area - src_area) / max(src_area, 1e-6)

    print(f"\n[check4] build_time={build_time:.2f}s src_mean={src_mean:.3f} tgt_mean={tgt_mean:.3f} "
          f"rel_diff={rel_mean_diff:.3f}; src_area>20dBZ={src_area:.0f}km2 "
          f"tgt_area>20dBZ={tgt_area:.0f}km2 rel_diff={rel_area_diff:.3f}; coverage={report}")

    # conservative remap should closely preserve domain-mean reflectivity and
    # above-threshold area over a fully-overlapping region; allow generous
    # tolerance for grid-representativeness error between very different resolutions
    assert rel_mean_diff < 0.25
    assert rel_area_diff < 0.35


# --- Check 5: weight-cache reuse --------------------------------------------

def test_weight_cache_reuse(mrms_field, mpas_grid):
    src_grid, src_data, tgt_grid = _small_real_subset(mrms_field, mpas_grid)

    regridder1 = build_conservative_regridder(src_grid, tgt_grid, weight_cache_dir=WEIGHT_CACHE_DIR)
    cache_files = glob.glob(os.path.join(WEIGHT_CACHE_DIR, "conservative__*.nc"))
    assert len(cache_files) >= 1
    weight_file = cache_files[0]
    mtime_1 = os.path.getmtime(weight_file)

    t0 = time.time()
    regridder2 = build_conservative_regridder(src_grid, tgt_grid, weight_cache_dir=WEIGHT_CACHE_DIR)
    reuse_time = time.time() - t0
    mtime_2 = os.path.getmtime(weight_file)

    print(f"\n[check5] weight file={weight_file} mtime unchanged={mtime_1 == mtime_2} reuse_call_time={reuse_time:.3f}s")
    assert mtime_1 == mtime_2, "weight file was rewritten instead of reused"


# --- Check 6: hard failure, no fallback -------------------------------------
#
# Note: zero geometric overlap between source/target is *not* by itself an ESMF
# construction failure -- xesmf/ESMF happily builds a (fully-unmapped) regridder
# for disjoint grids, and that case is exercised as a legitimate coverage gap in
# check 7 below (warn + pad, not an error). A genuine *method* failure is e.g.
# malformed/inconsistent grid geometry (verified empirically: a lat_b/lon_b
# corner array whose shape doesn't match (ny+1, nx+1) makes ESMF/xesmf itself
# raise, which is what this test exercises).

def test_regrid_error_on_malformed_grid():
    lat = np.linspace(35.0, 36.0, 10)
    lon = np.linspace(-100.0, -99.0, 10)
    lon2d, lat2d = np.meshgrid(lon, lat)
    tgt_grid = GridSpec(lat2d=lat2d.copy(), lon2d=lon2d.copy())

    malformed_src = GridSpec(
        lat2d=lat2d, lon2d=lon2d,
        lat_b=np.zeros((3, 3)), lon_b=np.zeros((3, 3)),  # wrong shape: should be (11, 11)
    )

    with pytest.raises(RegridError) as exc_info:
        build_conservative_regridder(malformed_src, tgt_grid, weight_cache_dir=WEIGHT_CACHE_DIR)
    print(f"\n[check6] RegridError raised as expected: {exc_info.value}")


# --- Check 7: legitimate coverage gap -> warn + pad, not an error -----------

def test_coverage_gap_warns_and_pads(mrms_field, mpas_grid):
    src_grid, src_data, tgt_grid_real = _small_real_subset(mrms_field, mpas_grid)

    # extend the target grid well past the source's actual extent on one side,
    # simulating a forecast domain larger than MRMS's observed coverage
    extra_lat = np.linspace(tgt_grid_real.lat2d.max() + 0.05, tgt_grid_real.lat2d.max() + 2.0, 20)
    padded_lat_col = np.tile(extra_lat[:, None], (1, tgt_grid_real.lon2d.shape[1]))
    padded_lon_col = np.tile(tgt_grid_real.lon2d[0:1, :], (20, 1))
    tgt_lat2d = np.vstack([tgt_grid_real.lat2d, padded_lat_col])
    tgt_lon2d = np.vstack([tgt_grid_real.lon2d, padded_lon_col])
    tgt_grid = GridSpec(lat2d=tgt_lat2d, lon2d=tgt_lon2d)

    coverage_precheck = check_coverage(tgt_grid, src_grid)
    assert coverage_precheck.n_outside_source_bbox > 0

    regridder = build_conservative_regridder(src_grid, tgt_grid, weight_cache_dir=WEIGHT_CACHE_DIR)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, report = regrid_field(regridder, src_data, fill_value=0.0)
        coverage_warnings = [w for w in caught if "no valid source (MRMS) coverage" in str(w.message)]
        assert len(coverage_warnings) == 1, "expected exactly one clear coverage warning, not silence or an error"

    assert report.n_outside_source_bbox > 0
    assert np.all(out[-5:, :] == 0.0), "unmapped padded rows should be filled with fill_value, not NaN"
    assert not np.any(np.isnan(out)), "no unremarked NaNs should remain in the output"
    print(f"\n[check7] coverage gap handled: {report}; warning message="
          f"'{coverage_warnings[0].message}'")
