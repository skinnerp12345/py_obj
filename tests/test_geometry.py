"""Step 2 validation: format adapters + the geometry/anisotropy fix.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_geometry.py -v -s
"""

import os
from datetime import datetime

import numpy as np
import pytest

from python_obj.obj_core import boundary_dist_km, build_projected_coords, centroid_dist_km, pixel_area_km2
from python_obj.regrid import load_mrms_netcdf, load_model_netcdf

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")


# --- Check 1: anisotropy bug/fix regression -------------------------------

def _synthetic_mrms_shaped_grid(lat_step=0.05, lon_step=0.05):
    """A smaller-but-representative stand-in for the real 0.01-deg, 20-55N MRMS
    grid (coarsened purely so the test runs fast; the anisotropy is a property
    of the lat/lon spacing and range, not the resolution)."""
    lat = np.arange(20.0, 55.0, lat_step)
    lon = np.arange(230.0, 300.0, lon_step)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lat2d, lon2d


def test_anisotropy_demonstrated_and_fixed():
    lat2d, lon2d = _synthetic_mrms_shaped_grid()
    x2d, y2d = build_projected_coords(lat2d, lon2d)

    dx_low_lat = np.median(np.abs(np.gradient(x2d[0, :])))    # ~20N row
    dx_high_lat = np.median(np.abs(np.gradient(x2d[-1, :])))  # ~55N row
    ratio = dx_high_lat / dx_low_lat

    print(f"\n[geom-check1] km/gridpoint-in-longitude: 20N={dx_low_lat:.4f} km, "
          f"55N={dx_high_lat:.4f} km, ratio={ratio:.3f}")

    # the documented effect is ~35% shrinkage (ratio ~0.65); assert it's
    # substantial and in the right direction, not exactly 1.0 (isotropic)
    assert 0.5 < ratio < 0.8, "expected a real, substantial anisotropy between 20N and 55N"

    # Demonstrate the actual bug this fixes: two object pairs separated by the
    # SAME number of longitude gridpoints at low vs. high latitude would look
    # identically distant under the legacy pixel-index-distance approach, but
    # are physically very different distances apart -- a fixed 40 km threshold
    # should behave consistently (in physical terms) at both latitudes when
    # measured in km-space, unlike a fixed pixel-count threshold would.
    n_cols_apart = 100
    cent_low = ((0, 0), (0, n_cols_apart))       # (row, col)-style index pair at row 0
    cent_high = ((-1, 0), (-1, n_cols_apart))    # same column separation, top row

    # old (buggy) approach: pixel-index Euclidean distance, identical at both latitudes
    old_style_dist_low = float(np.hypot(0, n_cols_apart))
    old_style_dist_high = float(np.hypot(0, n_cols_apart))
    assert old_style_dist_low == old_style_dist_high  # this is exactly the bug

    # new (fixed) approach: actual km-space distance, correctly different
    xy_low_a = (x2d[0, 0], y2d[0, 0])
    xy_low_b = (x2d[0, n_cols_apart], y2d[0, n_cols_apart])
    xy_high_a = (x2d[-1, 0], y2d[-1, 0])
    xy_high_b = (x2d[-1, n_cols_apart], y2d[-1, n_cols_apart])

    km_dist_low = centroid_dist_km(xy_low_a, xy_low_b)
    km_dist_high = centroid_dist_km(xy_high_a, xy_high_b)
    print(f"[geom-check1] {n_cols_apart}-gridpoint separation: {km_dist_low:.1f} km at 20N vs "
          f"{km_dist_high:.1f} km at 55N (physically different, correctly)")
    assert km_dist_low > km_dist_high * 1.2, "km-space distance should differ meaningfully by latitude"


# --- Check 2: load_mrms_netcdf against real Step 1b output -----------------

def test_load_mrms_netcdf_real_file():
    path = os.path.join(SAMPLE_DATA_DIR, "mpas_case/interp_mrms/20230501/interp_mrms_20230501_010041.nc")
    assert os.path.exists(path), f"expected bundled Step 1b-style output at {path}"

    field = load_mrms_netcdf(path)
    assert field.data.shape == (250, 250)
    assert 41.0 < field.lat2d.min() < 42.0
    assert -75.0 > field.lon2d.max() > -76.0 or field.lon2d.max() < 0  # signed-lon sanity
    assert field.valid_time == datetime(2023, 5, 1, 1, 0, 41)
    assert field.missing_value == -999.0
    print(f"\n[geom-check2] loaded {path}: shape={field.data.shape} "
          f"valid_time={field.valid_time} data range=({field.data.min():.1f},{field.data.max():.1f})")


def test_load_mrms_netcdf_missing_valid_time_attr_raises(tmp_path):
    import netCDF4
    bad_file = str(tmp_path / "no_valid_time.nc")
    with netCDF4.Dataset(bad_file, "w") as ds:
        ds.createDimension("y", 2)
        ds.createDimension("x", 2)
        ds.createVariable("lat", "f8", ("y", "x"))[:, :] = 30.0
        ds.createVariable("lon", "f8", ("y", "x"))[:, :] = -90.0
        ds.createVariable("refl_consv", "f8", ("y", "x"))[:, :] = 20.0
        # deliberately no valid_time global attribute

    with pytest.raises(ValueError, match="valid_time"):
        load_mrms_netcdf(bad_file)
    print("\n[geom-check2b] load_mrms_netcdf correctly raises when valid_time attr is absent "
          "and none is supplied (no silent guess from filename)")


# --- Check 3: load_model_netcdf against test_mpas/ --------------------------

def test_load_model_netcdf_f001_and_f003():
    f001 = load_model_netcdf(
        os.path.join(SAMPLE_DATA_DIR, "mpas_case/mpas_mem1/interp_mpas_3km_2023050100_mem1_f001.nc"),
        varname="refl10cm_max",
    )
    assert f001.data.shape == (250, 250)
    assert f001.valid_time == datetime(2023, 5, 1, 1, 0, 0)

    f003 = load_model_netcdf(
        os.path.join(SAMPLE_DATA_DIR, "mpas_case/mpas_mem1/interp_mpas_3km_2023050100_mem1_f003.nc"),
        varname="refl10cm_max",
    )
    assert f003.valid_time == datetime(2023, 5, 1, 3, 0, 0)
    print(f"\n[geom-check3] f001 valid_time={f001.valid_time}, f003 valid_time={f003.valid_time} "
          f"(correctly advanced by 2h via initializationTime+forecastHour)")


def test_load_model_netcdf_missing_attrs_raises(tmp_path):
    import netCDF4
    bad_file = str(tmp_path / "no_time_attrs.nc")
    with netCDF4.Dataset(bad_file, "w") as ds:
        ds.createDimension("y", 2)
        ds.createDimension("x", 2)
        ds.createVariable("latitude", "f8", ("y", "x"))[:, :] = 30.0
        ds.createVariable("longitude", "f8", ("y", "x"))[:, :] = -90.0
        ds.createVariable("refl10cm_max", "f8", ("y", "x"))[:, :] = 20.0

    with pytest.raises(ValueError, match="valid_time"):
        load_model_netcdf(bad_file, varname="refl10cm_max")
    print("\n[geom-check3b] load_model_netcdf correctly raises when init/lead attrs are absent")


# --- Check 4: pixel_area_km2 sanity check -----------------------------------

def test_pixel_area_km2_matches_independent_estimate():
    lat2d, lon2d = _synthetic_mrms_shaped_grid(lat_step=0.05, lon_step=0.05)
    x2d, y2d = build_projected_coords(lat2d, lon2d)
    area = pixel_area_km2(x2d, y2d)

    # independent reference: cos(lat)-weighted approximate cell area for a
    # regular lat/lon grid (deliberately not reusing any pyproj/LCC machinery)
    dlat_km = 0.05 * 111.0
    dlon_km = 0.05 * 111.0 * np.cos(np.deg2rad(lat2d))
    reference_area = dlat_km * dlon_km

    # compare in the interior (away from the domain edges, where both the LCC
    # projection and the gradient-based area estimate are least distorted)
    interior = np.s_[200:-200, 200:-200]
    rel_diff = np.abs(area[interior] - reference_area[interior]) / reference_area[interior]
    print(f"\n[geom-check4] median relative difference vs. independent cos(lat) estimate "
          f"(interior only): {np.median(rel_diff):.3f}")
    assert np.median(rel_diff) < 0.1
