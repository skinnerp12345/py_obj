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


def test_load_mrms_netcdf_falls_back_to_variable_level_valid_time(tmp_path):
    """Real-world case: MET-tool-produced MRMS files (e.g. wofs_MRMS_RAD_*.nc)
    carry valid_time as an attribute on the data variable itself, not as a
    global attribute -- confirmed via direct inspection of a real such file."""
    import netCDF4
    var_level_file = str(tmp_path / "var_level_valid_time.nc")
    with netCDF4.Dataset(var_level_file, "w") as ds:
        ds.createDimension("y", 2)
        ds.createDimension("x", 2)
        ds.createVariable("lat", "f8", ("y", "x"))[:, :] = 30.0
        ds.createVariable("lon", "f8", ("y", "x"))[:, :] = -90.0
        refl = ds.createVariable("refl_consv", "f8", ("y", "x"))
        refl[:, :] = 20.0
        refl.valid_time = "20260501_020000"  # MET-tool convention, variable-level, not global
        # deliberately no global valid_time attribute

    field = load_mrms_netcdf(var_level_file)
    assert field.valid_time == datetime(2026, 5, 1, 2, 0, 0)
    print(f"\n[geom-check2c] load_mrms_netcdf falls back to the data variable's own "
          f"valid_time attribute when no global one exists: {field.valid_time}")


def test_load_mrms_netcdf_prefers_global_valid_time_over_variable_level(tmp_path):
    """When both are present, the global attribute (this library's own
    interpolated-output convention) wins -- confirms the fallback is only a
    fallback, not a silent override of existing, already-correct behavior."""
    import netCDF4
    both_file = str(tmp_path / "both_valid_time.nc")
    with netCDF4.Dataset(both_file, "w") as ds:
        ds.createDimension("y", 2)
        ds.createDimension("x", 2)
        ds.createVariable("lat", "f8", ("y", "x"))[:, :] = 30.0
        ds.createVariable("lon", "f8", ("y", "x"))[:, :] = -90.0
        refl = ds.createVariable("refl_consv", "f8", ("y", "x"))
        refl[:, :] = 20.0
        refl.valid_time = "20260501_020000"       # variable-level -- should be ignored
        ds.valid_time = "2023-05-01T01:00:41"      # global -- should win

    field = load_mrms_netcdf(both_file)
    assert field.valid_time == datetime(2023, 5, 1, 1, 0, 41)
    print(f"\n[geom-check2d] load_mrms_netcdf prefers the global valid_time attribute "
          f"when both are present: {field.valid_time}")


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


# --- Check 3c: valid_time_var -- CF-convention time coordinate override -----
#
# Real bug this guards against: a real WoFSCast output file's global
# init_time/valid_time attributes were both identical, stale copies of the
# WoFS input file it was derived from (confirmed via direct inspection) --
# the real valid_time only existed in a proper CF-convention time coordinate
# variable (units="<unit> since <reference>" + a numeric offset), cross-
# confirmed against the file's own embedded filename timestamp.

def test_load_model_netcdf_valid_time_var_decodes_cf_time_coordinate(tmp_path):
    import netCDF4
    cf_file = str(tmp_path / "cf_time.nc")
    with netCDF4.Dataset(cf_file, "w") as ds:
        ds.createDimension("y", 2)
        ds.createDimension("x", 2)
        ds.createVariable("latitude", "f8", ("y", "x"))[:, :] = 30.0
        ds.createVariable("longitude", "f8", ("y", "x"))[:, :] = -90.0
        ds.createVariable("refl10cm_max", "f8", ("y", "x"))[:, :] = 20.0
        t = ds.createVariable("datetime", "i8", ())
        t.units = "hours since 2023-01-01T00:00:00"
        t.calendar = "proleptic_gregorian"
        t[...] = 5  # -> 2023-01-01T05:00:00, a non-trivial, non-zero offset
        # deliberately WRONG global attrs, mirroring the real WoFSCast bug --
        # valid_time_var must win over these, not just work in their absence
        ds.init_time = "20230101_000000"
        ds.valid_time = "20230101_000000"

    field = load_model_netcdf(
        cf_file, varname="refl10cm_max", lat_name="latitude", lon_name="longitude",
        valid_time_var="datetime",
    )
    assert field.valid_time == datetime(2023, 1, 1, 5, 0, 0)
    print(f"\n[geom-check3c] valid_time_var correctly decoded {field.valid_time} from the CF time "
          f"coordinate, overriding the deliberately-wrong global attrs (would have given 2023-01-01 00:00)")


def test_load_model_netcdf_valid_time_var_real_wofscast_file():
    """Real end-to-end confirmation against the actual reported-erroneous
    WoFSCast file, if present locally (external data, not bundled)."""
    real_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "sfe_poster", "wofs_wofscast_WEB_62_20260506_2220_0310.nc",
    )
    if not os.path.exists(real_file):
        pytest.skip(f"real external WoFSCast sample not present at {real_file}")

    buggy = load_model_netcdf(
        real_file, varname="wofscast_comp_dz", lat_name="xlat", lon_name="xlon",
        valid_time_attr="valid_time", valid_time_format="%Y%m%d_%H%M%S", extra_dim_index=0,
    )
    fixed = load_model_netcdf(
        real_file, varname="wofscast_comp_dz", lat_name="xlat", lon_name="xlon",
        valid_time_var="datetime", extra_dim_index=0,
    )
    print(f"\n[geom-check3d] real WoFSCast file: buggy (global attr) valid_time={buggy.valid_time}, "
          f"fixed (CF var) valid_time={fixed.valid_time}")
    assert buggy.valid_time == datetime(2026, 5, 6, 22, 0, 0)  # the confirmed-wrong value
    assert fixed.valid_time == datetime(2026, 5, 7, 3, 10, 0)  # matches the filename's embedded "0310"


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
