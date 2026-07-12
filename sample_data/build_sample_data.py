"""Builds python_obj/sample_data/ from real local test data (this developer's
own copies of test_mrms/, test_mpas/, test_wofs/, and the already-fetched
python_obj/configs/output/fetched_mrms_wofs/), trimming/cropping each source
file down to just the variables and global attributes this library actually
reads (load_model_netcdf()'s varname/lat_name/lon_name + its time-derivation
attrs). The trim is what makes the result small enough to commit to git --
the source MPAS/WoFS files are 234-278 MB each because they carry dozens of
unused WRF diagnostic variables; the library never reads any of those.

MPAS is also spatially cropped to a fixed 250x250-pixel window (the same
window across all 3 lead times) chosen empirically to contain a real,
evolving storm cluster (confirmed: 7.65% / 14.48% / 11.14% of the window
>=20 dBZ across f001/f002/f003) -- the full 1059x1799 grid mostly has no
matching storm-scale purpose bundled at this size. WoFS is NOT cropped
(its native domain is already small, 300x300) and keeps all NE=18 members
intact, since per-member identification is a real feature this sample needs
to demonstrate.

Idempotent -- safe to re-run; overwrites its own output. Not run automatically
by anything else; this is a provenance/regeneration record, not part of the
library's own runtime.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/sample_data/build_sample_data.py
"""

import os
import shutil
import sys
import tempfile

import netCDF4
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, REPO_ROOT)

from python_obj.regrid import run_batch_interpolation  # noqa: E402 -- needs REPO_ROOT on sys.path first

# ---- MPAS case: 3 real lead times with genuine storm signal (f000 has none) ----
MPAS_SRC_DIR = os.path.join(REPO_ROOT, "test_mpas", "mem1")
MPAS_LEADS = ["f001", "f002", "f003"]
# Upper Midwest / Great Lakes region (lat ~41.5-49.2N, lon ~-86.5 to -75.4W):
# a real, evolving storm cluster (3.9% / 33.1% / 39.1% of the window >=20 dBZ
# across f001/f002/f003) AND fully within MRMS's real radar coverage (unlike
# an earlier candidate crop near the Atlantic coast, confirmed empirically to
# have ~45% no-coverage cells offshore -- this region's corresponding MRMS
# footprint has effectively 0% missing coverage).
MPAS_ROW_SLICE = slice(700, 950)   # 250 px, ~750 km at 3-km grid spacing
MPAS_COL_SLICE = slice(1200, 1450)
MRMS_MPAS_SRC_DIR = os.path.join(REPO_ROOT, "test_mrms", "20230501")
MRMS_MPAS_FILES = [
    "MRMS_MergedReflectivityQCComposite_00.50_20230501-010041.grib2.gz",
    "MRMS_MergedReflectivityQCComposite_00.50_20230501-020038.grib2.gz",
    "MRMS_MergedReflectivityQCComposite_00.50_20230501-030035.grib2.gz",
]

# f048 is bundled separately, full-domain (NOT cropped): its real >=50.2 dBZ
# cells (the ones that make identify_objects's linear-classification produce
# all 3 categories -- 11 cellular/1 mixed/1 linear, the empirical scan
# documented in test_linear_classification.py) are scattered across nearly
# the entire CONUS domain (row 166-1050, col 138-1796), so no single small
# crop window could preserve that real result. Trimming to just the 3 needed
# variables (no unused WRF diagnostics) still shrinks it by >97% (280 MB -> 8.8 MB).
MPAS_LINEAR_CHECK_LEAD = "f048"

# ---- WoFS case: 3 consecutive real 5-min lead times, all 18 members kept ----
WOFS_SRC_DIR = os.path.join(REPO_ROOT, "test_wofs")
WOFS_FILES = [
    "wofs_ALL_00_20260518_2300_2300.nc",
    "wofs_ALL_01_20260518_2300_2305.nc",
    "wofs_ALL_02_20260518_2300_2310.nc",
]
MRMS_WOFS_SRC_DIR = os.path.join(
    REPO_ROOT, "python_obj", "configs", "output", "fetched_mrms_wofs", "20260518"
)
MRMS_WOFS_FILES = [
    "MRMS_MergedReflectivityQCComposite_00.50_20260518-230039.grib2.gz",
    "MRMS_MergedReflectivityQCComposite_00.50_20260518-230441.grib2.gz",
    "MRMS_MergedReflectivityQCComposite_00.50_20260518-231040.grib2.gz",
]


def trim_mpas_file(src_path: str, dst_path: str, row_slice: slice = slice(None), col_slice: slice = slice(None)) -> None:
    with netCDF4.Dataset(src_path) as src:
        refl = np.asarray(src.variables["refl10cm_max"][0, row_slice, col_slice], dtype=np.float32)
        lat = np.asarray(src.variables["latitude"][0, row_slice, col_slice], dtype=np.float32)
        lon = np.asarray(src.variables["longitude"][0, row_slice, col_slice], dtype=np.float32)
        init_time = src.initializationTime
        forecast_hour = src.forecastHour

    ny, nx = refl.shape
    with netCDF4.Dataset(dst_path, "w") as dst:
        dst.createDimension("time", 1)
        dst.createDimension("lat", ny)
        dst.createDimension("lon", nx)
        for name, data in (("refl10cm_max", refl), ("latitude", lat), ("longitude", lon)):
            var = dst.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True)
            var[0] = data
        dst.initializationTime = init_time
        dst.forecastHour = forecast_hour


def trim_wofs_file(src_path: str, dst_path: str) -> None:
    with netCDF4.Dataset(src_path) as src:
        comp_dz = np.asarray(src.variables["comp_dz"][:], dtype=np.float32)
        xlat = np.asarray(src.variables["xlat"][:], dtype=np.float32)
        xlon = np.asarray(src.variables["xlon"][:], dtype=np.float32)
        valid_time = src.valid_time
        init_time = src.init_time

    ne, ny, nx = comp_dz.shape
    with netCDF4.Dataset(dst_path, "w") as dst:
        dst.createDimension("NE", ne)
        dst.createDimension("NY", ny)
        dst.createDimension("NX", nx)
        var = dst.createVariable("comp_dz", "f4", ("NE", "NY", "NX"), zlib=True)
        var[:] = comp_dz
        for name, data in (("xlat", xlat), ("xlon", xlon)):
            v = dst.createVariable(name, "f4", ("NY", "NX"), zlib=True)
            v[:] = data
        dst.valid_time = valid_time
        dst.init_time = init_time


def _report(src_path: str, dst_path: str) -> None:
    src_mb = os.path.getsize(src_path) / 1e6
    dst_mb = os.path.getsize(dst_path) / 1e6
    print(f"wrote {dst_path} ({dst_mb:.2f} MB, was {src_mb:.1f} MB)")


def main() -> None:
    mpas_out = os.path.join(_THIS_DIR, "mpas_case", "mpas_mem1")
    mrms_mpas_out = os.path.join(_THIS_DIR, "mpas_case", "mrms", "20230501")
    wofs_out = os.path.join(_THIS_DIR, "wofs_case", "wofs")
    mrms_wofs_out = os.path.join(_THIS_DIR, "wofs_case", "mrms", "20260518")
    for d in (mpas_out, mrms_mpas_out, wofs_out, mrms_wofs_out):
        os.makedirs(d, exist_ok=True)

    for lead in MPAS_LEADS:
        fname = f"interp_mpas_3km_2023050100_mem1_{lead}.nc"
        src, dst = os.path.join(MPAS_SRC_DIR, fname), os.path.join(mpas_out, fname)
        trim_mpas_file(src, dst, MPAS_ROW_SLICE, MPAS_COL_SLICE)
        _report(src, dst)

    # full-domain (uncropped) f048, for test_linear_classification.py's
    # real-data check only -- see MPAS_LINEAR_CHECK_LEAD's comment above
    fname = f"interp_mpas_3km_2023050100_mem1_{MPAS_LINEAR_CHECK_LEAD}.nc"
    src, dst = os.path.join(MPAS_SRC_DIR, fname), os.path.join(mpas_out, fname)
    trim_mpas_file(src, dst)
    _report(src, dst)

    for fname in MRMS_MPAS_FILES:
        src, dst = os.path.join(MRMS_MPAS_SRC_DIR, fname), os.path.join(mrms_mpas_out, fname)
        shutil.copy(src, dst)
        print(f"copied {dst}")

    # Real interpolated-MRMS-onto-MPAS-grid output, produced via the actual
    # library function (not hand-rolled) -- dogfoods the real pipeline rather
    # than shipping a hand-crafted fixture. Needed by test_geometry.py's
    # load_mrms_netcdf check and test_identify.py's obs-tracking-series check,
    # both of which previously depended on the separate, non-bundled
    # interp_mrms/ directory from an earlier session.
    interp_mrms_out = os.path.join(_THIS_DIR, "mpas_case", "interp_mrms")
    target_grid_file = os.path.join(mpas_out, "interp_mpas_3km_2023050100_mem1_f001.nc")
    with tempfile.TemporaryDirectory() as weight_cache_dir:
        summary = run_batch_interpolation(
            input_dir=os.path.dirname(mrms_mpas_out),
            output_dir=interp_mrms_out,
            target_grid_file=target_grid_file,
            target_lat_name="latitude",
            target_lon_name="longitude",
            weight_cache_dir=weight_cache_dir,
            n_workers=1,
        )
        print(f"interpolated MRMS onto MPAS grid: {summary}")

    for fname in WOFS_FILES:
        src, dst = os.path.join(WOFS_SRC_DIR, fname), os.path.join(wofs_out, fname)
        trim_wofs_file(src, dst)
        _report(src, dst)

    for fname in MRMS_WOFS_FILES:
        src, dst = os.path.join(MRMS_WOFS_SRC_DIR, fname), os.path.join(mrms_wofs_out, fname)
        shutil.copy(src, dst)
        print(f"copied {dst}")


if __name__ == "__main__":
    main()
