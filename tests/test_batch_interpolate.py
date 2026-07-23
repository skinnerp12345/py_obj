"""Step 1b validation: batch MRMS interpolation with multiprocessing.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_batch_interpolate.py -v -s

Uses the bundled python_obj/sample_data/ (3 real MRMS files, one small real
MPAS target grid) for a fast, repeatable, self-contained check; see CLAUDE.md
/ the plan file for the larger 10-day run used to validate this at realistic
scale against the full local test_mrms/ (too slow, and too large to bundle,
for a routine pytest run).
"""

import glob
import os
import shutil

import netCDF4
import numpy as np
import pytest

from python_obj.regrid import discover_mrms_files, load_mrms_grib2, make_output_path, run_batch_interpolation
from python_obj.regrid.grid_spec import GridSpec, crop_to_bbox
from python_obj.regrid.io_grid import load_target_grid
from python_obj.regrid.io_mrms import MRMS_MISSING_VALUE, clip_near_zero_sentinel
from python_obj.regrid.regridder import build_conservative_regridder, regrid_field

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
INPUT_DIR = os.path.join(SAMPLE_DATA_DIR, "mpas_case", "mrms")
MPAS_FILE = os.path.join(SAMPLE_DATA_DIR, "mpas_case/mpas_mem1/interp_mpas_3km_2023050100_mem1_f001.nc")
WEIGHT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_weight_cache")


def test_discover_mrms_files_finds_nested_and_filters_by_date():
    files = discover_mrms_files(INPUT_DIR)
    assert len(files) == 3, "expected the bundled 3-file sample_data/mpas_case/mrms/20230501 set to be discovered"
    assert all(f.endswith(".grib2.gz") for f in files)
    print(f"\n[batch-check1] discovered {len(files)} files")


# --- file_pattern: restricting discovery to one MRMS product ---------------
#
# Real risk this guards against: load_mrms_grib2() has no product-type
# awareness of its own (it just reads whichever GRIB2 message is first in the
# file) -- an unfiltered glob sweeping up MESH/RotationTrack files sitting
# alongside MergedReflectivityQCComposite in the same date directory would
# silently interpolate and write out non-reflectivity values as if they were
# dBZ. Uses empty placeholder files for the other-product decoys -- this test
# only needs to prove filename-based discovery excludes them, never reads
# their content.

def _mixed_product_dir(tmp_path) -> str:
    mixed_dir = tmp_path / "mixed_products" / "20230501"
    mixed_dir.mkdir(parents=True)
    for real_file in discover_mrms_files(INPUT_DIR):
        shutil.copy(real_file, mixed_dir / os.path.basename(real_file))
    (mixed_dir / "MRMS_MESH_00.50_20230501-000000.grib2.gz").touch()
    (mixed_dir / "MRMS_RotationTrack60min_00.50_20230501-000000.grib2.gz").touch()
    return str(tmp_path / "mixed_products")


def test_discover_mrms_files_default_pattern_sweeps_up_other_products(tmp_path):
    mixed_dir = _mixed_product_dir(tmp_path)
    files = discover_mrms_files(mixed_dir)  # default pattern: "**/*.grib2*"
    print(f"\n[batch-check-pattern1] default pattern discovered {len(files)} files (expect 5: 3 real + 2 decoys)")
    assert len(files) == 5, "default pattern should match every grib2 file regardless of product -- demonstrates the risk"


def test_discover_mrms_files_product_pattern_excludes_other_products(tmp_path):
    mixed_dir = _mixed_product_dir(tmp_path)
    files = discover_mrms_files(mixed_dir, pattern="**/*MergedReflectivityQCComposite*")
    print(f"\n[batch-check-pattern2] product-specific pattern discovered {len(files)} files (expect 3, decoys excluded)")
    assert len(files) == 3
    assert all("MergedReflectivityQCComposite" in f for f in files)


def test_run_batch_interpolation_file_pattern_excludes_other_products(tmp_path):
    mixed_dir = _mixed_product_dir(tmp_path)
    out_dir = str(tmp_path / "interp_mixed_test")

    summary = run_batch_interpolation(
        input_dir=mixed_dir,
        output_dir=out_dir,
        target_grid_file=MPAS_FILE,
        target_lat_name="latitude",
        target_lon_name="longitude",
        weight_cache_dir=WEIGHT_CACHE_DIR,
        n_workers=2,
        file_pattern="**/*MergedReflectivityQCComposite*",
    )
    print(f"\n[batch-check-pattern3] {summary}")
    assert summary.n_total == 3, "file_pattern should have excluded the 2 decoy other-product files before processing"
    assert summary.n_failed == 0


def test_make_output_path_naming_convention():
    from datetime import datetime
    vt = datetime(2023, 4, 1, 1, 0, 41)
    path = make_output_path("/some/input/dir/whatever.grib2.gz", vt, "/out")
    assert path == "/out/20230401/interp_mrms_20230401_010041.nc"
    print(f"\n[batch-check2] output path: {path}")


def test_batch_interpolation_small_subset(tmp_path):
    out_dir = str(tmp_path / "interp_mrms_test")

    summary = run_batch_interpolation(
        input_dir=INPUT_DIR,
        output_dir=out_dir,
        target_grid_file=MPAS_FILE,
        target_lat_name="latitude",
        target_lon_name="longitude",
        weight_cache_dir=WEIGHT_CACHE_DIR,
        n_workers=2,
        date_range=("20230501", "20230501"),
        max_files=3,
    )

    assert summary.n_total == 3
    assert summary.n_failed == 0, f"expected no failures, got: {summary.failures}"

    written = sorted(glob.glob(os.path.join(out_dir, "20230501", "*.nc")))
    assert len(written) == 3

    with netCDF4.Dataset(written[0]) as ds:
        assert set(["lat", "lon", "refl_consv"]).issubset(set(ds.variables.keys()))
        assert ds.variables["refl_consv"].shape == (250, 250)
        assert hasattr(ds, "valid_time")
        assert hasattr(ds, "source_file")
        assert ds.variables["refl_consv"]._FillValue == MRMS_MISSING_VALUE
        data = ds.variables["refl_consv"][:]
        # no unremarked NaNs -- coverage gaps are filled with the documented sentinel
        assert not np.any(np.isnan(data))

    print(f"\n[batch-check3] {summary}")


def test_batch_output_matches_step1_inmemory_regrid(tmp_path):
    """Spot-check: one batch-produced file's data should exactly match Step 1's
    already-validated in-memory regrid of the same source file."""
    out_dir = str(tmp_path / "interp_mrms_crosscheck")

    summary = run_batch_interpolation(
        input_dir=INPUT_DIR,
        output_dir=out_dir,
        target_grid_file=MPAS_FILE,
        target_lat_name="latitude",
        target_lon_name="longitude",
        weight_cache_dir=WEIGHT_CACHE_DIR,
        n_workers=1,
        date_range=("20230501", "20230501"),
        max_files=1,
    )
    assert summary.n_failed == 0
    batch_file = summary.n_success and glob.glob(os.path.join(out_dir, "20230501", "*.nc"))[0]

    with netCDF4.Dataset(batch_file) as ds:
        batch_data = np.asarray(ds.variables["refl_consv"][:])

    # independently reproduce the same regrid via the Step 1 in-memory API
    input_file = discover_mrms_files(INPUT_DIR)
    input_file = [f for f in input_file if os.path.basename(os.path.dirname(f)) == "20230501"][0]

    mrms = load_mrms_grib2(input_file)
    data = clip_near_zero_sentinel(mrms.data)
    tgt_grid = load_target_grid(MPAS_FILE, "latitude", "longitude")
    src_grid_full = GridSpec(lat2d=mrms.lat2d, lon2d=mrms.lon2d)
    bbox = (tgt_grid.lat2d.min(), tgt_grid.lat2d.max(), tgt_grid.lon2d.min(), tgt_grid.lon2d.max())
    cropped_grid, cropped_data = crop_to_bbox(src_grid_full, data, *bbox, buffer_deg=0.3)
    regridder = build_conservative_regridder(cropped_grid, tgt_grid, weight_cache_dir=WEIGHT_CACHE_DIR)
    expected, _ = regrid_field(regridder, cropped_data, fill_value=MRMS_MISSING_VALUE, missing_value=MRMS_MISSING_VALUE)

    np.testing.assert_allclose(batch_data, expected)
    print("\n[batch-check4] batch output matches independent in-memory regrid exactly")
