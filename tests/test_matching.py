"""Step 5 validation: object matching (Total Interest score, global greedy
assignment), plus match-file I/O and the manifest-driven pipeline.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_matching.py -v -s
"""

import glob
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from python_obj.obj_core import (
    MatchResult,
    SeriesEntry,
    identify_objects,
    match_objects_one_timestep,
    precompute_grid_geometry,
    read_match_file,
    run_matching_series,
    run_object_id_series,
    total_interest,
    total_interest_area_ratio,
    write_match_file,
)
from python_obj.regrid import load_mrms_netcdf, load_model_netcdf

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")


def _synthetic_grid(ny=100, nx=100, lat0=30.0, lon0=-100.0, step=0.01):
    lat = lat0 + np.arange(ny) * step
    lon = lon0 + np.arange(nx) * step
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lat2d, lon2d


def _blob(cy, cx, r=2, val=50.0, shape=(100, 100)):
    d = np.zeros(shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    d[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = val
    return d


# --- Check 1: TI formula correctness, synthetic -----------------------------

def test_total_interest_formula_correctness():
    # two point-objects (single-pixel coords) at known distances on an
    # isotropic ~1km/gridpoint grid (small lat/lon step keeps km-per-degree
    # effectively constant over this tiny synthetic domain)
    cent1 = (0.0, 0.0)
    cent2 = (20.0, 0.0)  # 20 km away
    coords1 = np.array([[0.0, 0.0]])
    coords2 = np.array([[20.0, 0.0]])

    ti = total_interest(cent1, coords1, cent2, coords2, max_centroid_km=40.0, max_boundary_km=40.0)
    # cent_ti = (40-20)/40 = 0.5; bound_ti = (40-20)/40 = 0.5 (single-point "boundary" == centroid here)
    expected = 0.5 * (0.5 + 0.5)
    print(f"\n[match-check1] ti at 20km separation: {ti:.4f} (expected {expected:.4f})")
    assert abs(ti - expected) < 1e-9

    # beyond max_km -> both terms clip to 0
    cent3 = (100.0, 0.0)
    coords3 = np.array([[100.0, 0.0]])
    ti_far = total_interest(cent1, coords1, cent3, coords3, max_centroid_km=40.0, max_boundary_km=40.0)
    print(f"[match-check1] ti at 100km separation (beyond max): {ti_far:.4f} (expected 0.0)")
    assert ti_far == 0.0

    area_ratio_ti = total_interest_area_ratio(ti=0.8, area1_km2=50.0, area2_km2=100.0)
    print(f"[match-check1] area-weighted ti: {area_ratio_ti:.4f} (expected {0.8*0.5:.4f})")
    assert abs(area_ratio_ti - 0.4) < 1e-9


# --- Check 2: global greedy assignment correctness, all 5 categories -------

def test_match_objects_one_timestep_all_categories():
    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)

    # T1 hits F1; T2 isolated -> miss; T3a & T3b both compete for F3 (T3b closer, wins)
    truth_data = np.maximum.reduce([
        _blob(10, 10), _blob(80, 80), _blob(50, 50), _blob(50, 62),
    ])
    truth_labels, truth_objects = identify_objects(truth_data, gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)

    # F1 hits T1; F2 isolated -> false alarm; F3 contested target (closer to T3b)
    forecast_data = np.maximum.reduce([_blob(11, 11), _blob(20, 80), _blob(50, 58)])
    forecast_labels, forecast_objects = identify_objects(forecast_data, gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)

    records = match_objects_one_timestep(
        truth_objects, truth_labels, forecast_objects, forecast_labels, gg,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2,
    )
    by_category = {}
    for r in records:
        by_category.setdefault(r.category, []).append(r)
        print(f"[match-check2] {r.category:15s} truth_id={r.truth_id:3d} forecast_id={r.forecast_id:3d} ti={r.ti_score:.3f}")

    assert len(by_category.get("hit", [])) == 2
    assert len(by_category.get("miss", [])) == 1
    assert len(by_category.get("false_alarm", [])) == 1
    assert len(by_category.get("truth_extra", [])) == 1
    assert "forecast_extra" not in by_category

    # the T3a/T3b contest: T3b (closer to F3) must be the hit, T3a the truth_extra,
    # and T3a's recorded (informational) partner must be F3
    hit_ids = {r.truth_id for r in by_category["hit"]}
    extra = by_category["truth_extra"][0]
    truth_id_at_50_50 = next(o.id for o in truth_objects if o.centroid_rowcol == (50.0, 50.0))
    truth_id_at_50_62 = next(o.id for o in truth_objects if o.centroid_rowcol == (50.0, 62.0))
    assert truth_id_at_50_62 in hit_ids
    assert extra.truth_id == truth_id_at_50_50
    assert extra.forecast_id != -1  # recorded its near-miss candidate, not -1


# --- Check 3: real end-to-end run -------------------------------------------

def test_real_end_to_end_matching(tmp_path):
    """Generates real truth (MRMS) and forecast (MPAS) object files inline
    from the bundled sample_data/mpas_case/ (3 real hourly times each), rather
    than depending on precomputed test_obj/ output from an earlier session --
    keeps this test self-contained (no data outside python_obj/)."""
    mrms_files = sorted(glob.glob(os.path.join(SAMPLE_DATA_DIR, "mpas_case/interp_mrms/20230501/*.nc")))
    mpas_files = sorted(glob.glob(os.path.join(SAMPLE_DATA_DIR, "mpas_case/mpas_mem1/interp_mpas_3km_2023050100_mem1_f00[1-3].nc")))
    assert len(mrms_files) == 3 and len(mpas_files) == 3

    mrms_loader = lambda fp: load_mrms_netcdf(fp)
    truth_manifest = [SeriesEntry(valid_time=mrms_loader(f).valid_time, filepath=f, member_id=None) for f in mrms_files]
    truth_files = run_object_id_series(
        truth_manifest, lambda entry: mrms_loader(entry.filepath),
        thresh_1=40.0, thresh_2=45.0, area_thresh_km2=108.0,
        output_dir=str(tmp_path / "truth_obj"), file_grouping="single", track_in_time=True, track_bound_disp_km=0.0,
    )

    mpas_loader = lambda fp: load_model_netcdf(fp, varname="refl10cm_max", lat_name="latitude", lon_name="longitude",
                                                init_attr="initializationTime", lead_attr="forecastHour", init_format="%Y%m%d%H")
    forecast_manifest = [SeriesEntry(valid_time=mpas_loader(f).valid_time, filepath=f, member_id=None) for f in mpas_files]
    forecast_files = run_object_id_series(
        forecast_manifest, lambda entry: mpas_loader(entry.filepath),
        thresh_1=35.0, thresh_2=40.0, area_thresh_km2=108.0,
        output_dir=str(tmp_path / "forecast_obj"), file_grouping="single", track_in_time=False,
        init_time=datetime(2023, 5, 1, 0, 0, 0),
    )

    summary = run_matching_series(
        truth_files, forecast_files,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2,
        output_dir=str(tmp_path / "matches"), max_time_offset_minutes=5.0,
    )
    print(f"\n[match-check3] {len(summary.output_paths)} match files, "
          f"{len(summary.skipped_forecast_times)} skipped forecast times")
    assert len(summary.output_paths) == 3
    assert summary.skipped_forecast_times == []

    from collections import Counter
    total = Counter()
    for p in summary.output_paths:
        c = read_match_file(p)
        total.update(r.category for r in c.records)
    print(f"[match-check3] category totals across all times: {dict(total)}")
    # sanity: every category should be representable across real, imperfect
    # data, and there should be no category outside the known five
    assert set(total.keys()) <= {"hit", "miss", "false_alarm", "truth_extra", "forecast_extra"}
    assert total["hit"] > 0


# --- Check 4: time-tolerance behavior ---------------------------------------

def test_time_tolerance_skip_and_match(tmp_path):
    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)

    from python_obj.obj_core import IdentificationResult, SeriesEntry, write_object_file

    data = _blob(50, 50)
    labels, objects = identify_objects(data, gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)

    t0 = datetime(2023, 5, 1, 0, 0, 0)
    truth_path = str(tmp_path / "truth.nc")
    write_object_file(
        truth_path, t0, lat2d, lon2d,
        [IdentificationResult(labels=labels, objects=objects, valid_time=t0 + timedelta(minutes=2), member_id=None)],
        ["x"], 20.0, 30.0, 1.0,
    )

    # forecast A: within 5 min of truth (offset 2 min) -> should match
    forecast_a_path = str(tmp_path / "forecast_a.nc")
    write_object_file(
        forecast_a_path, t0, lat2d, lon2d,
        [IdentificationResult(labels=labels, objects=objects, valid_time=t0, member_id=None)],
        ["y"], 20.0, 30.0, 1.0,
    )

    # forecast B: 10 min offset from the only truth time -> should be skipped
    forecast_b_path = str(tmp_path / "forecast_b.nc")
    write_object_file(
        forecast_b_path, t0, lat2d, lon2d,
        [IdentificationResult(labels=labels, objects=objects, valid_time=t0 + timedelta(minutes=12), member_id=None)],
        ["y"], 20.0, 30.0, 1.0,
    )

    summary = run_matching_series(
        [truth_path], [forecast_a_path, forecast_b_path],
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2,
        output_dir=str(tmp_path / "matches"), max_time_offset_minutes=5.0,
    )
    print(f"\n[match-check4] output files: {len(summary.output_paths)}, skipped: {summary.skipped_forecast_times}")
    assert len(summary.output_paths) == 1
    assert summary.skipped_forecast_times == [t0 + timedelta(minutes=12)]


# --- Check 5: match file round-trip ------------------------------------------

def test_match_file_roundtrip(tmp_path):
    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    truth_data = np.maximum.reduce([_blob(10, 10), _blob(80, 80)])
    truth_labels, truth_objects = identify_objects(truth_data, gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    forecast_data = _blob(11, 11)
    forecast_labels, forecast_objects = identify_objects(forecast_data, gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)

    records = match_objects_one_timestep(
        truth_objects, truth_labels, forecast_objects, forecast_labels, gg, 40.0, 40.0, 0.2,
    )
    t0 = datetime(2023, 5, 1, 0, 0, 0)
    result = MatchResult(records=records, valid_time=t0, member_id=None)

    p = str(tmp_path / "match.nc")
    write_match_file(p, [result], ["truth.nc"], ["forecast.nc"], 40.0, 40.0, 0.2, max_time_offset_minutes=5.0)
    contents = read_match_file(p)

    print(f"\n[match-check5] n records original={len(records)} roundtrip={len(contents.records)}")
    assert len(contents.records) == len(records)
    for orig, rt in zip(records, contents.records):
        assert orig.category == rt.category
        assert orig.truth_id == rt.truth_id
        assert orig.forecast_id == rt.forecast_id
        assert abs(orig.ti_score - rt.ti_score) < 1e-9
    assert contents.max_time_offset_minutes == 5.0


# --- Check 6: performance sanity check ---------------------------------------

def test_matching_performance_with_large_objects():
    """Objects with ~1000 pixels each (MCS-scale, per the tracking.py
    performance bug precedent) shouldn't blow up wall-clock time -- coords_km
    is cached once per object, not recomputed per pair."""
    lat2d, lon2d = _synthetic_grid(ny=400, nx=400)
    gg = precompute_grid_geometry(lat2d, lon2d)

    def big_blob(cy, cx, r=18):
        return _blob(cy, cx, r=r, shape=(400, 400))

    truth_data = np.maximum.reduce([big_blob(50 + 40 * i, 50 + 40 * j) for i in range(4) for j in range(4)])
    forecast_data = np.maximum.reduce([big_blob(52 + 40 * i, 52 + 40 * j) for i in range(4) for j in range(4)])

    truth_labels, truth_objects = identify_objects(truth_data, gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    forecast_labels, forecast_objects = identify_objects(forecast_data, gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    print(f"\n[match-check6] {len(truth_objects)} truth objects, {len(forecast_objects)} forecast objects, "
          f"~{truth_objects[0].area_px} px each")

    t0 = time.time()
    records = match_objects_one_timestep(
        truth_objects, truth_labels, forecast_objects, forecast_labels, gg, 40.0, 40.0, 0.2,
    )
    elapsed = time.time() - t0
    print(f"[match-check6] {len(truth_objects)}x{len(forecast_objects)} pairs, {len(records)} records, {elapsed:.2f}s")
    assert elapsed < 10.0, f"matching took {elapsed:.2f}s -- expected coords_km caching to keep this fast"
