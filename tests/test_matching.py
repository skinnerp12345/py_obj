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
        1, 20.0, 30.0, 1.0,
    )

    # forecast A: within 5 min of truth (offset 2 min) -> should match
    forecast_a_path = str(tmp_path / "forecast_a.nc")
    write_object_file(
        forecast_a_path, t0, lat2d, lon2d,
        [IdentificationResult(labels=labels, objects=objects, valid_time=t0, member_id=None)],
        1, 20.0, 30.0, 1.0,
    )

    # forecast B: 10 min offset from the only truth time -> should be skipped
    forecast_b_path = str(tmp_path / "forecast_b.nc")
    write_object_file(
        forecast_b_path, t0, lat2d, lon2d,
        [IdentificationResult(labels=labels, objects=objects, valid_time=t0 + timedelta(minutes=12), member_id=None)],
        1, 20.0, 30.0, 1.0,
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
    write_match_file(p, [result], 1, 1, 40.0, 40.0, 0.2, max_time_offset_minutes=5.0)
    contents = read_match_file(p)

    print(f"\n[match-check5] n records original={len(records)} roundtrip={len(contents.records)}")
    assert len(contents.records) == len(records)
    for orig, rt in zip(records, contents.records):
        assert orig.category == rt.category
        assert orig.truth_id == rt.truth_id
        assert orig.forecast_id == rt.forecast_id
        assert abs(orig.ti_score - rt.ti_score) < 1e-9
        assert orig.truth_mean_intensity == rt.truth_mean_intensity
        assert orig.truth_solidity == rt.truth_solidity
        assert orig.truth_major_axis_length == rt.truth_major_axis_length
        assert orig.truth_minor_axis_length == rt.truth_minor_axis_length
        assert orig.truth_eccentricity == rt.truth_eccentricity
    assert contents.max_time_offset_minutes == 5.0
    assert contents.n_truth_source_files == 1
    assert contents.n_forecast_source_files == 1


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


# --- Check 7: iter_object_slices_lazy correctness + memory ------------------

def test_iter_object_slices_lazy_matches_eager_read(tmp_path):
    """iter_object_slices_lazy(path) must yield the exact same
    (member_id, valid_time, labels2d, objects) sequence as
    iter_object_slices(read_object_file(path)) on a real init_snapshot-shaped
    (member AND time dims both present) file -- the shape large ensemble/
    long-forecast cases take, and the one this lazy reader exists for."""
    from python_obj.obj_core import (
        IdentificationResult, iter_object_slices, iter_object_slices_lazy, read_object_file, write_object_file,
    )

    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    t0 = datetime(2023, 5, 1, 0, 0, 0)

    results = []
    for mi, member_id in enumerate(["mem_1", "mem_2", "mem_3"]):
        for ti in range(4):
            data = _blob(20 + 5 * mi, 20 + 5 * ti, r=2)
            labels, objects = identify_objects(data, gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
            results.append(IdentificationResult(
                labels=labels, objects=objects, valid_time=t0 + timedelta(hours=ti), member_id=member_id,
            ))

    path = str(tmp_path / "init_snapshot.nc")
    write_object_file(path, t0, lat2d, lon2d, results, len(results), 20.0, 30.0, 1.0)

    import netCDF4
    with netCDF4.Dataset(path) as ds:
        assert "member" in ds.dimensions and "time" in ds.dimensions

    eager = list(iter_object_slices(read_object_file(path)))
    lazy = list(iter_object_slices_lazy(path))

    assert len(eager) == len(lazy) == 12  # 3 members x 4 times
    for (em, ev, el, eo), (lm, lv, ll, lo) in zip(eager, lazy):
        assert em == lm and ev == lv
        assert np.array_equal(el, ll)
        assert [o.id for o in eo] == [o.id for o in lo]
    print(f"\n[lazy-check1] {len(eager)} slices identical between eager and lazy reads")


def test_iter_object_slices_lazy_peak_memory_bounded(tmp_path):
    """Direct proof of the measured claim motivating this reader: peak memory
    for the lazy, member-by-member read must stay far below what eagerly
    loading the whole (member, time, y, x) labels array would need -- mirrors
    the tracemalloc-based regression test already used for the write_object_file
    streaming fix."""
    import tracemalloc

    from python_obj.obj_core import IdentificationResult, iter_object_slices_lazy, write_object_file

    lat2d, lon2d = _synthetic_grid(ny=300, nx=300)
    gg = precompute_grid_geometry(lat2d, lon2d)
    t0 = datetime(2023, 5, 1, 0, 0, 0)

    n_members, n_times = 10, 20
    results = []
    for mi in range(n_members):
        for ti in range(n_times):
            data = _blob(50, 50, r=2, shape=(300, 300))
            labels, objects = identify_objects(data, gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
            results.append(IdentificationResult(
                labels=labels, objects=objects, valid_time=t0 + timedelta(hours=ti), member_id=f"mem_{mi}",
            ))

    path = str(tmp_path / "lazy_memory.nc")
    write_object_file(path, t0, lat2d, lon2d, results, len(results), 20.0, 30.0, 1.0)

    eager_full_array_bytes = n_members * n_times * 300 * 300 * 4  # what one eager read would materialize
    one_member_bytes = n_times * 300 * 300 * 4

    tracemalloc.start()
    n_slices = 0
    for member_id, vt, labels2d, objects in iter_object_slices_lazy(path):
        n_slices += 1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n[lazy-check2] {n_slices} slices consumed, eager-equivalent array would be "
          f"{eager_full_array_bytes / 1e6:.1f} MB, lazy traced peak: {peak / 1e6:.2f} MB")
    assert n_slices == n_members * n_times
    # the lazy reader should never hold more than roughly one member's worth
    # of decompressed data (n_times x 300 x 300 x 4 bytes) at once, not all
    # n_members at once -- bound against one member's own size (generous 4x
    # margin: netCDF4/HDF5 decompression itself has a measured ~2x transient
    # overhead over the final array size, plus fixed Python/tracemalloc
    # overhead that matters more at this small synthetic grid's scale) rather
    # than a fixed fraction, so the assertion scales with n_members chosen above
    assert peak < one_member_bytes * 4, (
        f"lazy read peak ({peak / 1e6:.2f} MB) is more than 4x one member's own size "
        f"({one_member_bytes / 1e6:.1f} MB) -- expected roughly one member's worth, not more"
    )
    assert peak < eager_full_array_bytes * 0.5, (
        f"lazy read peak ({peak / 1e6:.2f} MB) unexpectedly close to the full "
        f"eager-equivalent size ({eager_full_array_bytes / 1e6:.1f} MB)"
    )


def test_run_matching_series_with_init_snapshot_forecast_matches_per_member_correctly(tmp_path):
    """Real init_snapshot-shaped (member AND time dims both present) forecast
    file matched via run_matching_series -- the shape this project's real
    MPAS batch object files use, and the one iter_object_slices_lazy's
    member-by-member read path exists for. Confirms member bookkeeping
    survives the lazy read: each member's objects must be matched only
    against the correct valid_time's truth, independently per member, with
    no mixing across members or times."""
    from python_obj.obj_core import IdentificationResult, write_object_file

    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)

    t0 = datetime(2023, 5, 1, 0, 0, 0)
    t1 = t0 + timedelta(hours=1)

    truth_data = _blob(50, 50)
    truth_labels, truth_objects = identify_objects(truth_data, gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    truth_path_0 = str(tmp_path / "truth_t0.nc")
    write_object_file(truth_path_0, t0, lat2d, lon2d,
        [IdentificationResult(labels=truth_labels, objects=truth_objects, valid_time=t0, member_id=None)],
        1, 20.0, 30.0, 1.0)
    truth_path_1 = str(tmp_path / "truth_t1.nc")
    write_object_file(truth_path_1, t1, lat2d, lon2d,
        [IdentificationResult(labels=truth_labels, objects=truth_objects, valid_time=t1, member_id=None)],
        1, 20.0, 30.0, 1.0)

    # mem_1: co-located with truth at both times -> should hit both times.
    # mem_2: far from truth at both times -> should false_alarm both times.
    close_labels, close_objects = identify_objects(_blob(50, 50), gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    far_labels, far_objects = identify_objects(_blob(90, 90), gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)

    forecast_path = str(tmp_path / "forecast_init_snapshot.nc")
    write_object_file(forecast_path, t0, lat2d, lon2d, [
        IdentificationResult(labels=close_labels, objects=close_objects, valid_time=t0, member_id="mem_1"),
        IdentificationResult(labels=close_labels, objects=close_objects, valid_time=t1, member_id="mem_1"),
        IdentificationResult(labels=far_labels, objects=far_objects, valid_time=t0, member_id="mem_2"),
        IdentificationResult(labels=far_labels, objects=far_objects, valid_time=t1, member_id="mem_2"),
    ], 1, 20.0, 30.0, 1.0)

    import netCDF4
    with netCDF4.Dataset(forecast_path) as ds:
        assert "member" in ds.dimensions and "time" in ds.dimensions

    summary = run_matching_series(
        [truth_path_0, truth_path_1], [forecast_path],
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2,
        output_dir=str(tmp_path / "matches"), max_time_offset_minutes=5.0,
    )
    print(f"\n[lazy-check3] {len(summary.output_paths)} match files, skipped={summary.skipped_forecast_times}")
    assert len(summary.output_paths) == 2
    assert summary.skipped_forecast_times == []

    for path in summary.output_paths:
        contents = read_match_file(path)
        assert contents.member_ids == ["mem_1", "mem_2"]
        by_member: dict[str, list[str]] = {"mem_1": [], "mem_2": []}
        for rec, mi in zip(contents.records, contents.member_index):
            by_member[contents.member_ids[mi]].append(rec.category)
        print(f"[lazy-check3] {os.path.basename(path)}: {by_member}")
        assert "hit" in by_member["mem_1"] and "false_alarm" not in by_member["mem_1"]
        assert "false_alarm" in by_member["mem_2"] and "hit" not in by_member["mem_2"]


# --- Check 8: file_grouping="init_snapshot" consolidated match output -------

def test_run_matching_series_init_snapshot_consolidates_one_file_per_case(tmp_path):
    """file_grouping="init_snapshot" must produce exactly ONE match file per
    call (mirroring object files' own init_snapshot shape: both member and
    time dimensions in one file), named after the forecast case's own
    init_time -- and its n_truth_source_files/n_forecast_source_files counts
    must reflect the actual number of distinct files used, not the length of
    whatever list was passed in (the bug this replaces: a real match file
    from the per-time driver was found listing all 133 input truth files for
    a single-hour output)."""
    from python_obj.obj_core import IdentificationResult, write_object_file

    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    init_time = datetime(2023, 5, 1, 0, 0, 0)
    t0, t1 = init_time, init_time + timedelta(hours=1)

    truth_labels_0, truth_objects_0 = identify_objects(_blob(50, 50), gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    truth_labels_1, truth_objects_1 = identify_objects(_blob(60, 60), gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    truth_path_0 = str(tmp_path / "truth_t0.nc")
    write_object_file(truth_path_0, t0, lat2d, lon2d,
        [IdentificationResult(labels=truth_labels_0, objects=truth_objects_0, valid_time=t0, member_id=None)],
        1, 20.0, 30.0, 1.0)
    truth_path_1 = str(tmp_path / "truth_t1.nc")
    write_object_file(truth_path_1, t1, lat2d, lon2d,
        [IdentificationResult(labels=truth_labels_1, objects=truth_objects_1, valid_time=t1, member_id=None)],
        1, 20.0, 30.0, 1.0)

    close_labels_0, close_objects_0 = identify_objects(_blob(50, 50), gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    close_labels_1, close_objects_1 = identify_objects(_blob(60, 60), gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    forecast_path = str(tmp_path / "forecast_init_snapshot.nc")
    write_object_file(forecast_path, init_time, lat2d, lon2d, [
        IdentificationResult(labels=close_labels_0, objects=close_objects_0, valid_time=t0, member_id="mem_1"),
        IdentificationResult(labels=close_labels_1, objects=close_objects_1, valid_time=t1, member_id="mem_1"),
        IdentificationResult(labels=close_labels_0, objects=close_objects_0, valid_time=t0, member_id="mem_2"),
        IdentificationResult(labels=close_labels_1, objects=close_objects_1, valid_time=t1, member_id="mem_2"),
    ], 1, 20.0, 30.0, 1.0)

    summary = run_matching_series(
        [truth_path_0, truth_path_1], [forecast_path],
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2,
        output_dir=str(tmp_path / "matches"), max_time_offset_minutes=5.0,
        file_grouping="init_snapshot",
    )
    print(f"\n[lazy-check4] {summary.output_paths}")
    assert len(summary.output_paths) == 1
    assert os.path.basename(summary.output_paths[0]) == f"match_init_{init_time:%Y%m%d_%H%M%S}.nc"

    contents = read_match_file(summary.output_paths[0])
    assert contents.member_ids == ["mem_1", "mem_2"]
    assert len(contents.valid_times) == 2
    # every (member, time) combination's records must be present -- confirms
    # nothing was dropped when consolidating from 4 separate per-(member,time)
    # match computations into one file
    assert len(contents.records) == 4  # 2 members x 2 times, 1 hit record each (1 truth obj, 1 forecast obj)
    print(f"[lazy-check4] n_truth_source_files={contents.n_truth_source_files}, "
          f"n_forecast_source_files={contents.n_forecast_source_files}")
    assert contents.n_truth_source_files == 2  # both truth_path_0 and truth_path_1 were actually used
    assert contents.n_forecast_source_files == 1  # only the one forecast file


def test_run_matching_series_init_snapshot_with_jittered_truth_times(tmp_path):
    """Regression test for a real bug caught running this against actual
    MRMS data: truth files' own real valid_time is jittered (e.g. 00:00:41,
    not exactly 00:00:00) relative to the forecast's clean on-the-hour
    valid_time, matched via max_time_offset_minutes tolerance -- not exact
    equality. n_truth_source_files must still resolve correctly (previously
    raised KeyError because the fix looked up the truth file by the
    FORECAST's valid_time instead of the actual matched truth valid_time)."""
    from python_obj.obj_core import IdentificationResult, write_object_file

    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    init_time = datetime(2023, 5, 1, 0, 0, 0)
    forecast_t0 = init_time  # exactly on the hour
    truth_t0 = init_time + timedelta(seconds=41)  # jittered, within the 5-min tolerance

    truth_labels, truth_objects = identify_objects(_blob(50, 50), gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    truth_path = str(tmp_path / "truth.nc")
    write_object_file(truth_path, truth_t0, lat2d, lon2d,
        [IdentificationResult(labels=truth_labels, objects=truth_objects, valid_time=truth_t0, member_id=None)],
        1, 20.0, 30.0, 1.0)

    forecast_labels, forecast_objects = identify_objects(_blob(50, 50), gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    forecast_path = str(tmp_path / "forecast.nc")
    write_object_file(forecast_path, init_time, lat2d, lon2d,
        [IdentificationResult(labels=forecast_labels, objects=forecast_objects, valid_time=forecast_t0, member_id=None)],
        1, 20.0, 30.0, 1.0)

    summary = run_matching_series(
        [truth_path], [forecast_path],
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2,
        output_dir=str(tmp_path / "matches"), max_time_offset_minutes=5.0,
        file_grouping="init_snapshot",
    )
    assert len(summary.output_paths) == 1
    contents = read_match_file(summary.output_paths[0])
    print(f"\n[lazy-check5] n_truth_source_files={contents.n_truth_source_files}")
    assert contents.n_truth_source_files == 1
    assert len(contents.records) == 1
    assert contents.records[0].category == "hit"


def test_run_matching_series_init_snapshot_requires_shared_init_time(tmp_path):
    """A clear, named error rather than an ambiguous/wrong output filename if
    forecast_files don't share one common init_time -- real synthetic files
    with genuinely different init_times, not a fake-path shortcut."""
    from python_obj.obj_core import IdentificationResult, write_object_file

    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    t0 = datetime(2023, 5, 1, 0, 0, 0)

    truth_labels, truth_objects = identify_objects(_blob(50, 50), gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    truth_path = str(tmp_path / "truth.nc")
    write_object_file(truth_path, t0, lat2d, lon2d,
        [IdentificationResult(labels=truth_labels, objects=truth_objects, valid_time=t0, member_id=None)],
        1, 20.0, 30.0, 1.0)

    forecast_labels, forecast_objects = identify_objects(_blob(50, 50), gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)
    forecast_path_a = str(tmp_path / "forecast_a.nc")
    write_object_file(forecast_path_a, datetime(2023, 5, 1, 0, 0, 0), lat2d, lon2d,
        [IdentificationResult(labels=forecast_labels, objects=forecast_objects, valid_time=t0, member_id=None)],
        1, 20.0, 30.0, 1.0)
    forecast_path_b = str(tmp_path / "forecast_b.nc")
    write_object_file(forecast_path_b, datetime(2023, 5, 2, 0, 0, 0), lat2d, lon2d,
        [IdentificationResult(labels=forecast_labels, objects=forecast_objects, valid_time=t0, member_id=None)],
        1, 20.0, 30.0, 1.0)

    with pytest.raises(ValueError, match="init_snapshot"):
        run_matching_series(
            [truth_path], [forecast_path_a, forecast_path_b],
            max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2,
            output_dir=str(tmp_path / "matches"), file_grouping="init_snapshot",
        )


def test_run_matching_series_unknown_file_grouping_raises(tmp_path):
    from python_obj.obj_core import IdentificationResult, write_object_file

    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    t0 = datetime(2023, 5, 1, 0, 0, 0)
    labels, objects = identify_objects(_blob(50, 50), gg, thresh_1=20, thresh_2=30, area_thresh_km2=1.0)

    truth_path = str(tmp_path / "truth.nc")
    write_object_file(truth_path, t0, lat2d, lon2d,
        [IdentificationResult(labels=labels, objects=objects, valid_time=t0, member_id=None)], 1, 20.0, 30.0, 1.0)
    forecast_path = str(tmp_path / "forecast.nc")
    write_object_file(forecast_path, t0, lat2d, lon2d,
        [IdentificationResult(labels=labels, objects=objects, valid_time=t0, member_id=None)], 1, 20.0, 30.0, 1.0)

    with pytest.raises(ValueError, match="file_grouping"):
        run_matching_series(
            [truth_path], [forecast_path],
            max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2,
            output_dir=str(tmp_path / "matches"), file_grouping="not_a_real_mode",
        )
