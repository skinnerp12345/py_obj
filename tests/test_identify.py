"""Step 4 validation: object identification + optional in-time tracking.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_identify.py -v -s
"""

import glob
import os
from datetime import datetime, timedelta

import netCDF4
import numpy as np
import pytest

from python_obj.obj_core import (
    IdentificationResult,
    SeriesEntry,
    StormObject,
    build_model_manifest,
    identify_objects,
    iter_object_slices,
    precompute_grid_geometry,
    read_object_file,
    run_object_id_series,
    track_objects_incremental,
    write_object_file,
)
from python_obj.regrid import load_mrms_netcdf, load_model_netcdf

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
MPAS_DIR = os.path.join(SAMPLE_DATA_DIR, "mpas_case", "mpas_mem1")
INTERP_MRMS_DIR = os.path.join(SAMPLE_DATA_DIR, "mpas_case", "interp_mrms", "20230501")


def _synthetic_grid(ny=60, nx=60, lat0=30.0, lon0=-100.0, step=0.05):
    lat = lat0 + np.arange(ny) * step
    lon = lon0 + np.arange(nx) * step
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lat2d, lon2d


# --- Check 1: identify_objects on a synthetic field -------------------------

def test_identify_objects_synthetic():
    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)

    data = np.zeros((60, 60))
    data[5:15, 5:15] = 50.0    # 10x10 blob, well above area threshold
    data[40:43, 40:43] = 50.0  # 3x3 blob, small -- should be dropped by area_thresh
    data[20:22, 20:22] = 5.0   # below thresh_1 entirely -- never becomes a candidate

    # at this synthetic grid's ~5.5km spacing, one cell is ~26 km^2, so the 3x3
    # blob is ~236 km^2 and the 10x10 blob is ~2660 km^2 -- pick a threshold
    # between them so the area filter actually differentiates the two
    labels, objects = identify_objects(data, gg, thresh_1=20.0, thresh_2=30.0, area_thresh_km2=500.0)

    print(f"\n[id-check1] {len(objects)} objects: "
          f"{[(o.id, round(o.area_km2,1), o.max_intensity) for o in objects]}")
    assert len(objects) == 1, "the small blob should have been dropped by the area threshold"
    assert objects[0].max_intensity == 50.0
    assert objects[0].area_km2 > 500.0


# --- Check 2: identify_objects on real data ---------------------------------

def test_identify_objects_real_data():
    mpas = load_model_netcdf(os.path.join(MPAS_DIR, "interp_mpas_3km_2023050100_mem1_f001.nc"), varname="refl10cm_max")
    gg = precompute_grid_geometry(mpas.lat2d, mpas.lon2d)

    labels, objects = identify_objects(mpas.data, gg, thresh_1=39.8, thresh_2=45.6, area_thresh_km2=108.0)
    print(f"\n[id-check2] MPAS f000: {len(objects)} objects")
    if objects:
        px_per_km2 = [o.area_px / o.area_km2 for o in objects]
        print(f"  pixels-per-km2 range: {min(px_per_km2):.3f}-{max(px_per_km2):.3f} "
              f"(naive 3km-grid guess would be 1/9={1/9:.3f} px/km2)")
        # physical area should differ meaningfully from a naive pixel-count*dx^2
        # estimate (Step 2's anisotropy/projection-distortion finding), not
        # exactly 1/9 for every object
        assert not all(abs(p - 1 / 9) < 1e-6 for p in px_per_km2)


# --- Check 3: tracking correctness, synthetic 3-timestep series ------------

def test_tracking_correctness_synthetic_series():
    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    t0 = datetime(2023, 5, 1, 0, 0, 0)

    d0 = np.zeros((60, 60)); d0[10:15, 10:15] = 50.0
    labels0, objs0 = identify_objects(d0, gg, 20.0, 30.0, 1.0)
    tracked0, nid = track_objects_incremental(None, None, None, objs0, labels0, t0, gg, next_track_id=1)

    d1 = np.zeros((60, 60)); d1[10:15, 11:16] = 50.0  # shifted, overlapping -> persists
    labels1, objs1 = identify_objects(d1, gg, 20.0, 30.0, 1.0)
    t1 = t0 + timedelta(minutes=5)
    tracked1, nid = track_objects_incremental(tracked0, labels0, t0, objs1, labels1, t1, gg, next_track_id=nid)

    d2 = np.zeros((60, 60)); d2[45:48, 45:48] = 50.0  # unrelated new object elsewhere
    labels2, objs2 = identify_objects(d2, gg, 20.0, 30.0, 1.0)
    t2 = t1 + timedelta(minutes=5)
    tracked2, nid = track_objects_incremental(tracked1, labels1, t1, objs2, labels2, t2, gg, next_track_id=nid)

    print(f"\n[id-check3] t0={[(o.age_seconds, o.track_id) for o in tracked0]} "
          f"t1={[(o.age_seconds, o.track_id) for o in tracked1]} "
          f"t2={[(o.age_seconds, o.track_id) for o in tracked2]}")

    assert tracked0[0].age_seconds == 0.0
    assert tracked1[0].age_seconds == 300.0
    assert tracked1[0].track_id == tracked0[0].track_id, "persisting object should keep the same track_id"
    assert tracked2[0].age_seconds == 0.0, "unrelated new object should start at age 0"
    assert tracked2[0].track_id != tracked1[0].track_id, "new object should get a new track_id"


# --- Check 4: tracking genericity (no obs-specific behavior) ---------------

def test_tracking_genericity_obs_vs_forecast_labels():
    """Run the identical tracking scenario twice, once labeled as if it were an
    obs series and once as a forecast series -- results must be identical,
    proving nothing in track_objects_incremental is source-specific."""
    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    t0 = datetime(2023, 5, 1, 0, 0, 0)
    t1 = t0 + timedelta(minutes=5)

    d0 = np.zeros((60, 60)); d0[10:15, 10:15] = 50.0
    d1 = np.zeros((60, 60)); d1[10:15, 11:16] = 50.0

    def run_series():
        labels0, objs0 = identify_objects(d0, gg, 20.0, 30.0, 1.0)
        tracked0, nid = track_objects_incremental(None, None, None, objs0, labels0, t0, gg, next_track_id=1)
        labels1, objs1 = identify_objects(d1, gg, 20.0, 30.0, 1.0)
        tracked1, nid = track_objects_incremental(tracked0, labels0, t0, objs1, labels1, t1, gg, next_track_id=nid)
        return [(o.age_seconds, o.track_id) for o in tracked1]

    obs_result = run_series()      # "obs" -- but nothing in the call distinguishes it
    forecast_result = run_series()  # "forecast" -- identical inputs/call shape

    print(f"\n[id-check4] obs-labeled result={obs_result} forecast-labeled result={forecast_result}")
    assert obs_result == forecast_result


# --- Check 5: object file round-trip, all four shapes -----------------------

# --- Check 4b: write_object_file streams labels, never builds a duplicate
# full-size consolidated array (the real, confirmed OOM cause fixed here) ---

def test_write_object_file_does_not_allocate_duplicate_labels_array(tmp_path):
    """Real production bug: for a large multi-member/multi-lead-time case
    (e.g. 5 members x 133 hourly lead times on a full-CONUS grid),
    write_object_file() used to build ONE consolidated labels_stack array
    the same total size as every individual result's label array combined,
    on top of those individual arrays the caller already holds -- a real
    OOM kill in production. This test proves the fix quantitatively: peak
    memory traced *inside* write_object_file() (results' own arrays are
    built and held BEFORE tracing starts, so they don't count) must stay
    well below the size of one duplicate consolidated array -- which the
    pre-fix implementation would have allocated in full.
    """
    import tracemalloc

    ny, nx = 200, 200
    n_members, n_times = 20, 20
    lat2d, lon2d = _synthetic_grid(ny=ny, nx=nx)
    t0 = datetime(2023, 5, 1, 0, 0, 0)

    # built and held BEFORE tracing starts -- mirrors how the caller
    # (id_pipeline.py's all_results) already owns every individual label
    # array before write_object_file() is ever called.
    results = [
        IdentificationResult(
            labels=np.zeros((ny, nx), dtype=np.int32), objects=[],
            valid_time=t0 + timedelta(minutes=t), member_id=f"mem{m}",
        )
        for m in range(n_members) for t in range(n_times)
    ]

    one_consolidated_array_bytes = ny * nx * 4 * n_members * n_times  # int32

    tracemalloc.start()
    path = str(tmp_path / "big.nc")
    write_object_file(path, t0, lat2d, lon2d, results, 1, 20.0, 30.0, 1.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n[id-check4b] peak traced memory inside write_object_file={peak / 1e6:.1f} MB "
          f"(one duplicate consolidated array would be {one_consolidated_array_bytes / 1e6:.1f} MB)")
    assert peak < one_consolidated_array_bytes * 0.5, (
        "write_object_file must not build a second, full-size consolidated labels array"
    )

    # correctness: streaming write must still produce the exact right shape/content
    c = read_object_file(path)
    assert c.labels.shape == (n_members, n_times, ny, nx)
    np.testing.assert_array_equal(c.labels, np.zeros((n_members, n_times, ny, nx), dtype=np.int32))


def test_object_file_roundtrip_all_shapes(tmp_path):
    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    t0 = datetime(2023, 5, 1, 0, 0, 0)

    def make_result(offset, member_id, valid_time):
        d = np.zeros((60, 60))
        d[10:15, 10 + offset:15 + offset] = 50.0
        labels, objs = identify_objects(d, gg, 20.0, 30.0, 1.0)
        return IdentificationResult(labels=labels, objects=objs, valid_time=valid_time, member_id=member_id)

    # (a) single
    r_single = make_result(0, None, t0)
    p = str(tmp_path / "single.nc")
    write_object_file(p, t0, lat2d, lon2d, [r_single], 1, 20.0, 30.0, 1.0)
    c = read_object_file(p)
    assert c.labels.shape == (60, 60)
    assert c.member_ids is None and c.time_index is None and c.member_index is None
    print(f"\n[id-check5] single: labels.shape={c.labels.shape}")

    # (b) member_series
    results = [make_result(i, "mem1", t0 + timedelta(minutes=5 * i)) for i in range(3)]
    p = str(tmp_path / "member_series.nc")
    write_object_file(p, t0, lat2d, lon2d, results, 1, 20.0, 30.0, 1.0)
    c = read_object_file(p)
    assert c.labels.shape == (3, 60, 60)
    assert c.member_ids == ["mem1"] and c.member_index is None
    assert list(c.time_index) == sorted(c.time_index.tolist())
    print(f"[id-check5] member_series: labels.shape={c.labels.shape} valid_times={len(c.valid_times)}")

    # (c) ensemble_snapshot
    results = [make_result(i, f"mem{i+1}", t0) for i in range(3)]
    p = str(tmp_path / "ensemble_snapshot.nc")
    write_object_file(p, t0, lat2d, lon2d, results, 1, 20.0, 30.0, 1.0)
    c = read_object_file(p)
    assert c.labels.shape == (3, 60, 60)
    assert c.member_ids == ["mem1", "mem2", "mem3"] and c.time_index is None
    print(f"[id-check5] ensemble_snapshot: labels.shape={c.labels.shape} member_ids={c.member_ids}")

    # (d) full
    results = []
    for m in range(2):
        for t in range(2):
            results.append(make_result(m + t, f"mem{m+1}", t0 + timedelta(minutes=5 * t)))
    p = str(tmp_path / "full.nc")
    write_object_file(p, t0, lat2d, lon2d, results, 1, 20.0, 30.0, 1.0)
    c = read_object_file(p)
    assert c.labels.shape == (2, 2, 60, 60)
    assert c.member_ids == ["mem1", "mem2"] and len(c.valid_times) == 2
    # filter the flat table down to member 1 ("mem2", index 1), time index 0
    filtered = [
        o for o, mi, ti in zip(c.objects, c.member_index, c.time_index)
        if mi == 1 and ti == 0
    ]
    assert len(filtered) >= 1
    print(f"[id-check5] full: labels.shape={c.labels.shape}, filtered subset size={len(filtered)}")


# --- Check 5b: legacy files (written before n_source_files existed) --------
#
# Real, not hypothetical: a real downloaded MRMS object archive (produced
# before this schema field existed) has no n_source_files attribute at all
# -- read_object_file() used to raise AttributeError on every one of them.
# A "single" grouping file is, by construction, always derived from exactly
# one input file, so n_source_files=1 can be safely inferred for that shape
# specifically -- without touching the old source_files string attribute at
# all. Any other shape genuinely could have >1 source file, so inference
# must NOT be attempted there -- confirmed via a deliberate failure case.

def _write_legacy_file_without_n_source_files(path: str, results, lat2d, lon2d, t0) -> None:
    """Writes via the real writer, then strips n_source_files to simulate a
    file written before that attribute existed -- avoids hand-rolling a
    second, parallel netCDF-writing implementation just for this test."""
    write_object_file(path, t0, lat2d, lon2d, results, 1, 20.0, 30.0, 1.0)
    with netCDF4.Dataset(path, "a") as ds:
        del ds.n_source_files


def test_legacy_single_file_infers_n_source_files_as_one(tmp_path):
    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    t0 = datetime(2023, 5, 1, 0, 0, 0)
    d = np.zeros((60, 60)); d[10:15, 10:15] = 50.0
    labels, objs = identify_objects(d, gg, 20.0, 30.0, 1.0)
    result = IdentificationResult(labels=labels, objects=objs, valid_time=t0, member_id=None)

    p = str(tmp_path / "legacy_single.nc")
    _write_legacy_file_without_n_source_files(p, [result], lat2d, lon2d, t0)

    c = read_object_file(p)
    print(f"\n[id-check5b] legacy single-shape file (no n_source_files attr): inferred n_source_files={c.n_source_files}")
    assert c.n_source_files == 1


def test_legacy_multi_file_shape_without_n_source_files_raises(tmp_path):
    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    t0 = datetime(2023, 5, 1, 0, 0, 0)

    results = []
    for i in range(3):
        d = np.zeros((60, 60)); d[10:15, 10 + i:15 + i] = 50.0
        labels, objs = identify_objects(d, gg, 20.0, 30.0, 1.0)
        results.append(IdentificationResult(labels=labels, objects=objs, valid_time=t0, member_id=f"mem{i+1}"))

    p = str(tmp_path / "legacy_ensemble_snapshot.nc")
    _write_legacy_file_without_n_source_files(p, results, lat2d, lon2d, t0)

    with pytest.raises(ValueError, match="n_source_files"):
        read_object_file(p)
    print("\n[id-check5b] legacy multi-file-shape file (no n_source_files attr) correctly raises, not silently guesses")


# --- Check 6: real end-to-end series, "single" and "member_series" ---------

def test_real_end_to_end_single_and_member_series(tmp_path):
    # restrict to the 3-hour f001-f003 series -- MPAS_DIR also bundles f048
    # (full-domain, for test_linear_classification.py only), which would
    # otherwise get swept into this glob and break the "consecutive timestep"
    # tracking scenario this check exercises.
    files = sorted(glob.glob(os.path.join(MPAS_DIR, "interp_mpas_3km_2023050100_mem1_f00[1-3].nc")))
    loader = lambda fp: load_model_netcdf(fp, varname="refl10cm_max")
    manifest = [SeriesEntry(valid_time=loader(f).valid_time, filepath=f, member_id="mem1") for f in files]

    # thresholds chosen (empirically, not the production 45.0/50.2 MPAS pair)
    # to produce genuine overlapping storm objects across all 3 bundled lead
    # times in this small sample crop -- confirmed real persistence (age>0)
    # below, not just fresh detections every frame.
    out_single = run_object_id_series(
        manifest, lambda entry: loader(entry.filepath), thresh_1=35.0, thresh_2=40.0, area_thresh_km2=108.0,
        output_dir=str(tmp_path / "single"), file_grouping="single",
        track_in_time=True, track_bound_disp_km=0.0, init_time=datetime(2023, 5, 1, 0, 0, 0),
    )
    assert len(out_single) == len(manifest)

    out_series = run_object_id_series(
        manifest, lambda entry: loader(entry.filepath), thresh_1=35.0, thresh_2=40.0, area_thresh_km2=108.0,
        output_dir=str(tmp_path / "series"), file_grouping="member_series",
        track_in_time=True, track_bound_disp_km=0.0, init_time=datetime(2023, 5, 1, 0, 0, 0),
    )
    assert len(out_series) == 1
    c = read_object_file(out_series[0])
    assert c.labels.shape[0] == len(manifest)
    ages = [o.age_seconds for o in c.objects]
    print(f"\n[id-check6] MPAS single: {len(out_single)} files; member_series: "
          f"{len(c.objects)} objects, age range {min(ages)}-{max(ages)}s")
    assert min(ages) == 0.0 and max(ages) > 0.0, "expected some objects to persist (age>0) and some to start fresh"

    # separately, obs series (real interpolated MRMS, no member concept)
    mrms_files = sorted(glob.glob(os.path.join(INTERP_MRMS_DIR, "*.nc")))
    obs_manifest = [SeriesEntry(valid_time=load_mrms_netcdf(f).valid_time, filepath=f, member_id=None) for f in mrms_files]
    obs_loader = lambda fp: load_mrms_netcdf(fp)
    out_obs = run_object_id_series(
        obs_manifest, lambda entry: obs_loader(entry.filepath), thresh_1=40.0, thresh_2=45.0, area_thresh_km2=108.0,
        output_dir=str(tmp_path / "obs_series"), file_grouping="member_series",
        track_in_time=True, track_bound_disp_km=0.0,
    )
    assert len(out_obs) == 1
    c_obs = read_object_file(out_obs[0])
    assert c_obs.member_ids is None
    obs_ages = [o.age_seconds for o in c_obs.objects]
    print(f"[id-check6] MRMS obs series: {len(c_obs.objects)} objects, "
          f"age range {min(obs_ages) if obs_ages else None}-{max(obs_ages) if obs_ages else None}s")


# --- Check 7: synthetic multi-member coverage for ensemble groupings -------

def test_synthetic_multi_member_ensemble_groupings(tmp_path):
    mpas_file = os.path.join(MPAS_DIR, "interp_mpas_3km_2023050100_mem1_f001.nc")
    real = load_model_netcdf(mpas_file, varname="refl10cm_max")

    # fabricate 3 fake "members" by rolling the real field slightly -- real
    # multi-member data isn't available yet (see plan notes), so this is
    # explicitly a synthetic-coverage check of the ensemble code paths, not a
    # scientific validation of any particular ensemble's behavior.
    rng = np.random.default_rng(0)
    member_fields = {f"mem{i+1}": np.roll(real.data, shift=i * 3, axis=1) for i in range(3)}

    class _Field:
        def __init__(self, data, lat2d, lon2d, valid_time):
            self.data, self.lat2d, self.lon2d, self.valid_time = data, lat2d, lon2d, valid_time

    def loader(key: str):
        member_id, iso_time = key.split("@")
        return _Field(member_fields[member_id], real.lat2d, real.lon2d, datetime.fromisoformat(iso_time))

    t0 = datetime(2023, 5, 2, 0, 0, 0)
    t1 = t0 + timedelta(hours=1)
    manifest = []
    for member_id in member_fields:
        for vt in (t0, t1):
            manifest.append(SeriesEntry(valid_time=vt, filepath=f"{member_id}@{vt.isoformat()}", member_id=member_id))

    out_snap = run_object_id_series(
        manifest, lambda entry: loader(entry.filepath), thresh_1=39.8, thresh_2=45.6, area_thresh_km2=108.0,
        output_dir=str(tmp_path / "ensemble_snapshot"), file_grouping="ensemble_snapshot",
        track_in_time=True, track_bound_disp_km=0.0, init_time=t0,
    )
    assert len(out_snap) == 2  # one per distinct valid_time
    c_snap = read_object_file(out_snap[0])
    assert c_snap.member_ids == sorted(member_fields.keys())
    assert c_snap.labels.shape[0] == 3
    print(f"\n[id-check7] ensemble_snapshot: {len(out_snap)} files, "
          f"labels.shape={c_snap.labels.shape}, member_ids={c_snap.member_ids}")

    out_full = run_object_id_series(
        manifest, lambda entry: loader(entry.filepath), thresh_1=39.8, thresh_2=45.6, area_thresh_km2=108.0,
        output_dir=str(tmp_path / "full"), file_grouping="full",
        track_in_time=True, track_bound_disp_km=0.0, init_time=t0,
    )
    assert len(out_full) == 1
    c_full = read_object_file(out_full[0])
    assert c_full.labels.shape == (3, 2) + real.lat2d.shape
    # tracking must not link objects across different members -- every
    # track_id should only ever appear under one member_index
    track_id_to_members = {}
    for obj, mi in zip(c_full.objects, c_full.member_index):
        track_id_to_members.setdefault(obj.track_id, set()).add(int(mi))
    cross_member_tracks = {tid: ms for tid, ms in track_id_to_members.items() if len(ms) > 1}
    print(f"[id-check7] full: labels.shape={c_full.labels.shape}, "
          f"cross-member track_ids (should be empty)={cross_member_tracks}")
    assert not cross_member_tracks, "tracking must only link objects within the same member's timeline"


# --- Check 8: init_snapshot -- one file per forecast case, not per valid_time

def _two_case_manifest(real):
    """2 forecast cases (A init 2023-05-01 00Z, B init 2023-05-02 00Z), 2
    members each, lead hours 0/24 -- deliberately overlapping in valid_time:
    case A's lead=24 (2023-05-02 00Z) exactly equals case B's lead=0
    (2023-05-02 00Z), the exact scenario that collides under
    ensemble_snapshot's valid-time-only filename."""
    class _Field:
        def __init__(self, data, lat2d, lon2d, valid_time):
            self.data, self.lat2d, self.lon2d, self.valid_time = data, lat2d, lon2d, valid_time

    rng = np.random.default_rng(0)
    member_fields = {f"mem{i+1}": np.roll(real.data, shift=i * 3, axis=1) for i in range(2)}

    def loader(key: str):
        member_id, iso_time = key.split("@")
        return _Field(member_fields[member_id], real.lat2d, real.lon2d, datetime.fromisoformat(iso_time))

    init_a = datetime(2023, 5, 1, 0, 0, 0)
    init_b = datetime(2023, 5, 2, 0, 0, 0)
    manifest = []
    for init_time in (init_a, init_b):
        for member_id in member_fields:
            for lead_hours in (0, 24):
                vt = init_time + timedelta(hours=lead_hours)
                manifest.append(SeriesEntry(
                    valid_time=vt, filepath=f"{member_id}@{vt.isoformat()}",
                    member_id=member_id, init_time=init_time,
                ))
    return manifest, loader, init_a, init_b


def test_init_snapshot_groups_by_forecast_case_not_valid_time(tmp_path):
    mpas_file = os.path.join(MPAS_DIR, "interp_mpas_3km_2023050100_mem1_f001.nc")
    real = load_model_netcdf(mpas_file, varname="refl10cm_max")
    manifest, loader, init_a, init_b = _two_case_manifest(real)

    out_paths = run_object_id_series(
        manifest, lambda entry: loader(entry.filepath), thresh_1=39.8, thresh_2=45.6, area_thresh_km2=108.0,
        output_dir=str(tmp_path / "init_snapshot"), file_grouping="init_snapshot",
    )
    print(f"\n[id-check8] init_snapshot: {len(out_paths)} files (expect 2, one per forecast case): {[os.path.basename(p) for p in out_paths]}")
    assert len(out_paths) == 2
    assert any(f"{init_a:%Y%m%d_%H%M%S}" in p for p in out_paths)
    assert any(f"{init_b:%Y%m%d_%H%M%S}" in p for p in out_paths)

    # each file must contain BOTH members x BOTH lead times for its own case
    # (4 slices), and valid_time/member_id must be recoverable per-object --
    # confirms the "objects from different lead times/members can be parsed
    # out afterward" property, not just that the file writes without error.
    for path in out_paths:
        contents = read_object_file(path)
        assert contents.labels.shape == (2, 2) + real.lat2d.shape  # (member, time, y, x)
        assert sorted(contents.member_ids) == ["mem1", "mem2"]
        assert len(contents.valid_times) == 2
        slices = list(iter_object_slices(contents))
        assert len(slices) == 4  # 2 members x 2 times
        seen = {(member_id, vt) for member_id, vt, _, _ in slices}
        assert len(seen) == 4, "every (member, valid_time) slice must be distinctly recoverable"
    print("[id-check8] both files: 4 distinct (member, valid_time) slices recovered via iter_object_slices")


def test_init_snapshot_avoids_ensemble_snapshot_collision(tmp_path):
    """Real proof of the bug this feature fixes: two separate
    run_object_id_series calls (mimicking one call per forecast case, exactly
    how the batch workflow invokes this driver) writing to the SAME shared
    output_dir. Under ensemble_snapshot, case B's lead=0 file overwrites case
    A's lead=24 file (both valid at 2023-05-02 00Z) -- confirmed by a file
    count deficit. Under init_snapshot, no collision occurs."""
    mpas_file = os.path.join(MPAS_DIR, "interp_mpas_3km_2023050100_mem1_f001.nc")
    real = load_model_netcdf(mpas_file, varname="refl10cm_max")
    manifest, loader, init_a, init_b = _two_case_manifest(real)
    manifest_a = [e for e in manifest if e.init_time == init_a]
    manifest_b = [e for e in manifest if e.init_time == init_b]

    shared_dir_bad = str(tmp_path / "shared_ensemble_snapshot")
    out_a = run_object_id_series(
        manifest_a, lambda entry: loader(entry.filepath), thresh_1=39.8, thresh_2=45.6, area_thresh_km2=108.0,
        output_dir=shared_dir_bad, file_grouping="ensemble_snapshot",
    )
    out_b = run_object_id_series(
        manifest_b, lambda entry: loader(entry.filepath), thresh_1=39.8, thresh_2=45.6, area_thresh_km2=108.0,
        output_dir=shared_dir_bad, file_grouping="ensemble_snapshot",
    )
    distinct_files_on_disk = len(set(out_a) | set(out_b))
    print(f"\n[id-check8b] ensemble_snapshot into a shared dir: case A wrote {len(out_a)}, case B wrote {len(out_b)}, "
          f"distinct files on disk = {distinct_files_on_disk} (expect 3, NOT 4 -- one overwritten)")
    assert distinct_files_on_disk == 3, "demonstrates the real collision: case B's lead=0 file overwrites case A's lead=24 file"

    shared_dir_good = str(tmp_path / "shared_init_snapshot")
    out_a2 = run_object_id_series(
        manifest_a, lambda entry: loader(entry.filepath), thresh_1=39.8, thresh_2=45.6, area_thresh_km2=108.0,
        output_dir=shared_dir_good, file_grouping="init_snapshot",
    )
    out_b2 = run_object_id_series(
        manifest_b, lambda entry: loader(entry.filepath), thresh_1=39.8, thresh_2=45.6, area_thresh_km2=108.0,
        output_dir=shared_dir_good, file_grouping="init_snapshot",
    )
    distinct_files_fixed = len(set(out_a2) | set(out_b2))
    print(f"[id-check8b] init_snapshot into the same shared dir: distinct files on disk = {distinct_files_fixed} (expect 2, no collision)")
    assert distinct_files_fixed == 2, "init_snapshot must not collide even when both cases share one output_dir"


def test_init_snapshot_requires_init_time(tmp_path):
    mpas_file = os.path.join(MPAS_DIR, "interp_mpas_3km_2023050100_mem1_f001.nc")
    real = load_model_netcdf(mpas_file, varname="refl10cm_max")
    manifest = [SeriesEntry(valid_time=real.valid_time, filepath="x", member_id="mem1", init_time=None)]

    with pytest.raises(ValueError, match="init_snapshot.*init_time"):
        run_object_id_series(
            manifest, lambda entry: real, thresh_1=39.8, thresh_2=45.6, area_thresh_km2=108.0,
            output_dir=str(tmp_path / "missing_init"), file_grouping="init_snapshot",
        )
    print("\n[id-check8c] init_snapshot without init_time raises a clear, named error")


def test_init_snapshot_real_data_matches_full_mode(tmp_path):
    """Real MPAS data via build_model_manifest() (not hand-built): confirms
    init_time is actually derived correctly end to end (not just in the
    synthetic tests above), and that init_snapshot on a manifest representing
    exactly one real forecast case is numerically identical to "full" --
    same underlying results, just grouped/named differently."""
    manifest, loader = build_model_manifest(
        input_dir=MPAS_DIR, file_pattern="*_f00[1-3].nc",  # excludes the f048 file (used only by linear-classification tests)
        member_subdirs=False, stacked_members=False,
        var_name="refl10cm_max", lat_name="latitude", lon_name="longitude",
        init_attr="initializationTime", lead_attr="forecastHour", init_format="%Y%m%d%H",
    )
    assert len(manifest) == 3
    init_times = {e.init_time for e in manifest}
    print(f"\n[id-check8d] real manifest: {len(manifest)} entries, distinct init_times={init_times}")
    assert len(init_times) == 1, "every file in this one real forecast case must share the same real init_time"
    assert None not in init_times

    # 39.8/45.6 (not 45.0/50.2) -- confirmed on this small bundled crop that
    # f001/f003 each have exactly 1 real object at this threshold (max value
    # only 43.6-46.5 dBZ, below 50.2's stricter bar), giving the full-vs-
    # init_snapshot comparison below real, non-zero objects to compare.
    loader_fn = lambda entry: loader(entry.filepath, extra_dim_index=entry.extra_dim_index)
    out_full = run_object_id_series(
        manifest, loader_fn, thresh_1=39.8, thresh_2=45.6, area_thresh_km2=108.0,
        output_dir=str(tmp_path / "full"), file_grouping="full",
    )
    out_init = run_object_id_series(
        manifest, loader_fn, thresh_1=39.8, thresh_2=45.6, area_thresh_km2=108.0,
        output_dir=str(tmp_path / "init_snapshot"), file_grouping="init_snapshot",
    )
    assert len(out_init) == 1, "one real case -> exactly one init_snapshot file"

    c_full = read_object_file(out_full[0])
    c_init = read_object_file(out_init[0])
    print(f"[id-check8d] full: {len(c_full.objects)} objects, init_snapshot: {len(c_init.objects)} objects")
    assert len(c_full.objects) == len(c_init.objects)
    assert sorted(o.id for o in c_full.objects) == sorted(o.id for o in c_init.objects)
    assert c_init.init_time == next(iter(init_times))
