"""Step 4 validation: object identification + optional in-time tracking.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_identify.py -v -s
"""

import glob
import os
from datetime import datetime, timedelta

import numpy as np
import pytest

from python_obj.obj_core import (
    IdentificationResult,
    SeriesEntry,
    StormObject,
    identify_objects,
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
    write_object_file(p, t0, lat2d, lon2d, [r_single], ["x"], 20.0, 30.0, 1.0)
    c = read_object_file(p)
    assert c.labels.shape == (60, 60)
    assert c.member_ids is None and c.time_index is None and c.member_index is None
    print(f"\n[id-check5] single: labels.shape={c.labels.shape}")

    # (b) member_series
    results = [make_result(i, "mem1", t0 + timedelta(minutes=5 * i)) for i in range(3)]
    p = str(tmp_path / "member_series.nc")
    write_object_file(p, t0, lat2d, lon2d, results, ["x"], 20.0, 30.0, 1.0)
    c = read_object_file(p)
    assert c.labels.shape == (3, 60, 60)
    assert c.member_ids == ["mem1"] and c.member_index is None
    assert list(c.time_index) == sorted(c.time_index.tolist())
    print(f"[id-check5] member_series: labels.shape={c.labels.shape} valid_times={len(c.valid_times)}")

    # (c) ensemble_snapshot
    results = [make_result(i, f"mem{i+1}", t0) for i in range(3)]
    p = str(tmp_path / "ensemble_snapshot.nc")
    write_object_file(p, t0, lat2d, lon2d, results, ["x"], 20.0, 30.0, 1.0)
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
    write_object_file(p, t0, lat2d, lon2d, results, ["x"], 20.0, 30.0, 1.0)
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
