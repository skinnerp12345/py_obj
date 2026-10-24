"""Tests for skip_existing (default true in configs) across every processing step,
on bundled sample data. For each step: a second run skips (outputs untouched),
skip_existing: false overwrites, and a truncated (unreadable) output is recomputed.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj/tests/test_skip_existing.py -v -s
"""

import glob
import os
import shutil

import pytest
import yaml

from py_obj.config import load_config
from py_obj.output_check import readable_output

_PY_OBJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(_PY_OBJ, "sample_data")
RAW_MRMS = os.path.join(SAMPLE, "mpas_case", "mrms")
INTERP_MRMS = os.path.join(SAMPLE, "mpas_case", "interp_mrms")
MPAS = os.path.join(SAMPLE, "mpas_case", "mpas_mem1")
HRRR_GRIB2 = os.path.join(SAMPLE, "grib2_case", "hrrr")

_LINEAR = {"linear_eccentricity_threshold": 0.8, "linear_length_threshold_km": 200.0,
           "mixed_eccentricity_threshold": 0.75, "mixed_length_threshold_km": 100.0}


def _cfg(tmp_path, sections, name="cfg.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(sections))
    return str(p)


def _mtimes(paths):
    return {p: os.stat(p).st_mtime_ns for p in paths}


def _truncate(path):
    with open(path, "rb") as fh:
        head = fh.read(2000)
    with open(path, "wb") as fh:
        fh.write(head)
    assert not readable_output(path)


def _obs_section(out_dir, track, **extra):
    sec = {"file_format": "netcdf", "var_name": "refl_consv", "lat_name": "lat", "lon_name": "lon",
           "interp_mrms_dir": INTERP_MRMS, "object_output_dir": out_dir, "boundary_threshold": 40.0,
           "max_value_threshold": 45.0, "area_threshold_km2": 108.0, "track": track, "file_grouping": "single"}
    sec.update(extra)
    return sec


# --- shared helper + config ---------------------------------------------------------

def test_readable_output(tmp_path):
    assert not readable_output(str(tmp_path / "missing.nc"))
    bad = tmp_path / "bad.nc"
    bad.write_bytes(b"not netcdf")
    assert not readable_output(str(bad))
    good = sorted(glob.glob(os.path.join(INTERP_MRMS, "**", "*.nc"), recursive=True))[0]
    assert readable_output(good)


def test_config_default_true_and_bool_validation(tmp_path):
    c = load_config(_cfg(tmp_path, {"observations": _obs_section("o", False)}))
    assert c.observations.skip_existing is True
    c = load_config(_cfg(tmp_path, {"observations": _obs_section("o", False, skip_existing=False)}))
    assert c.observations.skip_existing is False
    with pytest.raises(ValueError, match="skip_existing must be true or false"):
        load_config(_cfg(tmp_path, {"observations": _obs_section("o", False, skip_existing="false")}))


# --- interpolation (per MRMS file) --------------------------------------------------

def test_interpolation_skip_overwrite_and_truncated(tmp_path):
    from py_obj.drivers.interpolate_mrms import run_one_case
    out = str(tmp_path / "interp")
    sec = {"raw_mrms_dir": RAW_MRMS, "interp_mrms_dir": out, "target_grid_sw_lat": 42.0, "target_grid_sw_lon": -86.0,
           "target_grid_dx_km": 3.0, "target_grid_nx": 50, "target_grid_ny": 50, "n_workers": 2,
           "weight_cache_dir": str(tmp_path / "wc")}
    cfg = _cfg(tmp_path, {"interpolation": sec})
    first = run_one_case(cfg)
    assert first.n_success == 3 and first.n_skipped_existing == 0
    files = sorted(glob.glob(os.path.join(out, "**", "*.nc"), recursive=True))
    before = _mtimes(files)

    again = run_one_case(cfg)
    assert again.n_skipped_existing == 3 and _mtimes(files) == before

    _truncate(files[0])
    resumed = run_one_case(cfg)
    assert resumed.n_skipped_existing == 2 and readable_output(files[0])

    over = run_one_case(_cfg(tmp_path, {"interpolation": dict(sec, skip_existing=False)}, "cfg_over.yaml"))
    assert over.n_skipped_existing == 0 and all(_mtimes(files)[f] != before[f] for f in files[1:])


# --- object identification ----------------------------------------------------------

def test_objects_tracking_off_recomputes_only_missing(tmp_path):
    from py_obj.drivers.identify_track_mrms import run_one_case
    out = str(tmp_path / "obj")
    cfg = _cfg(tmp_path, {"observations": _obs_section(out, track=False), "linear_classification": _LINEAR})
    written = run_one_case(cfg)
    assert len(written) == 3
    before = _mtimes(written)
    assert run_one_case(cfg) == [] and _mtimes(written) == before  # everything exists -> nothing done

    os.remove(written[1])
    resumed = run_one_case(cfg)
    assert resumed == [written[1]]                                    # only the missing file
    assert _mtimes([written[0], written[2]]) == {w: before[w] for w in (written[0], written[2])}


def test_objects_tracking_on_recomputes_whole_run(tmp_path):
    from py_obj.drivers.identify_track_mrms import run_one_case
    out = str(tmp_path / "obj")
    cfg = _cfg(tmp_path, {"observations": _obs_section(out, track=True), "linear_classification": _LINEAR})
    written = run_one_case(cfg)
    before = _mtimes(written)
    assert run_one_case(cfg) == []
    _truncate(written[1])
    resumed = run_one_case(cfg)
    assert sorted(resumed) == sorted(written)                        # all-or-nothing: tracks stay continuous
    assert all(_mtimes(written)[w] != before[w] for w in written)


def test_objects_grib2_init_snapshot_skip_and_overwrite(tmp_path):
    from py_obj.drivers.identify_track_model import run_one_case
    model = {"file_format": "grib2", "var_name": "REFC", "input_dir": HRRR_GRIB2, "file_pattern": "*.grib2",
             "boundary_threshold": 45.0, "max_value_threshold": 50.0, "area_threshold_km2": 108.0,
             "file_grouping": "init_snapshot", "object_output_dir": str(tmp_path / "obj")}
    cfg = _cfg(tmp_path, {"model": model, "linear_classification": _LINEAR})
    written = run_one_case(cfg)
    assert len(written) == 1
    before = _mtimes(written)
    assert run_one_case(cfg) == [] and _mtimes(written) == before
    over = run_one_case(_cfg(tmp_path, {"model": dict(model, skip_existing=False), "linear_classification": _LINEAR}, "o.yaml"))
    assert over == written and _mtimes(written) != before


def test_library_default_still_overwrites(tmp_path):
    """Direct library calls keep their old behavior (skip_existing=False)."""
    from py_obj.obj_core import SeriesEntry, run_object_id_series
    from py_obj.regrid import load_mrms_netcdf
    files = sorted(glob.glob(os.path.join(INTERP_MRMS, "**", "*.nc"), recursive=True))
    manifest = [SeriesEntry(valid_time=load_mrms_netcdf(f).valid_time, filepath=f, member_id=None) for f in files]
    kw = dict(thresh_1=40.0, thresh_2=45.0, area_thresh_km2=108.0, output_dir=str(tmp_path / "obj"))
    first = run_object_id_series(manifest, lambda e: load_mrms_netcdf(e.filepath), **kw)
    before = _mtimes(first)
    second = run_object_id_series(manifest, lambda e: load_mrms_netcdf(e.filepath), **kw)
    assert second == first and all(_mtimes(first)[p] != before[p] for p in first)


# --- matching (per forecast time) ----------------------------------------------------

def test_matching_per_time_resume(tmp_path):
    from py_obj.drivers import identify_track_model, identify_track_mrms, run_matching
    obj_mrms, obj_mpas = str(tmp_path / "obj_mrms"), str(tmp_path / "obj_mpas")
    identify_track_mrms.run_one_case(_cfg(tmp_path, {"observations": _obs_section(obj_mrms, False),
                                                    "linear_classification": _LINEAR}, "obs.yaml"))
    model = {"file_format": "netcdf", "var_name": "refl10cm_max", "lat_name": "latitude", "lon_name": "longitude",
             "input_dir": MPAS, "file_pattern": "*_f00[1-3].nc", "init_attr": "initializationTime",
             "lead_attr": "forecastHour", "init_format": "%Y%m%d%H", "boundary_threshold": 35.0,
             "max_value_threshold": 40.0, "area_threshold_km2": 108.0, "object_output_dir": obj_mpas}
    identify_track_model.run_one_case(_cfg(tmp_path, {"model": model, "linear_classification": _LINEAR}, "model.yaml"))
    match = {"truth_object_dir": obj_mrms, "forecast_object_dir": obj_mpas, "output_dir": str(tmp_path / "match"),
             "file_pattern": "obj_*.nc", "max_boundary_disp_km": 40.0, "max_centroid_disp_km": 40.0, "ti_threshold": 0.2}
    cfg = _cfg(tmp_path, {"matching": match}, "match.yaml")
    first = run_matching.run_one_case(cfg).output_paths
    assert len(first) == 3
    before = _mtimes(first)
    assert run_matching.run_one_case(cfg).output_paths == [] and _mtimes(first) == before
    os.remove(first[2])
    assert run_matching.run_one_case(cfg).output_paths == [first[2]]
    assert _mtimes(first[:2]) == {p: before[p] for p in first[:2]}


# --- histograms ------------------------------------------------------------------------

def test_histogram_mrms_per_day_and_model_case(tmp_path):
    from py_obj.drivers import build_histogram_model, build_histogram_mrms
    obs = {"interp_mrms_dir": INTERP_MRMS, "output_dir": str(tmp_path / "hist_mrms")}
    model = {"input_dir": HRRR_GRIB2, "file_format": "grib2", "var_name": "REFC", "file_pattern": "*.grib2",
             "output_dir": str(tmp_path / "hist_model")}
    cfg = _cfg(tmp_path, {"histogram_observations": obs, "histogram_model": model})
    day_files = build_histogram_mrms.run_one_case(cfg)
    model_file = build_histogram_model.run_one_case(cfg)
    before = _mtimes(day_files + [model_file])
    assert build_histogram_mrms.run_one_case(cfg) == []
    assert build_histogram_model.run_one_case(cfg) == model_file
    assert _mtimes(day_files + [model_file]) == before

    _truncate(model_file)
    assert build_histogram_model.run_one_case(cfg) == model_file and readable_output(model_file)

    over_cfg = _cfg(tmp_path, {"histogram_observations": dict(obs, skip_existing=False),
                               "histogram_model": dict(model, skip_existing=False)}, "over.yaml")
    assert build_histogram_mrms.run_one_case(over_cfg) == day_files
    assert _mtimes(day_files) != {p: before[p] for p in day_files}


def test_batch_histogram_resumes_after_interruption(tmp_path):
    """The failure scenario: a batch dies after some cases; rerunning the same
    template recomputes only the cases without readable output."""
    from py_obj.batch_config import expand_batch_config
    from py_obj.drivers.build_histogram_model import run_one_case
    for day in ("20260518", "20260519"):
        d = tmp_path / "archive" / day
        d.mkdir(parents=True)
        for f in glob.glob(os.path.join(HRRR_GRIB2, "*.grib2")):
            shutil.copy(f, d / os.path.basename(f).replace("20260518", day))
    template = _cfg(tmp_path, {
        "cases": {"dates": ["20260518", "20260519"], "init_times": ["00"], "init_time_format": "%H"},
        "histogram_model": {"input_dir": str(tmp_path / "archive" / "{date}"), "file_format": "grib2",
                            "var_name": "REFC", "file_pattern": "hrrr_ncep_{date}{init_time}f*.grib2",
                            "output_dir": str(tmp_path / "hist" / "{date}_{init_time}")},
    }, "template.yaml")
    cases = expand_batch_config(template, str(tmp_path / "materialized")).case_paths
    outputs = [run_one_case(c) for c in cases]
    before = _mtimes(outputs)
    _truncate(outputs[1])                       # case 2 "died mid-write"
    resumed = [run_one_case(c) for c in cases]  # same template, rerun
    assert resumed == outputs
    assert os.stat(outputs[0]).st_mtime_ns == before[outputs[0]]   # case 1 skipped
    assert readable_output(outputs[1])                              # case 2 recomputed
