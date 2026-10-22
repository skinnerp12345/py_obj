"""Tests for GRIB2 model-forecast input (regrid/io_grib.py), against the bundled
real HRRR/RRFS sample (sample_data/grib2_case/: per file, a decoy reflectivity
record FIRST, then the composite reflectivity (REFC) record in each model's own
encoding -- HRRR 0/16/196, RRFS 0/16/5).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj/tests/test_grib2_model.py -v -s
"""

import glob
import os
from datetime import datetime

import numpy as np
import pygrib
import pytest

from py_obj.regrid import GRIB2_FIELD_PRESETS, load_grib2_grid, load_model_grib2, read_grib2_times, resolve_grib_selector

_PY_OBJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(_PY_OBJ, "sample_data", "grib2_case")
HRRR_F001 = os.path.join(SAMPLE, "hrrr", "hrrr_ncep_2026051800f001.grib2")
RRFS_F001 = os.path.join(SAMPLE, "rrfs", "rrfs_2026051800f001.grib2")
REAL_DIR = os.path.join(os.path.dirname(_PY_OBJ), "grib2_testing")  # external, not bundled

_RAW_REFC = {HRRR_F001: dict(discipline=0, parameterCategory=16, parameterNumber=196),
             RRFS_F001: dict(discipline=0, parameterCategory=16, parameterNumber=5)}


def _raw(path, keys):
    g = pygrib.open(path)
    try:
        return g.select(**keys)[0]
    finally:
        g.close()


@pytest.mark.parametrize("path", [HRRR_F001, RRFS_F001])
def test_refc_preset_reads_the_composite_record_not_the_decoy(path):
    fld = load_model_grib2(path, var_name="REFC")
    raw = _raw(path, _RAW_REFC[path])
    rl, rn = raw.latlons()
    assert np.array_equal(fld.data, np.asarray(raw.values, dtype=float))
    assert np.array_equal(fld.lat2d, rl) and np.array_equal(fld.lon2d, rn)
    assert fld.data.shape == (1059, 1799)
    assert fld.valid_time == datetime(2026, 5, 18, 1)
    g = pygrib.open(path)
    first = np.asarray(g.message(1).values, dtype=float)
    g.close()
    assert not np.array_equal(fld.data, first)  # message 1 is the decoy


@pytest.mark.parametrize("path", [HRRR_F001, RRFS_F001])
def test_times_come_from_grib2_keys(path):
    assert read_grib2_times(path) == (datetime(2026, 5, 18, 0), datetime(2026, 5, 18, 1))


def test_explicit_selector_overrides_preset():
    decoy = {"discipline": 0, "parameterCategory": 16, "parameterNumber": 195}
    fld = load_model_grib2(HRRR_F001, var_name="REFC", grib_selector=decoy)
    assert np.array_equal(fld.data, np.asarray(_raw(HRRR_F001, decoy).values, dtype=float))


def test_selector_list_means_any_of():
    alts = [{"discipline": 0, "parameterCategory": 16, "parameterNumber": 999},
            {"discipline": 0, "parameterCategory": 16, "parameterNumber": 196}]
    assert load_model_grib2(HRRR_F001, grib_selector=alts).data.shape == (1059, 1799)


@pytest.mark.parametrize("kwargs, match", [
    (dict(grib_selector={"discipline": 0, "parameterCategory": 16, "parameterNumber": 999}), "found 0"),
    (dict(grib_selector={"discipline": 0, "parameterCategory": 16}), "found 2"),  # REFC + decoy: ambiguous
    (dict(grib_selector={"noSuchKey": 1}), "found 0"),
    (dict(var_name="NOT_A_PRESET"), "no built-in preset"),
    (dict(grib_selector=[]), "non-empty"),
])
def test_bad_selection_fails_loud(kwargs, match):
    with pytest.raises(ValueError, match=match):
        load_model_grib2(HRRR_F001, **kwargs)


def test_grid_reader_matches_field_reader():
    fld = load_model_grib2(RRFS_F001)
    lat, lon = load_grib2_grid(RRFS_F001)
    assert np.array_equal(lat, fld.lat2d) and np.array_equal(lon, fld.lon2d)


def test_bitmap_masked_points_become_nan(monkeypatch):
    """Bitmap-masked points (pygrib returns a masked array) must become NaN.
    The real HRRR/RRFS files carry no bitmap, and pygrib could not write one onto
    these complex-packed messages, so a stub pygrib file object supplies the
    masked values; everything else is load_model_grib2's own code path."""
    vals = np.ma.masked_array(np.arange(12, dtype=float).reshape(3, 4), mask=False)
    vals.mask[0, :2] = True

    class _Msg:
        messagenumber, discipline, parameterCategory, parameterNumber, typeOfLevel, level = 1, 0, 16, 196, "atmosphere", 0
        values, validDate, analDate = vals, datetime(2026, 5, 18, 1), datetime(2026, 5, 18, 0)

        def latlons(self):
            return np.zeros((3, 4)), np.zeros((3, 4))

    class _File:
        def rewind(self): pass
        def select(self, **kw): return [_Msg()]
        def close(self): pass

    monkeypatch.setattr(pygrib, "open", lambda path: _File())
    fld = load_model_grib2("stub.grib2", var_name="REFC")
    assert np.isnan(fld.data[0, :2]).all()
    assert np.array_equal(fld.data[1:], np.arange(4, 12, dtype=float).reshape(2, 4))


def test_preset_table_is_the_two_verified_refc_encodings():
    assert resolve_grib_selector("REFC") == GRIB2_FIELD_PRESETS["REFC"]
    assert {(d["parameterCategory"], d["parameterNumber"]) for d in GRIB2_FIELD_PRESETS["REFC"]} == {(16, 196), (16, 5)}


@pytest.mark.skipif(not os.path.isdir(REAL_DIR), reason="external grib2_testing/ not present")
def test_real_full_files_refc_unique_and_times_match_filename_lead():
    files = sorted(glob.glob(os.path.join(REAL_DIR, "*", "*.grib2")))
    assert len(files) == 26
    for f in files:
        init, valid = read_grib2_times(f)
        lead_h = int(os.path.basename(f).split("f")[-1][:3])
        assert init == datetime(2026, 5, 18, 0) and (valid - init).total_seconds() == lead_h * 3600, f


# --- Stage 2: manifest, drivers, config --------------------------------------

import shutil

import yaml

from py_obj.config import load_config
from py_obj.obj_core import build_model_manifest, conus_mask_east, read_object_file
from py_obj.obj_core.identify import identify_objects, precompute_grid_geometry

HRRR_DIR = os.path.join(SAMPLE, "hrrr")
RRFS_DIR = os.path.join(SAMPLE, "rrfs")
_MANIFEST_KW = dict(member_subdirs=False, stacked_members=False, var_name="REFC", lat_name=None, lon_name=None,
                    file_format="grib2")


def test_manifest_flat_grib2_times_and_loader():
    manifest, loader = build_model_manifest(input_dir=HRRR_DIR, file_pattern="hrrr_ncep_*.grib2", **_MANIFEST_KW)
    assert [e.valid_time.hour for e in manifest] == [0, 1, 2]
    assert {e.init_time for e in manifest} == {datetime(2026, 5, 18, 0)}
    assert {e.member_id for e in manifest} == {None}
    fld = loader(manifest[1].filepath)
    assert np.array_equal(fld.data, load_model_grib2(HRRR_F001).data)


def test_manifest_member_subdirs_grib2(tmp_path):
    for name, src in (("hrrr", HRRR_DIR), ("rrfs", RRFS_DIR)):
        shutil.copytree(src, tmp_path / name)
    kw = dict(_MANIFEST_KW, member_subdirs=True)
    manifest, _ = build_model_manifest(input_dir=str(tmp_path), file_pattern="*.grib2", **kw)
    assert len(manifest) == 6
    assert sorted({e.member_id for e in manifest}) == ["hrrr", "rrfs"]


def test_manifest_filename_template_overrides_grib2_times(tmp_path):
    shutil.copy(HRRR_F001, tmp_path / "x_20260518_0000_0200.grib2")  # name says valid 02Z; message says 01Z
    manifest, loader = build_model_manifest(
        input_dir=str(tmp_path), file_pattern="x_*.grib2",
        filename_time_template="x_{init:%Y%m%d_%H%M}_{valid:%H%M}.grib2", **_MANIFEST_KW)
    assert manifest[0].valid_time == datetime(2026, 5, 18, 2)
    assert loader(manifest[0].filepath).valid_time == datetime(2026, 5, 18, 2)


@pytest.mark.parametrize("kw, match", [
    (dict(stacked_members=True), "NetCDF-only"),
    (dict(file_format="grib"), "file_format must be"),
])
def test_manifest_grib2_invalid_combinations_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        build_model_manifest(input_dir=HRRR_DIR, file_pattern="*.grib2", **dict(_MANIFEST_KW, **kw))


def test_manifest_grib2_loader_rejects_extra_dim_index():
    _, loader = build_model_manifest(input_dir=HRRR_DIR, file_pattern="*.grib2", **_MANIFEST_KW)
    with pytest.raises(ValueError, match="NetCDF-only"):
        loader(HRRR_F001, extra_dim_index=0)


_LINEAR = {"linear_eccentricity_threshold": 0.8, "linear_length_threshold_km": 200.0,
           "mixed_eccentricity_threshold": 0.75, "mixed_length_threshold_km": 100.0}


def _model_section(input_dir, out_dir, **extra):
    sec = {"file_format": "grib2", "var_name": "REFC", "boundary_threshold": 45.0, "max_value_threshold": 50.0,
           "area_threshold_km2": 108.0, "input_dir": input_dir, "file_pattern": "*.grib2", "mask": "conus_east",
           "file_grouping": "init_snapshot", "object_output_dir": out_dir}
    sec.update(extra)
    return sec


@pytest.mark.parametrize("src", [HRRR_DIR, RRFS_DIR])
def test_identify_track_model_end_to_end_grib2_init_snapshot(tmp_path, src):
    """One init_snapshot file with all 3 leads; object counts identical to calling
    identify_objects directly on the GRIB2 fields with the same mask."""
    from py_obj.drivers.identify_track_model import run_one_case
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"model": _model_section(src, str(tmp_path / "obj")), "linear_classification": _LINEAR}))
    paths = run_one_case(str(cfg))
    assert [os.path.basename(p) for p in paths] == ["obj_init_20260518_000000.nc"]
    contents = read_object_file(paths[0])
    assert contents.init_time == datetime(2026, 5, 18, 0)
    assert [t.hour for t in contents.valid_times] == [0, 1, 2]

    fields = [load_model_grib2(f) for f in sorted(glob.glob(os.path.join(src, "*.grib2")))]
    mask = conus_mask_east(fields[0].lat2d, fields[0].lon2d)
    geom = precompute_grid_geometry(fields[0].lat2d, fields[0].lon2d)
    direct = [len(identify_objects(np.where(mask, 0.0, f.data), geom, 45.0, 50.0, 108.0)[1]) for f in fields]
    per_time = [int(np.sum(contents.time_index == i)) for i in range(3)]
    print(f"\n[grib2-e2e] {os.path.basename(src)} objects per lead: pipeline={per_time} direct={direct}")
    assert per_time == direct and sum(direct) > 0


@pytest.mark.parametrize("src, floor", [(HRRR_DIR, -10.0), (RRFS_DIR, -20.0)])
def test_build_histogram_model_grib2(tmp_path, src, floor):
    from py_obj.drivers.build_histogram_model import run_one_case
    from py_obj.histogram import read_histogram_file
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"histogram_model": {
        "input_dir": src, "file_format": "grib2", "var_name": "REFC", "file_pattern": "*.grib2",
        "output_dir": str(tmp_path / "hist")}}))
    h = read_histogram_file(run_one_case(str(cfg)))
    assert sorted(s.lead_hours for s in h.slices) == [0.0, 1.0, 2.0]
    floor_bin = int(np.searchsorted(h.bins, floor, side="right") - 1)
    frac_floor = np.mean([s.hist[floor_bin] / s.hist.sum() for s in h.slices])
    print(f"\n[grib2-hist] {os.path.basename(src)} fraction of pixels in the {floor} dB bin: {frac_floor:.3f}")
    assert frac_floor > 0.5  # the model's own no-echo floor dominates


def _load(tmp_path, section_name, section):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({section_name: section}))
    return load_config(str(p))


def test_config_grib2_model_needs_no_lat_lon_or_time_fields(tmp_path):
    m = _load(tmp_path, "model", _model_section("in", "out")).model
    assert m.file_format == "grib2" and m.lat_name is None and m.grib_selector is None


@pytest.mark.parametrize("section_name, extra, match", [
    ("model", {"valid_time_attr": "valid_time", "valid_time_format": "%Y"}, "NetCDF-only"),
    ("model", {"stacked_members": True}, "NetCDF-only"),
    ("model", {"var_name": "comp_dz"}, "no built-in preset"),
    ("model", {"file_format": "grb"}, "file_format must be one of"),
    ("model", {"grib_selector": []}, "non-empty"),
    ("histogram_model", {"lead_attr": "forecastHour"}, "NetCDF-only"),
])
def test_config_grib2_invalid_raise(tmp_path, section_name, extra, match):
    sec = _model_section("in", "out", **extra) if section_name == "model" else \
        dict({"input_dir": "in", "file_format": "grib2", "var_name": "REFC"}, **extra)
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, section_name, sec)


def test_config_grib_selector_rejected_for_netcdf(tmp_path):
    sec = {"file_format": "netcdf", "var_name": "x", "lat_name": "lat", "lon_name": "lon", "boundary_threshold": 40.0,
           "max_value_threshold": 45.0, "area_threshold_km2": 108.0, "input_dir": "in", "valid_time_var": "t",
           "grib_selector": {"discipline": 0}}
    with pytest.raises(ValueError, match="only applies with file_format: grib2"):
        _load(tmp_path, "model", sec)


def test_config_grib2_explicit_selector_accepted(tmp_path):
    sel = {"discipline": 0, "parameterCategory": 16, "parameterNumber": 196}
    m = _load(tmp_path, "model", _model_section("in", "out", var_name="anything", grib_selector=sel)).model
    assert m.grib_selector == sel


def test_grib2_batch_template_materializes_loads_and_runs(tmp_path):
    """Regression: expand_batch_config writes every field (unset ones as null) into
    each per-case config; GRIB2 validation must treat null as unset."""
    from py_obj.batch_config import expand_batch_config
    from py_obj.drivers.identify_track_model import run_one_case
    day = tmp_path / "archive" / "20260518"
    shutil.copytree(HRRR_DIR, day)
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({
        "cases": {"dates": ["20260518"], "init_times": ["00"], "init_time_format": "%H"},
        "model": _model_section(str(tmp_path / "archive" / "{date}"), str(tmp_path / "obj"),
                                file_pattern="hrrr_ncep_{date}{init_time}f00[0-2]*.grib2"),
        "linear_classification": _LINEAR,
    }))
    expanded = expand_batch_config(str(template), str(tmp_path / "materialized"))
    assert len(expanded.case_paths) == 1
    cfg = load_config(expanded.case_paths[0])
    assert cfg.model.file_format == "grib2" and cfg.model.init_attr is None
    paths = run_one_case(expanded.case_paths[0])
    assert [os.path.basename(p) for p in paths] == ["obj_init_20260518_000000.nc"]


# --- Stage 3: fetch_mrms + interpolation target grid from GRIB2 ------------------

def _interp_section(**extra):
    sec = {"raw_mrms_dir": "raw", "interp_mrms_dir": "out", "target_grid_file": HRRR_F001, "target_grid_format": "grib2"}
    sec.update(extra)
    return sec


def test_config_interpolation_grib2_target_grid_accepted(tmp_path):
    i = _load(tmp_path, "interpolation", _interp_section()).interpolation
    assert i.target_grid_format == "grib2" and i.target_var_name == "REFC" and i.target_grib_selector is None


@pytest.mark.parametrize("extra, match", [
    ({"target_lat_name": "xlat"}, "NetCDF-only"),
    ({"target_grid_format": "grb"}, "target_grid_format must be one of"),
    ({"target_var_name": "TMP"}, "no built-in preset"),
    ({"target_grid_format": "netcdf", "target_grib_selector": {"discipline": 0}}, "only applies with target_grid_format: grib2"),
])
def test_config_interpolation_grib2_invalid_raise(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, "interpolation", _interp_section(**extra))


def test_config_interpolation_grib2_needs_target_grid_file(tmp_path):
    sec = {"raw_mrms_dir": "raw", "interp_mrms_dir": "out", "target_grid_format": "grib2", "target_grid_sw_lat": 30.0,
           "target_grid_sw_lon": -100.0, "target_grid_dx_km": 3.0, "target_grid_nx": 10, "target_grid_ny": 10}
    with pytest.raises(ValueError, match="needs target_grid_file"):
        _load(tmp_path, "interpolation", sec)


def test_interpolate_mrms_driver_uses_grib2_message_grid(tmp_path, monkeypatch):
    """The driver hands run_batch_interpolation a GridSpec equal to the GRIB2
    message's own grid (the interpolation itself is unchanged and covered elsewhere)."""
    import py_obj.drivers.interpolate_mrms as drv
    captured = {}
    monkeypatch.setattr(drv, "run_batch_interpolation", lambda **kw: captured.update(kw) or "summary")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"interpolation": _interp_section()}))
    assert drv.run_one_case(str(cfg)) == "summary"
    raw = _raw(HRRR_F001, _RAW_REFC[HRRR_F001])
    rl, rn = raw.latlons()
    grid = captured["target_grid"]
    assert "target_grid_file" not in captured
    assert np.array_equal(grid.lat2d, rl) and np.array_equal(grid.lon2d, rn)


def test_config_fetch_mrms_grib2_model_driven(tmp_path):
    f = _load(tmp_path, "fetch_mrms", {"output_dir": "out", "model_input_dir": HRRR_DIR, "file_pattern": "*.grib2",
                                       "file_format": "grib2"}).fetch_mrms
    assert f.file_format == "grib2" and f.var_name == "REFC"
    with pytest.raises(ValueError, match="NetCDF-only"):
        _load(tmp_path, "fetch_mrms", {"output_dir": "out", "model_input_dir": HRRR_DIR, "file_format": "grib2",
                                       "lead_attr": "forecastHour"})


def test_real_fetch_mrms_model_driven_from_grib2(tmp_path):
    """Network: fetch the MRMS file nearest a GRIB2 forecast's valid time (2026-05-18 00Z)."""
    from py_obj.drivers.fetch_mrms import run_one_case
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"fetch_mrms": {
        "model_input_dir": HRRR_DIR, "file_pattern": "*.grib2", "file_format": "grib2",
        "output_dir": str(tmp_path / "mrms"), "tolerance_minutes": 2.5, "max_files": 1}}))
    summary = run_one_case(str(cfg))
    r = summary.results[0]
    print(f"\n[grib2-fetch] {r.status} {r.model_valid_time} -> {r.mrms_key}")
    assert r.status == "downloaded" and r.model_valid_time == datetime(2026, 5, 18, 0)
    assert "20260518-00" in os.path.basename(r.mrms_local_path) or "20260517-23" in os.path.basename(r.mrms_local_path)


# --- Stage 4: example configs stay valid ----------------------------------------

_CONFIGS = os.path.join(_PY_OBJ, "configs")


def test_example_grib2_configs_load_and_expand(tmp_path):
    from py_obj.batch_config import _parse_cases_section, _substitute_placeholders
    sample = load_config(os.path.join(_CONFIGS, "config_sample_grib2.yaml"))
    assert sample.model.file_format == "grib2" and sample.histogram_model.file_format == "grib2"
    for name in ("config_example_hrrr_grib2_batch.yaml", "config_example_hrrr_grib2_histogram_batch.yaml"):
        path = os.path.join(_CONFIGS, name)
        dates, init_times = _parse_cases_section(path)
        assert (dates[0], dates[-1], len(dates), init_times) == ("20260401", "20260831", 153, ["00", "06", "12", "18"])
        cfg = load_config(path)
        section = cfg.model if cfg.model is not None else cfg.histogram_model
        case = _substitute_placeholders(section, "20260518", "00")
        assert case.input_dir == "/raid/hrefv2/20260518"
        assert case.file_pattern == "hrrr_ncep_2026051800f0[0-4][0-9]*"
        assert case.file_format == "grib2" and case.var_name == "REFC" and case.mask == "conus_east"
