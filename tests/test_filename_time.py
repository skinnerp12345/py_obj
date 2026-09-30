"""Tests for py_obj.filename_time -- opt-in init/valid time derivation from a
forecast file's NAME (filename_time_template), plus init_time_offset_minutes,
threaded through io_grid, build_model_manifest, and the config layer.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj/tests/test_filename_time.py -v -s
"""

import glob
import os
from datetime import datetime

import netCDF4
import numpy as np
import pytest
import yaml

from py_obj.config import load_config
from py_obj.filename_time import compile_filename_time_template, parse_filename_times
from py_obj.obj_core import build_model_manifest
from py_obj.regrid import load_model_netcdf, read_init_time_only, read_valid_time_only

WOFS_TEMPLATE = "wofs_*_*_{init:%Y%m%d_%H%M}_{valid:%H%M}.nc"

_REPO_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TEST_WOFS_DIR = os.path.join(_REPO_PARENT, "test_wofs")
_WOFSCAST_FILE = os.path.join(_REPO_PARENT, "sfe_poster", "wofs_wofscast_WEB_62_20260506_2220_0310.nc")


# --- Parser: rollover, offset, errors ------------------------------------------

def test_same_day_valid_time():
    init, valid = parse_filename_times("wofs_ALL_08_20260518_2300_2340.nc", WOFS_TEMPLATE)
    assert init == datetime(2026, 5, 18, 23, 0)
    assert valid == datetime(2026, 5, 18, 23, 40)


def test_valid_time_wraps_past_00_utc_to_next_day():
    init, valid = parse_filename_times("/some/dir/wofs_ALL_08_20260518_2300_0140.nc", WOFS_TEMPLATE)
    assert init == datetime(2026, 5, 18, 23, 0)
    assert valid == datetime(2026, 5, 19, 1, 40)


def test_lead_zero_file():
    init, valid = parse_filename_times("wofs_ALL_00_20260518_2300_2300.nc", WOFS_TEMPLATE)
    assert init == valid == datetime(2026, 5, 18, 23, 0)


def test_valid_slot_with_full_date_is_used_as_is():
    template = "fcst_{init:%Y%m%d%H}_{valid:%Y%m%d%H}.nc"
    init, valid = parse_filename_times("fcst_2026051823_2026052105.nc", template)
    assert init == datetime(2026, 5, 18, 23)
    assert valid == datetime(2026, 5, 21, 5)  # 54 h lead, no rollover logic involved


def test_init_offset_subtracts_from_init_only():
    # WoFSCast: filename init 2220 came from the parent WoFS 2200 run.
    init, valid = parse_filename_times("wofs_wofscast_WEB_62_20260506_2220_0310.nc", WOFS_TEMPLATE, 20)
    assert init == datetime(2026, 5, 6, 22, 0)
    assert valid == datetime(2026, 5, 7, 3, 10)  # rollover decided from the RAW 22:20 init, valid never shifted


def test_init_offset_across_midnight():
    init, valid = parse_filename_times("wofs_ALL_01_20260519_0010_0015.nc", WOFS_TEMPLATE, 20)
    assert init == datetime(2026, 5, 18, 23, 50)
    assert valid == datetime(2026, 5, 19, 0, 15)


def test_non_matching_filename_raises():
    with pytest.raises(ValueError, match="does not match"):
        parse_filename_times("mpas_2023050100_f012.nc", WOFS_TEMPLATE)


def test_invalid_clock_time_in_filename_raises():
    with pytest.raises(ValueError, match="not a valid time"):
        parse_filename_times("wofs_ALL_08_20260518_2300_2575.nc", WOFS_TEMPLATE)


@pytest.mark.parametrize("bad_template, match", [
    ("wofs_*_{valid:%H%M}.nc", "must come from an init slot"),
    ("wofs_*.nc", "no '{init"),
    ("wofs_{init:%Y%m%d_%H%M}_{valid:%H%q}.nc", "unsupported strftime code"),
    ("wofs_{init:%H%M}_{valid:%H%M}.nc", "init slot must include date codes"),
    ("wofs_{init:%Y%m%d}_{init:%Y%m%d}.nc", "more than once"),
    ("wofs_{start:%Y%m%d}.nc", "malformed slot"),
])
def test_malformed_templates_raise(bad_template, match):
    with pytest.raises(ValueError, match=match):
        compile_filename_time_template(bad_template)


# --- io_grid / manifest integration (synthetic file with WRONG metadata) --------

def _write_stacked_file(path, n_members=2, valid_attr="20260518_230000", init_attr="20260518_230000"):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("ne", n_members)
        ds.createDimension("y", 4)
        ds.createDimension("x", 4)
        lat = ds.createVariable("xlat", "f8", ("y", "x"))
        lon = ds.createVariable("xlon", "f8", ("y", "x"))
        var = ds.createVariable("comp_dz", "f8", ("ne", "y", "x"))
        lat[:, :] = np.linspace(30, 31, 16).reshape(4, 4)
        lon[:, :] = np.linspace(-98, -97, 16).reshape(4, 4)
        var[:, :, :] = 25.0
        ds.valid_time = valid_attr  # deliberately stale/wrong: equal to init
        ds.init_time = init_attr


def test_template_overrides_wrong_metadata_in_readers(tmp_path):
    f = str(tmp_path / "wofs_ALL_20_20260518_2300_0040.nc")
    _write_stacked_file(f)

    # Default behavior unchanged: metadata (wrong) is used.
    assert read_valid_time_only(f, valid_time_attr="valid_time", valid_time_format="%Y%m%d_%H%M%S") == datetime(2026, 5, 18, 23)
    # Template wins over valid_time_attr when both are given.
    assert read_valid_time_only(
        f, valid_time_attr="valid_time", valid_time_format="%Y%m%d_%H%M%S", filename_time_template=WOFS_TEMPLATE,
    ) == datetime(2026, 5, 19, 0, 40)
    field = load_model_netcdf(f, "comp_dz", "xlat", "xlon", filename_time_template=WOFS_TEMPLATE, extra_dim_index=1)
    assert field.valid_time == datetime(2026, 5, 19, 0, 40)
    assert read_init_time_only(f, filename_time_template=WOFS_TEMPLATE, init_time_offset_minutes=20) == datetime(2026, 5, 18, 22, 40)


def test_manifest_stacked_and_flat_use_filename_times(tmp_path):
    for name in ("wofs_ALL_00_20260518_2300_2300.nc", "wofs_ALL_12_20260518_2300_0000.nc"):
        _write_stacked_file(str(tmp_path / name))

    manifest, _ = build_model_manifest(
        input_dir=str(tmp_path), file_pattern="wofs_*.nc", member_subdirs=False, stacked_members=True,
        var_name="comp_dz", lat_name="xlat", lon_name="xlon",
        valid_time_attr="valid_time", valid_time_format="%Y%m%d_%H%M%S",
        filename_time_template=WOFS_TEMPLATE, init_time_offset_minutes=20,
    )
    assert len(manifest) == 4  # 2 files x 2 members
    assert sorted({e.valid_time for e in manifest}) == [datetime(2026, 5, 18, 23), datetime(2026, 5, 19, 0, 0)]
    assert {e.init_time for e in manifest} == {datetime(2026, 5, 18, 22, 40)}


def test_manifest_non_matching_file_raises_rather_than_silent_none(tmp_path):
    _write_stacked_file(str(tmp_path / "wofs_ALL_00_20260518_2300_2300.nc"))
    _write_stacked_file(str(tmp_path / "wofs_badname.nc"))
    with pytest.raises(ValueError, match="does not match"):
        build_model_manifest(
            input_dir=str(tmp_path), file_pattern="wofs_*.nc", member_subdirs=False, stacked_members=True,
            var_name="comp_dz", lat_name="xlat", lon_name="xlon", filename_time_template=WOFS_TEMPLATE,
        )


# --- Config layer ----------------------------------------------------------------

def _model_section(**extra):
    base = {
        "file_format": "netcdf", "var_name": "comp_dz", "lat_name": "xlat", "lon_name": "xlon",
        "boundary_threshold": 40.0, "max_value_threshold": 45.0, "area_threshold_km2": 108.0,
        "input_dir": "models",
    }
    base.update(extra)
    return base


def _load(tmp_path, cfg_dict):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg_dict))
    return load_config(str(p))


def test_config_template_alone_satisfies_time_mode(tmp_path):
    cfg = _load(tmp_path, {
        "model": _model_section(filename_time_template=WOFS_TEMPLATE, init_time_offset_minutes=20),
        "histogram_model": {"input_dir": "models", "filename_time_template": WOFS_TEMPLATE},
        "fetch_mrms": {"output_dir": "out", "model_input_dir": "models", "filename_time_template": WOFS_TEMPLATE},
    })
    assert cfg.model.filename_time_template == WOFS_TEMPLATE
    assert cfg.model.init_time_offset_minutes == 20
    assert cfg.histogram_model.filename_time_template == WOFS_TEMPLATE
    assert cfg.histogram_model.init_time_offset_minutes is None
    assert cfg.fetch_mrms.filename_time_template == WOFS_TEMPLATE


def test_config_defaults_leave_feature_off(tmp_path):
    cfg = _load(tmp_path, {"model": _model_section(valid_time_attr="valid_time", valid_time_format="%Y%m%d_%H%M%S")})
    assert cfg.model.filename_time_template is None
    assert cfg.model.init_time_offset_minutes is None


@pytest.mark.parametrize("extra, match", [
    ({"valid_time_var": "datetime", "init_time_offset_minutes": 20}, "only used together with filename_time_template"),
    ({"filename_time_template": WOFS_TEMPLATE, "init_time_offset_minutes": 20.5}, "must be an integer"),
    ({"filename_time_template": WOFS_TEMPLATE, "init_time_offset_minutes": "20"}, "must be an integer"),
    ({"filename_time_template": "wofs_*_{valid:%H%M}.nc"}, "must come from an init slot"),
])
def test_config_rejects_bad_filename_options(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, {"model": _model_section(**extra)})


# --- Real data (external, non-bundled; skipped when absent) ------------------------

@pytest.mark.skipif(not os.path.isdir(_TEST_WOFS_DIR), reason="external test_wofs/ not present")
def test_real_wofs_filename_times_match_metadata_for_every_file():
    files = sorted(glob.glob(os.path.join(_TEST_WOFS_DIR, "wofs_ALL_*.nc")))
    assert files
    n_next_day = 0
    for f in files:
        from_meta = read_valid_time_only(f, valid_time_attr="valid_time", valid_time_format="%Y%m%d_%H%M%S")
        from_name = read_valid_time_only(f, filename_time_template=WOFS_TEMPLATE)
        init_meta = read_init_time_only(f, valid_time_attr="valid_time", valid_time_format="%Y%m%d_%H%M%S")
        init_name = read_init_time_only(f, filename_time_template=WOFS_TEMPLATE)
        assert from_name == from_meta, f
        assert init_name == init_meta, f
        n_next_day += from_name.date() > init_name.date()
    print(f"\n[filename-time-real-wofs] {len(files)} files agree with metadata; {n_next_day} cross 00 UTC")
    assert n_next_day > 0  # the rollover path is genuinely exercised on real data


@pytest.mark.skipif(not os.path.isfile(_WOFSCAST_FILE), reason="external sfe_poster/ WoFSCast file not present")
def test_real_wofscast_filename_matches_cf_time_var_and_offset_matches_parent():
    from_cf = read_valid_time_only(_WOFSCAST_FILE, valid_time_var="datetime")
    from_name = read_valid_time_only(_WOFSCAST_FILE, filename_time_template=WOFS_TEMPLATE)
    assert from_name == from_cf == datetime(2026, 5, 7, 3, 10)
    init = read_init_time_only(_WOFSCAST_FILE, filename_time_template=WOFS_TEMPLATE, init_time_offset_minutes=20)
    assert init == datetime(2026, 5, 6, 22, 0)  # the parent WoFS 2200 run named in the file's own history attr
