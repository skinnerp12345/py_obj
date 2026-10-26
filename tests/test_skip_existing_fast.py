"""Fast skip_existing path: identify_track_model (file_grouping='init_snapshot')
and build_histogram_model check an existing output from the FIRST file's times
only, before building the full manifest (which reads every file's header).
"""

import os
import shutil

import pytest
import yaml

from py_obj.config import load_config
from py_obj.drivers import build_histogram_model, identify_track_model
from py_obj.obj_core import build_model_manifest, peek_first_file

_PY_OBJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(_PY_OBJ, "configs")
HRRR = os.path.join(_PY_OBJ, "sample_data", "grib2_case", "hrrr")
_LINEAR = {"linear_eccentricity_threshold": 0.8, "linear_length_threshold_km": 200.0,
           "mixed_eccentricity_threshold": 0.75, "mixed_length_threshold_km": 100.0}
_TEMPLATE = "x_{init:%Y%m%d_%H%M}_{valid:%Y%m%d_%H%M}.grib2"


def _cfg(tmp_path, sections, name="cfg.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(sections))
    return str(p)


def _model(input_dir, out_dir, **extra):
    return dict({"file_format": "grib2", "var_name": "REFC", "input_dir": str(input_dir), "file_pattern": "*.grib2",
                 "boundary_threshold": 45.0, "max_value_threshold": 50.0, "area_threshold_km2": 108.0,
                 "file_grouping": "init_snapshot", "object_output_dir": str(out_dir)}, **extra)


def _no_full_manifest(*args, **kwargs):
    raise AssertionError("full manifest was built -- the fast skip check should have returned first")


@pytest.mark.parametrize("config_name", ["config_sample_grib2.yaml", "config_sample_mpas.yaml", "config_sample_wofs.yaml"])
def test_peek_agrees_with_full_manifest(config_name):
    """flat GRIB2, flat NetCDF and stacked-member NetCDF: the first-file peek gives
    the same first entry times and entry count as the full manifest."""
    model = load_config(os.path.join(CONFIGS, config_name)).model
    kwargs = identify_track_model._manifest_kwargs(model)
    peek = peek_first_file(**kwargs)
    manifest, _ = build_model_manifest(**kwargs)
    assert peek.n_entries == len(manifest)
    assert peek.n_files == len({e.filepath for e in manifest})
    assert peek.first_file == manifest[0].filepath
    assert (peek.valid_time, peek.init_time) == (manifest[0].valid_time, manifest[0].init_time)


def test_objects_existing_case_skipped_without_full_manifest(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, {"model": _model(HRRR, tmp_path / "obj"), "linear_classification": _LINEAR})
    written = identify_track_model.run_one_case(cfg)
    assert len(written) == 1
    mtime = os.stat(written[0]).st_mtime_ns
    monkeypatch.setattr(identify_track_model, "build_model_manifest", _no_full_manifest)
    assert identify_track_model.run_one_case(cfg) == []
    assert os.stat(written[0]).st_mtime_ns == mtime


def test_objects_directory_with_two_inits_is_not_skipped(tmp_path):
    """Only init A's output exists; the first file (init A) must not cause the
    whole directory -- which also holds init B -- to be skipped."""
    src = sorted(os.listdir(HRRR))
    archive = tmp_path / "archive"
    archive.mkdir()
    for init_hh, valid_hh in (("00", "00"), ("06", "06")):
        for k, name in enumerate(src):
            shutil.copy(os.path.join(HRRR, name),
                        archive / f"x_20260518_{init_hh}00_20260518_{int(valid_hh) + k:02d}00.grib2")
    out = tmp_path / "obj"
    only_a = _cfg(tmp_path, {"model": _model(archive, out, file_pattern="x_20260518_0000_*.grib2",
                                             filename_time_template=_TEMPLATE),
                             "linear_classification": _LINEAR}, "a.yaml")
    (path_a,) = identify_track_model.run_one_case(only_a)
    mtime_a = os.stat(path_a).st_mtime_ns

    both = _cfg(tmp_path, {"model": _model(archive, out, filename_time_template=_TEMPLATE),
                           "linear_classification": _LINEAR}, "both.yaml")
    written = identify_track_model.run_one_case(both)
    assert [os.path.basename(p) for p in written] == ["obj_init_20260518_060000.nc"]
    assert os.stat(path_a).st_mtime_ns == mtime_a


def test_histogram_existing_case_skipped_without_full_manifest(tmp_path, monkeypatch):
    hist = {"input_dir": HRRR, "file_format": "grib2", "var_name": "REFC", "file_pattern": "*.grib2",
            "output_dir": str(tmp_path / "hist")}
    cfg = _cfg(tmp_path, {"histogram_model": hist})
    out = build_histogram_model.run_one_case(cfg)
    mtime = os.stat(out).st_mtime_ns
    monkeypatch.setattr(build_histogram_model, "build_model_manifest", _no_full_manifest)
    assert build_histogram_model.run_one_case(cfg) == out
    assert os.stat(out).st_mtime_ns == mtime


def test_histogram_source_count_mismatch_falls_back_to_full_check(tmp_path, monkeypatch, capsys):
    hist = {"input_dir": HRRR, "file_format": "grib2", "var_name": "REFC", "output_dir": str(tmp_path / "hist")}
    out = build_histogram_model.run_one_case(_cfg(tmp_path, {"histogram_model": dict(hist, file_pattern="*f00[01].grib2")}, "two.yaml"))
    called = []
    real = build_histogram_model.build_model_manifest
    monkeypatch.setattr(build_histogram_model, "build_model_manifest", lambda **kw: called.append(1) or real(**kw))
    build_histogram_model.run_one_case(_cfg(tmp_path, {"histogram_model": dict(hist, file_pattern="*.grib2")}, "three.yaml"))
    assert called, "a 2-file output must not satisfy the fast check for a 3-file case"
    assert "covers 2 source files, not 3" in capsys.readouterr().out
    assert os.path.exists(out)
