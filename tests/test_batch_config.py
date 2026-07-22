"""Tests for batch_config.expand_batch_config -- turning one template config
(with a "cases:" section + "{date}" placeholders) into many materialized
per-case config files, without requiring one hand-written YAML per case.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_batch_config.py -v -s
"""

import os

import netCDF4
import numpy as np
import pytest
import yaml

from python_obj.batch_config import expand_batch_config
from python_obj.config import load_config


def _write_model_file(path: str) -> None:
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("y", 4)
        ds.createDimension("x", 4)
        lat = ds.createVariable("latitude", "f8", ("y", "x"))
        lon = ds.createVariable("longitude", "f8", ("y", "x"))
        var = ds.createVariable("refl10cm_max", "f8", ("y", "x"))
        lat[:, :] = np.linspace(30, 31, 16).reshape(4, 4)
        lon[:, :] = np.linspace(-98, -97, 16).reshape(4, 4)
        var[:, :] = 25.0
        ds.initializationTime = "2023050100"
        ds.forecastHour = "1"


def _write_template(tmp_path, cases_yaml: str, extra_output_dir="output/hist_model") -> str:
    template_path = str(tmp_path / "template.yaml")
    with open(template_path, "w") as f:
        f.write(cases_yaml)
        f.write(f"""
histogram_model:
  input_dir: {tmp_path}/cases/{{date}}/mem1
  var_name: refl10cm_max
  lat_name: latitude
  lon_name: longitude
  init_attr: initializationTime
  lead_attr: forecastHour
  init_format: "%Y%m%d%H"
  output_dir: {extra_output_dir}/{{date}}
""")
    return template_path


# --- Check 1: date_range expansion, {date} substitution ------------------

def test_date_range_expands_and_substitutes_date(tmp_path):
    for date in ("20230501", "20230502", "20230503"):
        case_dir = tmp_path / "cases" / date / "mem1"
        case_dir.mkdir(parents=True)
        _write_model_file(str(case_dir / "f001.nc"))

    template_path = _write_template(tmp_path, "cases:\n  date_range: [\"20230501\", \"20230503\"]\n")
    expanded = expand_batch_config(template_path, output_dir=str(tmp_path / "materialized"))

    print(f"\n[batch-config-check1] case_paths={[os.path.basename(p) for p in expanded.case_paths]}")
    assert len(expanded.case_paths) == 3
    assert expanded.skipped_no_directory == []
    assert expanded.skipped_no_files == []

    cfg = load_config(expanded.case_paths[1])  # the 20230502 case
    assert cfg.histogram_model.input_dir == str(tmp_path / "cases" / "20230502" / "mem1")
    assert cfg.histogram_model.output_dir.endswith(os.path.join("hist_model", "20230502"))


def test_explicit_dates_list_non_contiguous(tmp_path):
    for date in ("20230501", "20230511"):
        case_dir = tmp_path / "cases" / date / "mem1"
        case_dir.mkdir(parents=True)
        _write_model_file(str(case_dir / "f001.nc"))

    template_path = _write_template(tmp_path, 'cases:\n  dates: ["20230501", "20230511"]\n')
    expanded = expand_batch_config(template_path, output_dir=str(tmp_path / "materialized"))
    assert len(expanded.case_paths) == 2


def test_dates_and_date_range_mutually_exclusive(tmp_path):
    template_path = _write_template(
        tmp_path,
        'cases:\n  dates: ["20230501"]\n  date_range: ["20230501", "20230502"]\n',
    )
    with pytest.raises(ValueError, match="exactly one of"):
        expand_batch_config(template_path, output_dir=str(tmp_path / "materialized"))


# --- Check 2: missing vs. empty case directory, distinguished -------------

def test_missing_directory_and_empty_directory_are_distinguished(tmp_path):
    # 20230501: real case, has a file
    real_dir = tmp_path / "cases" / "20230501" / "mem1"
    real_dir.mkdir(parents=True)
    _write_model_file(str(real_dir / "f001.nc"))

    # 20230502: directory exists but is empty (e.g. forecast crashed early)
    (tmp_path / "cases" / "20230502" / "mem1").mkdir(parents=True)

    # 20230503: directory does not exist at all (e.g. never initialized)
    # (nothing created)

    template_path = _write_template(tmp_path, 'cases:\n  date_range: ["20230501", "20230503"]\n')
    expanded = expand_batch_config(template_path, output_dir=str(tmp_path / "materialized"))

    print(f"\n[batch-config-check2] case_paths={len(expanded.case_paths)}, "
          f"no_directory={expanded.skipped_no_directory}, no_files={expanded.skipped_no_files}")
    assert len(expanded.case_paths) == 1
    assert "20230501" in expanded.case_paths[0]
    assert expanded.skipped_no_directory == ["20230503"]
    assert expanded.skipped_no_files == ["20230502"]


# --- Check 3: relative paths in the template resolve against the template's
# own directory BEFORE {date} substitution, not the materialized output dir --

def test_relative_template_paths_resolve_before_date_substitution(tmp_path):
    template_dir = tmp_path / "template_dir"
    template_dir.mkdir()
    case_dir = tmp_path / "template_dir" / "cases" / "20230501" / "mem1"
    case_dir.mkdir(parents=True)
    _write_model_file(str(case_dir / "f001.nc"))

    template_path = str(template_dir / "template.yaml")
    with open(template_path, "w") as f:
        f.write(
            'cases:\n  dates: ["20230501"]\n'
            "histogram_model:\n"
            "  input_dir: cases/{date}/mem1\n"  # relative to template_dir, NOT the materialized output dir
            "  var_name: refl10cm_max\n"
            "  lat_name: latitude\n"
            "  lon_name: longitude\n"
            "  init_attr: initializationTime\n"
            "  lead_attr: forecastHour\n"
            '  init_format: "%Y%m%d%H"\n'
            "  output_dir: output/hist_model/{date}\n"
        )

    materialized_dir = tmp_path / "materialized"  # deliberately a different directory than template_dir
    expanded = expand_batch_config(template_path, output_dir=str(materialized_dir))

    assert len(expanded.case_paths) == 1
    cfg = load_config(expanded.case_paths[0])
    print(f"\n[batch-config-check3] resolved input_dir={cfg.histogram_model.input_dir}")
    assert cfg.histogram_model.input_dir == str(case_dir)


# --- Check 4: real end-to-end against the bundled sample_data + run_cases_in_parallel

def test_real_end_to_end_with_run_cases_in_parallel(tmp_path):
    from python_obj.batch_runner import run_cases_in_parallel
    from python_obj.drivers.build_histogram_model import run_one_case

    sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "mpas_case", "mpas_mem1")

    # two "cases" both pointing at the same real bundled mpas_mem1/ directory
    # (structurally two cases, not scientifically distinct -- same convention
    # already used elsewhere in this repo, e.g. test_mpas/mem2 mirroring mem1)
    for date in ("20230501", "20230502"):
        case_dir = tmp_path / "cases" / date
        case_dir.mkdir(parents=True)
        for fname in os.listdir(sample_dir):
            if fname.endswith("_f00[1-3].nc") or True:  # symlink everything, filtered by file_pattern below
                os.symlink(os.path.join(sample_dir, fname), str(case_dir / fname))

    template_path = str(tmp_path / "template.yaml")
    with open(template_path, "w") as f:
        f.write(f"""
cases:
  dates: ["20230501", "20230502"]

histogram_model:
  input_dir: {tmp_path}/cases/{{date}}
  file_pattern: "*_f00[1-3].nc"
  var_name: refl10cm_max
  lat_name: latitude
  lon_name: longitude
  init_attr: initializationTime
  lead_attr: forecastHour
  init_format: "%Y%m%d%H"
  output_dir: {tmp_path}/hist_out/{{date}}
""")

    expanded = expand_batch_config(template_path, output_dir=str(tmp_path / "materialized"))
    assert len(expanded.case_paths) == 2

    summary = run_cases_in_parallel(expanded.case_paths, run_one_case, n_workers=2)
    print(f"\n[batch-config-check4] {summary}")
    assert summary.n_success == 2
    assert summary.n_failed == 0
