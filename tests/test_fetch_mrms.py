"""Validation of drivers/fetch_mrms.py -- both the pure date-expansion helper
and, since this driver had NO tests at all before this file, real end-to-end
checks against the actual public noaa-mrms-pds S3 bucket for both modes
(model-driven and date-driven). These make real network calls (no mocking
library exists in this project) -- kept small/bounded via max_files so they
stay fast, matching this project's established "validate against real data"
convention for this exact bucket (see CLAUDE.md history).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_fetch_mrms.py -v -s
"""

import os

import pytest

from python_obj.drivers.fetch_mrms import _expand_date_range, run_one_case

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
WOFS_DIR = os.path.join(SAMPLE_DATA_DIR, "wofs_case", "wofs")


def _write_config(tmp_path, cfg_dict) -> str:
    import yaml
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg_dict))
    return str(p)


def test_expand_date_range_hand_computed():
    assert _expand_date_range("20230501", "20230501") == ["20230501"]
    assert _expand_date_range("20230428", "20230502") == [
        "20230428", "20230429", "20230430", "20230501", "20230502",
    ]


def test_expand_date_range_end_before_start_raises():
    with pytest.raises(ValueError, match="before start"):
        _expand_date_range("20230502", "20230501")


def test_real_date_driven_fetch_and_idempotency(tmp_path):
    """Real, live fetch: a single real date, capped to 2 files via max_files.
    Confirms both the download path and skip_existing idempotency (running
    twice must report all already_exists the second time, not re-download)."""
    out_dir = str(tmp_path / "fetched")
    config_path = _write_config(tmp_path, {
        "fetch_mrms": {"output_dir": out_dir, "dates": ["20230501"], "max_files": 2},
    })

    summary = run_one_case(config_path)
    print(f"\n[fetch-check1] first run: {summary}")
    assert summary.n_total == 2
    assert all(r.status == "downloaded" for r in summary.results)
    assert all(r.source_date == "20230501" for r in summary.results)
    assert all(r.model_input_path is None for r in summary.results)
    for r in summary.results:
        assert os.path.isfile(r.mrms_local_path)
        assert os.path.dirname(r.mrms_local_path) == os.path.join(out_dir, "20230501")

    summary2 = run_one_case(config_path)
    print(f"[fetch-check1] second run: {summary2}")
    assert all(r.status == "already_exists" for r in summary2.results)


def test_real_date_range_driven_fetch_spans_multiple_days(tmp_path):
    """A 2-day date_range, capped to 3 total files across BOTH days combined
    (not per-day) -- confirms max_files' documented "total across the whole
    date-mode run" semantics with real data."""
    out_dir = str(tmp_path / "fetched")
    config_path = _write_config(tmp_path, {
        "fetch_mrms": {"output_dir": out_dir, "date_range": ["20230501", "20230502"], "max_files": 3},
    })

    summary = run_one_case(config_path)
    print(f"\n[fetch-check2] {summary}")
    assert summary.n_total == 3
    assert all(r.status == "downloaded" for r in summary.results)
    # all 3 came from the FIRST day in the range (max_files applied to the
    # combined listing before downloading, day 1 alone already has ~700+ files)
    assert all(r.source_date == "20230501" for r in summary.results)


def test_real_model_driven_mode_still_works(tmp_path):
    """Regression: the pre-existing model-driven mode (bundled real WoFS
    sample files) must be unaffected by splitting run_one_case() into
    _run_model_driven/_run_date_driven."""
    out_dir = str(tmp_path / "fetched_wofs")
    config_path = _write_config(tmp_path, {
        "fetch_mrms": {
            "model_input_dir": WOFS_DIR, "file_pattern": "*.nc", "output_dir": out_dir,
            "valid_time_attr": "valid_time", "valid_time_format": "%Y%m%d_%H%M%S",
            "tolerance_minutes": 2.5, "max_files": 1,
        },
    })

    summary = run_one_case(config_path)
    print(f"\n[fetch-check3] {summary}")
    assert summary.n_total == 1
    assert summary.results[0].status == "downloaded"
    assert summary.results[0].source_date is None
    assert summary.results[0].model_input_path is not None
