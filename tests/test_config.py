"""Config layer validation: one unified config file, five independently
optional top-level sections.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_config.py -v -s
"""

import glob
import os
from datetime import datetime, timedelta

import netCDF4
import numpy as np
import pytest
import yaml

from python_obj.config import (
    Config,
    FetchMrmsConfig,
    HistogramModelConfig,
    InterpolationConfig,
    LinearClassificationConfig,
    MatchingConfig,
    ModelConfig,
    ObservationConfig,
    load_config,
    require_section,
)
from python_obj.obj_core import SeriesEntry, conus_mask_east, read_object_file, run_object_id_series
from python_obj.regrid import load_model_netcdf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON_OBJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PYTHON_OBJ_DIR, "configs")
REAL_CONFIG_PATH = os.path.join(CONFIGS_DIR, "config.yaml")
MPAS_MEM_DIR = os.path.join(REPO_ROOT, "test_mpas")   # unchanged -- test data doesn't move
# config.yaml/config_ensemble.yaml deliberately still point at this larger,
# non-bundled 2-member (mem1/mem2) local dataset (see python_obj/sample_data/README.md,
# "What's NOT bundled here") -- the 2 checks below that validate config.yaml's
# own real ensemble behavior against it skip cleanly (not fail) when it's
# absent, e.g. on a fresh clone that only has the bundled sample_data/.
_SKIP_NO_LOCAL_MPAS_ENSEMBLE = pytest.mark.skipif(
    not os.path.isdir(os.path.join(MPAS_MEM_DIR, "mem1")),
    reason="requires the larger local test_mpas/mem1,mem2/ dataset (not bundled -- see sample_data/README.md)",
)


def _resolved(relative: str) -> str:
    """Mirror load_config()'s own path-resolution (relative to configs/), so
    hand-built comparisons match exactly rather than comparing bare strings."""
    return os.path.normpath(os.path.join(CONFIGS_DIR, relative))


# --- Check 1: round-trip / defaults ----------------------------------------

def test_load_real_config_exact_values():
    cfg = load_config(REAL_CONFIG_PATH)

    assert cfg.interpolation.raw_mrms_dir == _resolved("../../test_mrms")
    assert cfg.interpolation.interp_mrms_dir == _resolved("output/interp_mrms")
    assert cfg.interpolation.n_workers == 8

    assert cfg.observations.boundary_threshold == 40.0
    assert cfg.observations.max_value_threshold == 45.0
    assert cfg.observations.area_threshold_km2 == 108.0
    assert cfg.observations.mask == "conus_east"
    assert cfg.observations.file_grouping == "single"
    assert cfg.observations.interp_mrms_dir == _resolved("output/interp_mrms")
    assert cfg.observations.object_output_dir == _resolved("output/obj_mrms")

    assert cfg.model.boundary_threshold == 45.0
    assert cfg.model.max_value_threshold == 50.2
    assert cfg.model.file_grouping == "ensemble_snapshot"
    assert cfg.model.lead_units == "hours"
    assert cfg.model.input_dir == _resolved("../../test_mpas")
    assert cfg.model.member_subdirs is True
    assert cfg.model.object_output_dir == _resolved("output/obj_model_ensemble")

    assert cfg.matching.max_boundary_disp_km == 40.0
    assert cfg.matching.max_centroid_disp_km == 40.0
    assert cfg.matching.ti_threshold == 0.2
    assert cfg.matching.max_time_offset_minutes == 5.0
    assert cfg.matching.truth_object_dir == _resolved("output/obj_mrms")
    assert cfg.matching.forecast_object_dir == _resolved("output/obj_model_ensemble")
    assert cfg.matching.output_dir == _resolved("output/matches")

    assert cfg.linear_classification.linear_eccentricity_threshold == 0.8
    assert cfg.linear_classification.linear_length_threshold_km == 200.0
    assert cfg.linear_classification.mixed_eccentricity_threshold == 0.75
    assert cfg.linear_classification.mixed_length_threshold_km == 100.0
    print(f"\n[config-check1] real config loaded: interpolation={cfg.interpolation}\nobs={cfg.observations}"
          f"\nmodel={cfg.model}\nmatching={cfg.matching}\nlinear_classification={cfg.linear_classification}")


def test_defaults_fill_in_when_optional_fields_omitted(tmp_path):
    minimal = {
        "observations": {
            "file_format": "netcdf", "var_name": "x", "lat_name": "lat", "lon_name": "lon",
            "boundary_threshold": 10.0, "max_value_threshold": 20.0, "area_threshold_km2": 50.0,
            "interp_mrms_dir": "some/dir",
        },
        "model": {
            "file_format": "netcdf", "var_name": "y", "lat_name": "lat", "lon_name": "lon",
            "boundary_threshold": 10.0, "max_value_threshold": 20.0, "area_threshold_km2": 50.0,
            "init_attr": "init", "lead_attr": "lead", "init_format": "%Y%m%d%H",
            "input_dir": "some/other/dir",
        },
        "matching": {
            "max_boundary_disp_km": 40.0, "max_centroid_disp_km": 40.0, "ti_threshold": 0.2,
            "truth_object_dir": "truth/dir", "forecast_object_dir": "forecast/dir",
        },
        "linear_classification": {
            "linear_eccentricity_threshold": 0.8, "linear_length_threshold_km": 200.0,
            "mixed_eccentricity_threshold": 0.75, "mixed_length_threshold_km": 100.0,
        },
    }
    p = tmp_path / "minimal.yaml"
    p.write_text(yaml.dump(minimal))

    cfg = load_config(str(p))
    print(f"\n[config-check1] defaults: mask={cfg.observations.mask!r} track={cfg.observations.track!r} "
          f"track_distance_km={cfg.observations.track_distance_km!r} file_grouping={cfg.observations.file_grouping!r} "
          f"lead_units={cfg.model.lead_units!r} max_time_offset_minutes={cfg.matching.max_time_offset_minutes!r} "
          f"object_output_dir={cfg.observations.object_output_dir!r} "
          f"member_subdirs={cfg.model.member_subdirs!r} file_pattern={cfg.model.file_pattern!r}")
    assert cfg.observations.mask == "none"
    assert cfg.observations.track is False
    assert cfg.observations.track_distance_km == 0.0
    assert cfg.observations.file_grouping == "single"
    assert cfg.observations.object_output_dir == os.path.join(str(tmp_path), "output/obj_mrms")
    assert cfg.model.lead_units == "hours"
    assert cfg.model.member_subdirs is False
    assert cfg.model.file_pattern == "*.nc"
    assert cfg.model.object_output_dir == os.path.join(str(tmp_path), "output/obj_model")
    assert cfg.matching.max_time_offset_minutes == 5.0
    assert cfg.matching.output_dir == os.path.join(str(tmp_path), "output/matches")
    assert cfg.matching.file_pattern == "*.nc"


# --- Check 2: clear errors, not guesses -------------------------------------

def test_missing_required_field_raises_clear_error(tmp_path):
    bad = {
        "observations": {
            "file_format": "netcdf", "var_name": "x", "lat_name": "lat", "lon_name": "lon",
            # boundary_threshold deliberately omitted
            "max_value_threshold": 20.0, "area_threshold_km2": 50.0,
            "interp_mrms_dir": "some/dir",
        },
        "model": {
            "file_format": "netcdf", "var_name": "y", "lat_name": "lat", "lon_name": "lon",
            "boundary_threshold": 10.0, "max_value_threshold": 20.0, "area_threshold_km2": 50.0,
            "init_attr": "init", "lead_attr": "lead", "init_format": "%Y%m%d%H",
            "input_dir": "some/other/dir",
        },
        "matching": {
            "max_boundary_disp_km": 40.0, "max_centroid_disp_km": 40.0, "ti_threshold": 0.2,
            "truth_object_dir": "truth/dir", "forecast_object_dir": "forecast/dir",
        },
        "linear_classification": {
            "linear_eccentricity_threshold": 0.8, "linear_length_threshold_km": 200.0,
            "mixed_eccentricity_threshold": 0.75, "mixed_length_threshold_km": 100.0,
        },
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(bad))

    with pytest.raises(ValueError) as excinfo:
        load_config(str(p))
    print(f"\n[config-check2] error message: {excinfo.value}")
    assert "boundary_threshold" in str(excinfo.value)
    assert "observations" in str(excinfo.value)


def test_section_present_but_incomplete_still_errors(tmp_path):
    """A section that IS present but missing one of its own required fields
    must still raise -- "optional at the top level" (a section can be
    entirely absent) and "validated once present" (a present section is
    still fully checked) are independent behaviors."""
    bad = {"matching": {"max_boundary_disp_km": 40.0}}  # missing everything else
    p = tmp_path / "incomplete_matching.yaml"
    p.write_text(yaml.dump(bad))

    with pytest.raises(ValueError) as excinfo:
        load_config(str(p))
    print(f"\n[config-check2b] error message: {excinfo.value}")
    assert "matching" in str(excinfo.value)


def test_missing_section_returns_none(tmp_path):
    """A section absent from the YAML entirely is NOT an error -- it's
    legitimate for a config to only populate what a given problem needs.
    Other, present sections still load normally."""
    partial = {
        "observations": {
            "file_format": "netcdf", "var_name": "x", "lat_name": "lat", "lon_name": "lon",
            "boundary_threshold": 10.0, "max_value_threshold": 20.0, "area_threshold_km2": 50.0,
            "interp_mrms_dir": "some/dir",
        },
        "matching": {
            "max_boundary_disp_km": 40.0, "max_centroid_disp_km": 40.0, "ti_threshold": 0.2,
            "truth_object_dir": "truth/dir", "forecast_object_dir": "forecast/dir",
        },
        # model, interpolation, linear_classification deliberately omitted
    }
    p = tmp_path / "partial.yaml"
    p.write_text(yaml.dump(partial))

    cfg = load_config(str(p))
    print(f"\n[config-check2c] model={cfg.model!r} interpolation={cfg.interpolation!r} "
          f"linear_classification={cfg.linear_classification!r}")
    assert cfg.model is None
    assert cfg.interpolation is None
    assert cfg.linear_classification is None
    assert cfg.observations is not None
    assert cfg.matching is not None


def test_all_sections_absent_yields_all_none(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")  # yaml.safe_load returns None for a fully empty file

    cfg = load_config(str(p))
    assert cfg == Config()
    print(f"\n[config-check2d] empty config: {cfg}")


def test_require_section_helper_raises_on_none():
    with pytest.raises(ValueError) as excinfo:
        require_section(None, "model", "/some/path.yaml")
    print(f"\n[config-check2e] error message: {excinfo.value}")
    assert "model" in str(excinfo.value)
    assert "/some/path.yaml" in str(excinfo.value)

    sentinel = object()
    assert require_section(sentinel, "model", "/some/path.yaml") is sentinel


# --- Check 3: real regeneration with corrected thresholds -------------------

@_SKIP_NO_LOCAL_MPAS_ENSEMBLE
def test_regenerate_mpas_ensemble_with_correct_thresholds(tmp_path):
    cfg = load_config(REAL_CONFIG_PATH).model
    assert (cfg.boundary_threshold, cfg.max_value_threshold) == (45.0, 50.2)

    loader = lambda fp: load_model_netcdf(fp, varname=cfg.var_name, lat_name=cfg.lat_name, lon_name=cfg.lon_name)

    manifest = []
    for member_id in ("mem1", "mem2"):
        files = sorted(glob.glob(os.path.join(MPAS_MEM_DIR, member_id, "*.nc")))
        for f in files:
            fhr = int(f.split("_f")[-1].split(".nc")[0])
            if fhr > 12:
                continue
            manifest.append(SeriesEntry(valid_time=loader(f).valid_time, filepath=f, member_id=member_id))

    first = loader(manifest[0].filepath)
    mask = conus_mask_east(first.lat2d, first.lon2d) if cfg.mask == "conus_east" else None

    out = run_object_id_series(
        manifest, lambda entry: loader(entry.filepath),
        thresh_1=cfg.boundary_threshold, thresh_2=cfg.max_value_threshold, area_thresh_km2=cfg.area_threshold_km2,
        output_dir=str(tmp_path / "mpas_ensemble_corrected"),
        file_grouping=cfg.file_grouping, track_in_time=cfg.track, mask=mask,
        init_time=datetime(2023, 5, 1, 0, 0, 0),
    )
    assert len(out) == 13

    n_objects_corrected = sum(len(read_object_file(p).objects) for p in out)

    # compare against a freshly-generated run using the earlier (wrong-threshold,
    # wofscast values) pair -- regenerated fresh rather than read from whatever
    # happens to be on disk, so this doesn't depend on/break on stale files from
    # a prior schema version (e.g. real test_obj/mpas_ensemble/ files written
    # before is_linear existed)
    out_old = run_object_id_series(
        manifest, lambda entry: loader(entry.filepath),
        thresh_1=39.8, thresh_2=45.6, area_thresh_km2=cfg.area_threshold_km2,
        output_dir=str(tmp_path / "mpas_ensemble_wofscast_values"),
        file_grouping=cfg.file_grouping, track_in_time=cfg.track, mask=mask,
        init_time=datetime(2023, 5, 1, 0, 0, 0),
    )
    n_objects_old = sum(len(read_object_file(p).objects) for p in out_old)
    print(f"\n[config-check3] object counts -- old (wofscast 39.8/45.6): {n_objects_old}, "
          f"corrected (mpas 45.0/50.2): {n_objects_corrected}")
    assert n_objects_corrected != n_objects_old, (
        "expected object counts to differ between the wrong-threshold and corrected runs"
    )


# --- Check 4: hand-built dataclass vs YAML round-trip -----------------------

def test_hand_built_config_matches_yaml_round_trip():
    hand_built = Config(
        interpolation=InterpolationConfig(
            raw_mrms_dir=_resolved("../../test_mrms"),
            interp_mrms_dir=_resolved("output/interp_mrms"),
            target_grid_file=_resolved("../../test_mpas/mem1/interp_mpas_3km_2023050100_mem1_f000.nc"),
            target_lat_name="latitude", target_lon_name="longitude",
            n_workers=8, weight_cache_dir=_resolved("output/weight_cache"),
            date_range=("20230401", "20230402"), max_files=None,
        ),
        observations=ObservationConfig(
            file_format="netcdf", var_name="refl_consv", lat_name="lat", lon_name="lon",
            boundary_threshold=40.0, max_value_threshold=45.0, area_threshold_km2=108.0,
            interp_mrms_dir=_resolved("output/interp_mrms"),
            mask="conus_east", track=False, track_distance_km=0.0, file_grouping="single",
            object_output_dir=_resolved("output/obj_mrms"),
        ),
        model=ModelConfig(
            file_format="netcdf", var_name="refl10cm_max", lat_name="latitude", lon_name="longitude",
            boundary_threshold=45.0, max_value_threshold=50.2, area_threshold_km2=108.0,
            init_attr="initializationTime", lead_attr="forecastHour", init_format="%Y%m%d%H",
            input_dir=_resolved("../../test_mpas"),
            mask="conus_east", track=False, track_distance_km=0.0, file_grouping="ensemble_snapshot",
            lead_units="hours", member_subdirs=True, file_pattern="*_f0[01]?.nc",
            object_output_dir=_resolved("output/obj_model_ensemble"),
        ),
        matching=MatchingConfig(
            max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2,
            truth_object_dir=_resolved("output/obj_mrms"),
            forecast_object_dir=_resolved("output/obj_model_ensemble"),
            max_time_offset_minutes=5.0, output_dir=_resolved("output/matches"), file_pattern="*.nc",
        ),
        linear_classification=LinearClassificationConfig(
            linear_eccentricity_threshold=0.8, linear_length_threshold_km=200.0,
            mixed_eccentricity_threshold=0.75, mixed_length_threshold_km=100.0,
        ),
        fetch_mrms=FetchMrmsConfig(
            model_input_dir=_resolved("../../test_wofs"),
            output_dir=_resolved("output/fetched_mrms_wofs"),
            file_pattern="*.nc",
            valid_time_attr="valid_time", valid_time_format="%Y%m%d_%H%M%S",
            tolerance_minutes=2.5, s3_bucket="noaa-mrms-pds",
            mrms_product="MergedReflectivityQCComposite_00.50",
            mirror_subdirs=True, skip_existing=True, max_files=None,
        ),
        histogram_model=HistogramModelConfig(
            input_dir=_resolved("../../test_mpas"),
            var_name="refl10cm_max", lat_name="latitude", lon_name="longitude",
            member_subdirs=True, file_pattern="*.nc",
            init_attr="initializationTime", lead_attr="forecastHour", init_format="%Y%m%d%H",
            output_dir=_resolved("output/hist_model_mpas"),
        ),
    )

    loaded = load_config(REAL_CONFIG_PATH)
    print(f"\n[config-check4] hand-built == loaded: {hand_built == loaded}")
    assert hand_built == loaded


# --- Check 5: new path-resolution behavior, and section independence -------

def test_relative_paths_resolved_against_config_file_directory(tmp_path):
    cfg_dict = {
        "observations": {
            "file_format": "netcdf", "var_name": "x", "lat_name": "lat", "lon_name": "lon",
            "boundary_threshold": 10.0, "max_value_threshold": 20.0, "area_threshold_km2": 50.0,
            "interp_mrms_dir": "../somewhere",
        },
    }
    p = tmp_path / "sub" / "config.yaml"
    p.parent.mkdir()
    p.write_text(yaml.dump(cfg_dict))

    cfg = load_config(str(p))
    expected = os.path.normpath(os.path.join(str(tmp_path), "somewhere"))
    print(f"\n[config-check5] resolved interp_mrms_dir={cfg.observations.interp_mrms_dir!r}, expected={expected!r}")
    assert cfg.observations.interp_mrms_dir == expected
    assert os.path.isabs(cfg.observations.interp_mrms_dir)


def test_interpolation_and_observations_interp_mrms_dir_are_independent_fields(tmp_path):
    """interp_mrms_dir appears in both interpolation: and observations: as
    plain, independent fields (no cross-section derivation) -- a config
    giving them different values must round-trip both independently."""
    cfg_dict = {
        "interpolation": {
            "raw_mrms_dir": "raw", "interp_mrms_dir": "interp_output_A", "target_grid_file": "grid.nc",
        },
        "observations": {
            "file_format": "netcdf", "var_name": "x", "lat_name": "lat", "lon_name": "lon",
            "boundary_threshold": 10.0, "max_value_threshold": 20.0, "area_threshold_km2": 50.0,
            "interp_mrms_dir": "interp_output_B",
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg_dict))

    cfg = load_config(str(p))
    assert cfg.interpolation.interp_mrms_dir.endswith("interp_output_A")
    assert cfg.observations.interp_mrms_dir.endswith("interp_output_B")
    assert cfg.interpolation.interp_mrms_dir != cfg.observations.interp_mrms_dir


# --- Check 6: lead_units generalization -------------------------------------

@_SKIP_NO_LOCAL_MPAS_ENSEMBLE
def test_lead_units_hours_real_mpas():
    f = os.path.join(MPAS_MEM_DIR, "mem1", "interp_mpas_3km_2023050100_mem1_f012.nc")
    field = load_model_netcdf(f, varname="refl10cm_max", lead_units="hours")
    assert field.valid_time == datetime(2023, 5, 1, 12, 0, 0)
    print(f"\n[config-check6] real MPAS f012, lead_units=hours -> valid_time={field.valid_time}")


def test_lead_units_minutes_synthetic(tmp_path):
    """Synthetic file standing in for a 5-minute-cadence model (e.g. WoFS) whose
    lead-time attribute is stored in minutes, not hours -- no real such test data
    exists in this repo yet."""
    p = str(tmp_path / "synthetic_5min.nc")
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("y", 4)
        ds.createDimension("x", 4)
        lat = ds.createVariable("latitude", "f8", ("y", "x"))
        lon = ds.createVariable("longitude", "f8", ("y", "x"))
        var = ds.createVariable("comp_dz", "f8", ("y", "x"))
        lat[:, :] = np.linspace(30, 31, 16).reshape(4, 4)
        lon[:, :] = np.linspace(-98, -97, 16).reshape(4, 4)
        var[:, :] = 0.0
        ds.initializationTime = "2023050100"
        ds.forecast_lead_minutes = "35"

    field = load_model_netcdf(
        p, varname="comp_dz", lat_name="latitude", lon_name="longitude",
        lead_attr="forecast_lead_minutes", lead_units="minutes",
    )
    print(f"[config-check6] synthetic 5-min model, lead_units=minutes, forecast_lead_minutes=35 "
          f"-> valid_time={field.valid_time}")
    assert field.valid_time == datetime(2023, 5, 1, 0, 35, 0)
