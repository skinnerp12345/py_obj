"""Cressman interpolation (regrid/regridder.py: CressmanRegridder / build_cressman_regridder),
its batch / config / driver wiring, and that the conservative default is unchanged.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj/tests/test_cressman.py -v
"""

import glob
import os

import netCDF4
import numpy as np
import pytest
import yaml

from py_obj.config import load_config
from py_obj.regrid import GridSpec, RegridError, build_cressman_regridder, regrid_field
from py_obj.regrid import regridder as regridder_module

_PY_OBJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_MRMS = os.path.join(_PY_OBJ, "sample_data", "mpas_case", "mrms")


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(np.asarray(lon2) - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * regridder_module.EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _src_grid(lat0=35.0, lon0=-97.0, n=41, step=0.01):
    lat = lat0 + step * (np.arange(n) - n // 2)
    lon = lon0 + step * (np.arange(n) - n // 2)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return GridSpec(lat2d=lat2d, lon2d=lon2d)


def _tgt_grid(lats, lons):
    return GridSpec(lat2d=np.atleast_2d(np.asarray(lats, float)), lon2d=np.atleast_2d(np.asarray(lons, float)))


# --- the operator itself ------------------------------------------------------------

def test_weights_follow_cressman_formula(tmp_path):
    src = _src_grid()
    tgt = _tgt_grid([[35.0023]], [[-96.9988]])
    R = 4.0
    rg = build_cressman_regridder(src, tgt, R, str(tmp_path))
    d = _haversine_km(35.0023, -96.9988, src.lat2d.ravel(), src.lon2d.ravel())
    expected = np.where(d < R, (R**2 - d**2) / (R**2 + d**2), 0.0)
    got = rg.weights.toarray().ravel()
    assert np.allclose(got, expected, rtol=1e-4, atol=1e-5)
    assert (got > 0).sum() == (d < R).sum()  # every point inside the radius, none outside


def test_uniform_and_linear_fields(tmp_path):
    src = _src_grid()
    center = (src.lat2d[20, 20], src.lon2d[20, 20])
    tgt = _tgt_grid([[center[0]]], [[center[1]]])
    rg = build_cressman_regridder(src, tgt, 3.0, str(tmp_path))
    assert np.allclose(rg(np.full(src.shape, 37.5)), 37.5)
    # linear in the index direction: symmetric stencil about the center -> exact
    lin = np.add.outer(np.zeros(src.shape[0]), np.arange(src.shape[1], dtype=float))
    assert np.isclose(rg(lin)[0, 0], lin[20, 20], atol=1e-4)


def test_coincident_point_included(tmp_path):
    """A target exactly on a source point gets that point at weight 1 (distance 0)."""
    src = _src_grid()
    tgt = _tgt_grid([[src.lat2d[20, 20]]], [[src.lon2d[20, 20]]])
    rg = build_cressman_regridder(src, tgt, 2.0, str(tmp_path))
    flat_index = 20 * src.shape[1] + 20
    assert np.isclose(rg.weights[0, flat_index], 1.0)


def test_unmapped_targets_filled_by_regrid_field(tmp_path):
    src = _src_grid()
    tgt = _tgt_grid([[35.0, 40.0]], [[-97.0, -90.0]])  # second target far from every source point
    rg = build_cressman_regridder(src, tgt, 3.0, str(tmp_path))
    raw = rg(np.full(src.shape, 30.0))
    assert np.isclose(raw[0, 0], 30.0) and np.isnan(raw[0, 1])
    with pytest.warns(UserWarning, match="no\nvalid source|no valid source"):
        out, report = regrid_field(rg, np.full(src.shape, 30.0), fill_value=-999.0)
    assert out[0, 1] == -999.0 and report.n_outside_source_bbox == 1


def test_missing_sources_excluded(tmp_path):
    src = _src_grid()
    tgt = _tgt_grid([[35.0]], [[-97.0]])
    rg = build_cressman_regridder(src, tgt, 4.0, str(tmp_path))
    rng = np.random.default_rng(0)
    data = rng.uniform(10, 50, src.shape)
    missing = rng.random(src.shape) < 0.3
    data[missing] = -999.0
    out, _ = regrid_field(rg, data, missing_value=-999.0)
    w = rg.weights.toarray().ravel()
    valid = ~missing.ravel()
    expected = (w[valid] * data.ravel()[valid]).sum() / w[valid].sum()
    assert np.isclose(out[0, 0], expected) and out[0, 0] > 0  # -999 never averaged in


def test_mostly_missing_target_is_filled(tmp_path):
    src = _src_grid()
    tgt = _tgt_grid([[35.0]], [[-97.0]])
    rg = build_cressman_regridder(src, tgt, 4.0, str(tmp_path))
    data = np.full(src.shape, -999.0)
    data[0, 0] = 40.0  # far corner: outside the radius, so the valid weighted fraction is 0
    with pytest.warns(UserWarning):
        out, _ = regrid_field(rg, data, fill_value=-999.0, missing_value=-999.0)
    assert out[0, 0] == -999.0


def test_weights_cached_and_reused(tmp_path, monkeypatch):
    src = _src_grid()
    tgt = _tgt_grid([[35.0]], [[-97.0]])
    first = build_cressman_regridder(src, tgt, 3.0, str(tmp_path))
    cached = glob.glob(str(tmp_path / "cressman_r3km__*.npz"))
    assert len(cached) == 1
    mtime = os.stat(cached[0]).st_mtime_ns

    import scipy.spatial

    def _no_rebuild(*a, **k):
        raise AssertionError("weights were rebuilt instead of loaded from the cache")
    monkeypatch.setattr(scipy.spatial, "cKDTree", _no_rebuild)
    second = build_cressman_regridder(src, tgt, 3.0, str(tmp_path))
    assert os.stat(cached[0]).st_mtime_ns == mtime
    assert (first.weights != second.weights).nnz == 0


@pytest.mark.parametrize("radius", [0, -1.0, None, float("nan")])
def test_bad_radius_raises(tmp_path, radius):
    with pytest.raises(RegridError, match="radius"):
        build_cressman_regridder(_src_grid(), _tgt_grid([[35.0]], [[-97.0]]), radius, str(tmp_path))


def test_size_guard_raises_before_building(tmp_path):
    with pytest.raises(RegridError, match="too large"):
        build_cressman_regridder(_src_grid(), _tgt_grid([[35.0]], [[-97.0]]), 4.0, str(tmp_path), max_nnz=10)
    assert not glob.glob(str(tmp_path / "*.npz"))


def test_no_overlap_raises(tmp_path):
    with pytest.raises(RegridError, match="no source point"):
        build_cressman_regridder(_src_grid(), _tgt_grid([[45.0]], [[-80.0]]), 3.0, str(tmp_path))


# --- config -------------------------------------------------------------------------

def _interp_cfg(tmp_path, **extra):
    sec = {"raw_mrms_dir": RAW_MRMS, "interp_mrms_dir": str(tmp_path / "out"), "target_grid_sw_lat": 42.0,
           "target_grid_sw_lon": -86.0, "target_grid_dx_km": 3.0, "target_grid_nx": 50, "target_grid_ny": 50,
           "n_workers": 2, "weight_cache_dir": str(tmp_path / "wc")}
    sec.update(extra)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({"interpolation": sec}))
    return str(p)


def test_config_default_and_valid_cressman(tmp_path):
    c = load_config(_interp_cfg(tmp_path)).interpolation
    assert c.method == "conservative" and c.cressman_radius_km is None
    c = load_config(_interp_cfg(tmp_path, method=None)).interpolation  # null = unset (materialized batch configs)
    assert c.method == "conservative"
    c = load_config(_interp_cfg(tmp_path, method="cressman", cressman_radius_km=3.5)).interpolation
    assert c.method == "cressman" and c.cressman_radius_km == 3.5


@pytest.mark.parametrize("extra, match", [
    ({"method": "cressman"}, "requires cressman_radius_km"),
    ({"method": "cressman", "cressman_radius_km": 0}, "requires cressman_radius_km"),
    ({"method": "cressman", "cressman_radius_km": "3"}, "requires cressman_radius_km"),
    ({"cressman_radius_km": 3.0}, "only used with method 'cressman'"),
    ({"method": "bilinear"}, "must be 'conservative' or 'cressman'"),
])
def test_config_rejects_bad_settings(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        load_config(_interp_cfg(tmp_path, **extra))


# --- batch / driver -----------------------------------------------------------------

def test_batch_rejects_radius_beyond_crop_buffer(tmp_path):
    from py_obj.regrid import build_corner_spacing_grid, run_batch_interpolation
    grid = build_corner_spacing_grid(sw_lat=42.0, sw_lon=-86.0, dx_km=3.0, nx=10, ny=10)
    with pytest.raises(ValueError, match="crop buffer"):
        run_batch_interpolation(RAW_MRMS, str(tmp_path / "o"), target_grid=grid, method="cressman",
                                cressman_radius_km=40.0, weight_cache_dir=str(tmp_path / "wc"), n_workers=1)
    with pytest.raises(ValueError, match="only valid with method='cressman'"):
        run_batch_interpolation(RAW_MRMS, str(tmp_path / "o"), target_grid=grid, cressman_radius_km=3.0,
                                weight_cache_dir=str(tmp_path / "wc"), n_workers=1)


def test_driver_cressman_end_to_end(tmp_path):
    from py_obj.drivers.interpolate_mrms import run_one_case
    cfg = _interp_cfg(tmp_path, method="cressman", cressman_radius_km=3.0)
    first = run_one_case(cfg)
    assert first.n_success == 3 and first.n_failed == 0
    files = sorted(glob.glob(str(tmp_path / "out" / "**" / "*.nc"), recursive=True))
    assert len(files) == 3
    with netCDF4.Dataset(files[0]) as d:
        assert "refl_cress" in d.variables and "refl_consv" not in d.variables
        assert d.interpolation_method == "cressman" and float(d.cressman_radius_km) == 3.0
        v = np.asarray(d["refl_cress"][:])
        assert v.shape == (50, 50) and np.nanmax(v) > 0
    assert run_one_case(cfg).n_skipped_existing == 3  # resumable like the conservative path


def test_conservative_default_output_unchanged_except_method_attribute(tmp_path):
    from py_obj.drivers.interpolate_mrms import run_one_case
    run_one_case(_interp_cfg(tmp_path))
    f = sorted(glob.glob(str(tmp_path / "out" / "**" / "*.nc"), recursive=True))[0]
    with netCDF4.Dataset(f) as d:
        assert "refl_consv" in d.variables and d.interpolation_method == "conservative"
        assert not hasattr(d, "cressman_radius_km")
