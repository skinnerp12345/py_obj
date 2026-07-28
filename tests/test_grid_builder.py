"""Validation of regrid.grid_builder.build_corner_spacing_grid -- a
computed target grid (southwest corner + regular km spacing + dimensions)
as an alternative to load_target_grid()'s file-backed grid.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_grid_builder.py -v -s
"""

import numpy as np
import pyproj
import pytest

from python_obj.regrid.grid_builder import build_corner_spacing_grid


def test_shape_and_sw_corner_round_trip():
    g = build_corner_spacing_grid(sw_lat=35.0, sw_lon=-98.0, dx_km=3.0, nx=20, ny=10)
    assert g.shape == (10, 20)
    assert g.lat2d.dtype == np.float64
    assert g.lon2d.dtype == np.float64
    # the SW corner IS the (row=0, col=0) grid point (a cell center, matching
    # every other grid in this library -- corners are a separate, estimated field)
    assert abs(g.lat2d[0, 0] - 35.0) < 1e-6
    assert abs(g.lon2d[0, 0] - (-98.0)) < 1e-6


def test_spacing_verified_independently_via_fresh_pyproj_call():
    """Deliberately does NOT reuse build_corner_spacing_grid's own internals
    (mirrors test_regrid_step1.py's own "independent check" convention) --
    projects the grid's own output lat/lon back onto a fresh LCC instance and
    confirms consecutive points really are dx_km/dy_km apart."""
    sw_lat, sw_lon, dx_km, dy_km, nx, ny = 35.0, -98.0, 3.0, 5.0, 6, 4
    g = build_corner_spacing_grid(sw_lat=sw_lat, sw_lon=sw_lon, dx_km=dx_km, dy_km=dy_km, nx=nx, ny=ny)

    # build an independent LCC projection (arbitrary but fixed parameters --
    # doesn't need to match the grid's own internally-derived projection,
    # since we're only measuring relative distances between the grid's own
    # already-computed lat/lon points, not comparing against a second grid)
    proj = pyproj.Proj(proj="lcc", lat_1=30.0, lat_2=40.0, lat_0=35.0, lon_0=-98.0)
    x_m, y_m = proj(g.lon2d, g.lat2d)

    # consecutive-column spacing (east-west, dx_km) at row 0
    dx_measured_km = np.diff(x_m[0, :]) / 1000.0
    print(f"\n[grid-builder-check1] dx measured (km): {dx_measured_km}")
    assert np.allclose(dx_measured_km, dx_km, atol=0.05)

    # consecutive-row spacing (north-south, dy_km) at col 0
    dy_measured_km = np.diff(y_m[:, 0]) / 1000.0
    print(f"[grid-builder-check1] dy measured (km): {dy_measured_km}")
    assert np.allclose(dy_measured_km, dy_km, atol=0.05)


def test_dy_km_defaults_to_dx_km():
    g_explicit = build_corner_spacing_grid(sw_lat=35.0, sw_lon=-98.0, dx_km=3.0, dy_km=3.0, nx=10, ny=10)
    g_default = build_corner_spacing_grid(sw_lat=35.0, sw_lon=-98.0, dx_km=3.0, nx=10, ny=10)
    assert np.allclose(g_explicit.lat2d, g_default.lat2d)
    assert np.allclose(g_explicit.lon2d, g_default.lon2d)


def test_explicit_projection_override_changes_result():
    """Confirms the override params are actually used, not silently ignored."""
    g_auto = build_corner_spacing_grid(sw_lat=35.0, sw_lon=-98.0, dx_km=3.0, nx=20, ny=20)
    g_override = build_corner_spacing_grid(
        sw_lat=35.0, sw_lon=-98.0, dx_km=3.0, nx=20, ny=20,
        true_lat_1=25.0, true_lat_2=45.0, cen_lat=35.0, cen_lon=-98.0,
    )
    assert not np.allclose(g_auto.lat2d, g_override.lat2d) or not np.allclose(g_auto.lon2d, g_override.lon2d)


def test_two_grids_with_same_explicit_projection_share_a_consistent_frame():
    """If two grids are built with the SAME explicit projection params, a
    point that should be identical between them (here: grid B's SW corner
    equals grid A's point at (row=2, col=3)) must actually land there --
    confirms the explicit-override escape hatch produces a genuinely
    consistent, reusable projection (same rationale as
    obj_core.geometry.build_projected_coords's own documented warning about
    silently-inconsistent auto-derived projections)."""
    proj_kwargs = dict(true_lat_1=30.0, true_lat_2=40.0, cen_lat=35.0, cen_lon=-98.0)
    dx_km = 3.0
    g_a = build_corner_spacing_grid(sw_lat=35.0, sw_lon=-98.0, dx_km=dx_km, nx=10, ny=10, **proj_kwargs)

    target_lat, target_lon = g_a.lat2d[2, 3], g_a.lon2d[2, 3]
    g_b = build_corner_spacing_grid(sw_lat=target_lat, sw_lon=target_lon, dx_km=dx_km, nx=5, ny=5, **proj_kwargs)

    print(f"\n[grid-builder-check2] g_a[2,3]=({target_lat:.6f},{target_lon:.6f}), g_b[0,0]=({g_b.lat2d[0,0]:.6f},{g_b.lon2d[0,0]:.6f})")
    assert abs(g_b.lat2d[0, 0] - target_lat) < 1e-9
    assert abs(g_b.lon2d[0, 0] - target_lon) < 1e-9


def test_invalid_dimensions_raise():
    with pytest.raises(ValueError, match="nx/ny"):
        build_corner_spacing_grid(sw_lat=35.0, sw_lon=-98.0, dx_km=3.0, nx=0, ny=10)
    with pytest.raises(ValueError, match="nx/ny"):
        build_corner_spacing_grid(sw_lat=35.0, sw_lon=-98.0, dx_km=3.0, nx=10, ny=-1)
