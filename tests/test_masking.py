"""Step 3 validation: CONUS domain masking.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_masking.py -v -s
"""

import os

import numpy as np
import pyproj
import pytest
import shapely

from python_obj.obj_core import conus_mask, conus_mask_east, load_conus_boundary
from python_obj.obj_core.masking import distance_to_boundary_km

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")


# --- Check 1: synthetic exact-math check ------------------------------------

def _make_square_boundary(cen_lat: float, cen_lon: float, half_width_km: float):
    """A square polygon of EXACTLY known dimensions (half_width_km from center to
    each edge), built via the inverse of a local LCC projection -- independent of
    the real, irregular CONUS geometry, so distances can be checked exactly."""
    proj = pyproj.Proj(
        proj="lcc", lat_1=cen_lat - 2, lat_2=cen_lat + 2, lat_0=cen_lat, lon_0=cen_lon,
        a=6370000, b=6370000,
    )
    hw_m = half_width_km * 1000.0
    corners_m = [(-hw_m, -hw_m), (hw_m, -hw_m), (hw_m, hw_m), (-hw_m, hw_m)]
    corners_lonlat = [proj(x, y, inverse=True) for x, y in corners_m]
    square = shapely.geometry.Polygon(corners_lonlat)
    # densify like load_conus_boundary does -- otherwise the boundary "vertices"
    # used for nearest-neighbor distance are just the 4 corners, and e.g. the
    # center-to-boundary distance would (correctly, given only corner vertices)
    # measure to the nearest *corner* (~70.7 km) rather than the nearest *edge*
    # (50 km), which would not exercise the same approximation the real
    # production code relies on.
    return shapely.segmentize(square, max_segment_length=0.05)


def test_synthetic_square_exact_distances():
    cen_lat, cen_lon = 35.0, -97.0
    half_width_km = 50.0
    square = _make_square_boundary(cen_lat, cen_lon, half_width_km)

    # center: well inside
    inside, dist = distance_to_boundary_km(np.array([[cen_lat]]), np.array([[cen_lon]]), square)
    print(f"\n[mask-check1] center: inside={inside[0,0]}, distance_to_edge={dist[0,0]:.2f} km "
          f"(expected ~{half_width_km})")
    assert inside[0, 0]
    assert abs(dist[0, 0] - half_width_km) / half_width_km < 0.03

    # a point exactly half_width_km + 60 km north of center -> ~60 km outside the
    # north edge (well outside a 100 km buffer measured from a point 60 km beyond
    # the edge would NOT be masked; use a point far enough out to unambiguously
    # exceed the buffer instead)
    proj = pyproj.Proj(
        proj="lcc", lat_1=cen_lat - 2, lat_2=cen_lat + 2, lat_0=cen_lat, lon_0=cen_lon,
        a=6370000, b=6370000,
    )
    far_lon, far_lat = proj(0.0, (half_width_km + 150.0) * 1000.0, inverse=True)
    inside2, dist2 = distance_to_boundary_km(np.array([[far_lat]]), np.array([[far_lon]]), square)
    print(f"[mask-check1] 150 km beyond north edge: inside={inside2[0,0]}, "
          f"distance_to_edge={dist2[0,0]:.2f} km (expected ~150)")
    assert not inside2[0, 0]
    assert abs(dist2[0, 0] - 150.0) / 150.0 < 0.05

    # exactly at the 100 km buffer edge (should be right at the boundary of masked/not)
    at_buffer_lon, at_buffer_lat = proj(0.0, (half_width_km + 100.0) * 1000.0, inverse=True)
    inside3, dist3 = distance_to_boundary_km(np.array([[at_buffer_lat]]), np.array([[at_buffer_lon]]), square)
    print(f"[mask-check1] exactly 100 km beyond edge: distance_to_edge={dist3[0,0]:.2f} km (expected ~100)")
    assert abs(dist3[0, 0] - 100.0) / 100.0 < 0.05


# --- Check 2: real-geography spot checks ------------------------------------

@pytest.mark.parametrize(
    "name,lat,lon,expect_inside,expect_masked",
    [
        ("Norman, OK (deep interior)", 35.22, -97.44, True, False),
        ("Deep central Canada", 60.0, -100.0, False, True),
        ("Just north of ND/Manitoba border (~91 km)", 49.8, -100.0, False, False),
        ("Mid-Atlantic Ocean", 35.0, -50.0, False, True),
        ("North Dakota just south of border (~56 km inside)", 48.5, -100.0, True, False),
    ],
)
def test_real_geography_spot_checks(name, lat, lon, expect_inside, expect_masked):
    lat2d, lon2d = np.array([[lat]]), np.array([[lon]])
    inside, dist = distance_to_boundary_km(lat2d, lon2d, load_conus_boundary())
    masked = conus_mask(lat2d, lon2d)[0, 0]
    print(f"\n[mask-check2] {name}: inside={inside[0,0]}, distance_km={dist[0,0]:.1f}, masked={masked}")
    assert inside[0, 0] == expect_inside
    assert masked == expect_masked


# --- Check 3: real MPAS grid ------------------------------------------------

def test_conus_mask_on_real_mpas_grid():
    from python_obj.regrid import load_target_grid

    grid = load_target_grid(
        os.path.join(SAMPLE_DATA_DIR, "mpas_case/mpas_mem1/interp_mpas_3km_2023050100_mem1_f001.nc"),
        "latitude", "longitude",
    )

    mask_full = conus_mask(grid.lat2d, grid.lon2d)
    mask_east = conus_mask_east(grid.lat2d, grid.lon2d)

    frac_full = mask_full.mean()
    frac_east = mask_east.mean()
    print(f"\n[mask-check3] full-CONUS preset: {frac_full*100:.1f}% of MPAS domain masked")
    print(f"[mask-check3] eastern-2/3 preset: {frac_east*100:.1f}% of MPAS domain masked")

    assert 0.0 < frac_full < 1.0
    assert frac_east >= frac_full, "eastern preset must mask a superset of the full-CONUS preset"
    assert np.all(mask_east[mask_full]), "every full-CONUS-masked cell must also be masked in the eastern preset"
