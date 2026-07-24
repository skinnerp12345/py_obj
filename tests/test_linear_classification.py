"""Linear vs. cellular object classification, adapted from
python_base/QLCS_Obj_ID.py (simplified to one eccentricity+length threshold
combination, computed in physical km-space rather than pixel-index space).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_linear_classification.py -v -s
"""

import os

import numpy as np

from python_obj.obj_core import (
    IdentificationResult,
    build_projected_coords,
    identify_objects,
    precompute_grid_geometry,
    principal_axis_km,
    read_object_file,
    write_object_file,
)
from python_obj.regrid import load_model_netcdf

MPAS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "mpas_case", "mpas_mem1")


def _synthetic_grid(ny=60, nx=60, lat0=30.0, lon0=-100.0, step=0.05):
    lat = lat0 + np.arange(ny) * step
    lon = lon0 + np.arange(nx) * step
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lat2d, lon2d


# --- Check 1: principal_axis_km correctness, synthetic ---------------------

def test_principal_axis_km_line_vs_blob():
    # isotropic 1 km grid: x_km == col index, y_km == row index directly
    ny, nx = 30, 30
    rows_line = np.full(20, 10)
    cols_line = np.arange(5, 25)
    maj_line, ecc_line = principal_axis_km(cols_line.astype(float), rows_line.astype(float))
    print(f"\n[linear-check1] line: major_axis_length_km={maj_line:.2f} eccentricity_km={ecc_line:.3f}")
    assert abs(maj_line - 23.06) < 0.5
    assert ecc_line > 0.99

    yy, xx = np.mgrid[0:ny, 0:nx]
    blob_mask = (yy - 15) ** 2 + (xx - 15) ** 2 <= 25
    rows_blob, cols_blob = np.where(blob_mask)
    maj_blob, ecc_blob = principal_axis_km(cols_blob.astype(float), rows_blob.astype(float))
    print(f"[linear-check1] blob: major_axis_length_km={maj_blob:.2f} eccentricity_km={ecc_blob:.3f}")
    assert ecc_blob < 0.2


# --- Check 2: anisotropy regression --------------------------------------

def test_principal_axis_km_anisotropy_regression():
    """Same pixel-index shape (row/col extent) placed at 20N vs 55N must give
    DIFFERENT major_axis_length_km, reflecting the real km/gridpoint difference
    already established in Step 2 -- unlike a naive pixel-count*scalar approach."""
    lon = np.arange(0, 30) * 0.01 - 130.0
    rows = np.full(20, 5)
    cols = np.arange(0, 20)

    lat_low = 20.0 + np.arange(30) * 0.01
    lon2d_low, lat2d_low = np.meshgrid(lon, lat_low)
    x2d_low, y2d_low = build_projected_coords(lat2d_low, lon2d_low)
    maj_low, _ = principal_axis_km(x2d_low[rows, cols], y2d_low[rows, cols])

    lat_high = 55.0 + np.arange(30) * 0.01
    lon2d_high, lat2d_high = np.meshgrid(lon, lat_high)
    x2d_high, y2d_high = build_projected_coords(lat2d_high, lon2d_high)
    maj_high, _ = principal_axis_km(x2d_high[rows, cols], y2d_high[rows, cols])

    print(f"\n[linear-check2] same pixel shape: major_axis_length_km at 20N={maj_low:.2f}, "
          f"at 55N={maj_high:.2f}, ratio={maj_high / maj_low:.3f}")
    assert maj_low != maj_high
    assert maj_high / maj_low < 0.9  # meaningfully shorter at high latitude, ~0.6-0.65 expected


# --- Check 3: three-tier classification logic, synthetic -------------------
#
# Grid spacing here is ~4.8-5.6 km/px (empirically confirmed), so a 1-px-wide
# horizontal line of N points has major_axis_length_km roughly proportional to
# N and eccentricity ~1.0 regardless of length -- line lengths below were
# picked to land cleanly in each tier under the default thresholds (linear:
# ecc>0.8 and length>200km; mixed: ecc>0.75 and length>100km, checked only if
# the linear tier fails).

def test_is_linear_classification_synthetic():
    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)

    # clearly linear: 45-px line, ~249 km (comfortably > 200 km strict bar)
    d_linear = np.zeros((60, 60))
    d_linear[10, 0:45] = 50.0
    # clearly mixed: 30-px line, ~166 km -- fails the linear tier's 200 km bar
    # but clears the mixed tier's 100 km bar (eccentricity ~1.0 clears both)
    d_mixed = np.zeros((60, 60))
    d_mixed[20, 0:30] = 50.0
    # clearly cellular: round blob (low eccentricity, fails both tiers regardless of size)
    d_cellular = np.zeros((60, 60))
    yy, xx = np.mgrid[0:60, 0:60]
    d_cellular[(yy - 40) ** 2 + (xx - 20) ** 2 <= 9] = 50.0
    # borderline: elongated but short -- 15-px line, ~83 km, under BOTH tiers'
    # length bars despite eccentricity ~1.0 -- confirms the mixed tier's lower
    # (length) bound is actually enforced, not just its upper (linear) bound
    d_borderline = np.zeros((60, 60))
    d_borderline[50, 0:15] = 50.0

    for name, data, expected in [
        ("linear", d_linear, 2), ("mixed", d_mixed, 1),
        ("cellular", d_cellular, 0), ("borderline (short)", d_borderline, 0),
    ]:
        labels, objects = identify_objects(data, gg, thresh_1=20.0, thresh_2=30.0, area_thresh_km2=1.0)
        assert len(objects) == 1, f"{name}: expected exactly 1 object"
        obj = objects[0]
        print(f"[linear-check3] {name}: major_axis_length={obj.major_axis_length:.1f}px "
              f"eccentricity={obj.eccentricity:.3f} is_linear={obj.is_linear}")
        assert obj.is_linear == expected, f"{name}: expected is_linear={expected}, got {obj.is_linear}"


# --- Check 4: real data sanity check -----------------------------------------

def test_is_linear_real_mpas_data():
    """Empirical scan across all 23 real MPAS lead times under the new
    default thresholds (0.8/200km linear, 0.75/100km mixed), reported here
    rather than assumed:

      file        n_total  n_cell  n_mixed  n_linear
      f000-f003        ...    all cellular
      f004-f006        ...    a few mixed, no linear
      f048             13       11        1         1   <- all 3 categories
      f096,f108,f120,f132       a few mixed each, no linear
      TOTAL (23 files) 533 cellular, 10 mixed, 1 linear

    Only f048 clears the new, much stricter 200 km linear bar anywhere in
    this dataset (the old 100 km rule was cleared far more often -- expected,
    since the new tier requires double the length). f048 is used here since
    it's the one real file exercising all three branches at once, not just
    synthetically.
    """
    field = load_model_netcdf(
        os.path.join(MPAS_DIR, "interp_mpas_3km_2023050100_mem1_f048.nc"), varname="refl10cm_max"
    )
    gg = precompute_grid_geometry(field.lat2d, field.lon2d)
    labels, objects = identify_objects(field.data, gg, thresh_1=45.0, thresh_2=50.2, area_thresh_km2=108.0)

    n_cellular = sum(1 for o in objects if o.is_linear == 0)
    n_mixed = sum(1 for o in objects if o.is_linear == 1)
    n_linear = sum(1 for o in objects if o.is_linear == 2)
    n_total = len(objects)
    print(f"\n[linear-check4] MPAS f048: {n_total} objects -- "
          f"{n_cellular} cellular, {n_mixed} mixed, {n_linear} linear")
    for o in objects:
        print(f"  id={o.id} area_km2={o.area_km2:.0f} major_axis_length_px={o.major_axis_length:.1f} "
              f"eccentricity={o.eccentricity:.3f} is_linear={o.is_linear}")
    assert all(o.is_linear in (0, 1, 2) for o in objects)
    assert n_total == 13 and n_cellular == 11 and n_mixed == 1 and n_linear == 1, (
        f"expected 13 total (11 cellular, 1 mixed, 1 linear) per the empirical scan above, "
        f"got {n_total} total ({n_cellular} cellular, {n_mixed} mixed, {n_linear} linear)"
    )


# --- Check 6: object file round-trip -----------------------------------------

def test_is_linear_roundtrip_single_and_ensemble_snapshot(tmp_path):
    from datetime import datetime, timedelta

    lat2d, lon2d = _synthetic_grid()
    gg = precompute_grid_geometry(lat2d, lon2d)
    t0 = datetime(2023, 5, 1, 0, 0, 0)

    def make_result(member_id):
        d_linear = np.zeros((60, 60))
        d_linear[10, 0:45] = 50.0  # linear (~249 km, see check 3)
        d_mixed = np.zeros((60, 60))
        d_mixed[20, 0:30] = 50.0  # mixed (~166 km, see check 3)
        d_cellular = np.zeros((60, 60))
        yy, xx = np.mgrid[0:60, 0:60]
        d_cellular[(yy - 40) ** 2 + (xx - 20) ** 2 <= 9] = 50.0  # cellular
        combined = np.maximum(np.maximum(d_linear, d_mixed), d_cellular)
        labels, objects = identify_objects(combined, gg, thresh_1=20.0, thresh_2=30.0, area_thresh_km2=1.0)
        return IdentificationResult(labels=labels, objects=objects, valid_time=t0, member_id=member_id)

    # single shape
    r = make_result(None)
    p_single = str(tmp_path / "single.nc")
    write_object_file(p_single, t0, lat2d, lon2d, [r], 1, 20.0, 30.0, 1.0)
    c = read_object_file(p_single)
    is_linear_values = sorted(o.is_linear for o in c.objects)
    print(f"\n[linear-check6] single: is_linear values={is_linear_values}")
    assert is_linear_values == sorted(o.is_linear for o in r.objects)
    assert set(is_linear_values) == {0, 1, 2}

    # ensemble_snapshot shape
    results = [make_result(f"mem{i+1}") for i in range(3)]
    p_ens = str(tmp_path / "ensemble.nc")
    write_object_file(p_ens, t0, lat2d, lon2d, results, 1, 20.0, 30.0, 1.0)
    c_ens = read_object_file(p_ens)
    print(f"[linear-check6] ensemble_snapshot: n_objects={len(c_ens.objects)}, "
          f"is_linear values={sorted(o.is_linear for o in c_ens.objects)}")
    assert set(o.is_linear for o in c_ens.objects) == {0, 1, 2}
