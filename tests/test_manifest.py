"""Tests for obj_core.manifest.build_model_manifest -- in particular
member_subdir_pattern, added after a real bug report: a real NCAR HPC MPAS
archive's input_dir held real per-member directories (mem1..mem10) alongside
an unrelated sibling directory (ens_mean_5mems) with no matching forecast
files, which made the unfiltered "every subdirectory is a member" discovery
raise FileNotFoundError on the first non-member directory it encountered.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_manifest.py -v -s
"""

import numpy as np
import netCDF4
import pytest

from python_obj.obj_core import build_model_manifest


def _write_member_file(path: str) -> None:
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


def _make_real_archive_layout(tmp_path):
    """mem1/, mem2/ hold real forecast files; ens_mean_5mems/ is a decoy
    sibling directory with no matching files -- mirrors the real reported
    layout exactly (just 2 members instead of 10, for a fast test)."""
    for member in ("mem1", "mem2"):
        member_dir = tmp_path / member
        member_dir.mkdir()
        _write_member_file(str(member_dir / "interp_mpas_3km_2023050100_f001.nc"))

    decoy_dir = tmp_path / "ens_mean_5mems"
    decoy_dir.mkdir()
    (decoy_dir / "readme.txt").write_text("not a forecast file")

    return str(tmp_path)


def test_default_member_subdir_pattern_reproduces_the_real_bug(tmp_path):
    input_dir = _make_real_archive_layout(tmp_path)
    with pytest.raises(FileNotFoundError, match="ens_mean_5mems"):
        build_model_manifest(
            input_dir=input_dir, file_pattern="interp_mpas*.nc",
            member_subdirs=True, stacked_members=False,
            var_name="refl10cm_max", lat_name="latitude", lon_name="longitude",
            init_attr="initializationTime", lead_attr="forecastHour", init_format="%Y%m%d%H",
        )


def test_member_subdir_pattern_excludes_the_decoy_directory(tmp_path):
    input_dir = _make_real_archive_layout(tmp_path)
    manifest, _ = build_model_manifest(
        input_dir=input_dir, file_pattern="interp_mpas*.nc",
        member_subdirs=True, stacked_members=False,
        var_name="refl10cm_max", lat_name="latitude", lon_name="longitude",
        init_attr="initializationTime", lead_attr="forecastHour", init_format="%Y%m%d%H",
        member_subdir_pattern="mem*",
    )
    member_ids = sorted({entry.member_id for entry in manifest})
    print(f"\n[manifest-check] member_ids discovered: {member_ids} (expect only mem1, mem2 -- decoy excluded)")
    assert member_ids == ["mem1", "mem2"]
    assert len(manifest) == 2
