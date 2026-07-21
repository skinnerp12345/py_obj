"""Shared model-file manifest builder -- promoted out of
drivers/identify_track_model.py (was a private, ModelConfig-coupled
function) so it can be reused by any driver that needs to discover a
model/forecast series' files (currently: identify_track_model.py and the
histogram-building driver), parameterized by plain values instead of a
specific config dataclass.
"""

import glob
import os
from typing import Callable

from python_obj.regrid import infer_stacked_member_count, load_model_netcdf, read_valid_time_only

from .object_io import SeriesEntry


def build_model_manifest(
    input_dir: str,
    file_pattern: str,
    member_subdirs: bool,
    stacked_members: bool,
    var_name: str,
    lat_name: str,
    lon_name: str,
    init_attr: str | None = None,
    lead_attr: str | None = None,
    lead_units: str = "hours",
    init_format: str | None = None,
    valid_time_attr: str | None = None,
    valid_time_format: str | None = None,
) -> tuple[list[SeriesEntry], Callable[..., object]]:
    """Build the (member, time, filepath) manifest run_object_id_series (or
    the histogram driver) needs, plus a matching loader closure.

    member_subdirs=True: one member per immediate subdirectory of input_dir
    (member_id = subdirectory basename), mirroring the real test_mpas/mem1/,
    test_mpas/mem2/ layout -- not parsed from any filename convention, since
    a model's own ensemble-naming scheme is the caller's business, not this
    pipeline's.

    stacked_members=True: each file contains ALL members stacked as a real
    array dimension (e.g. WoFS's comp_dz(ne=18, lat, lon), one file per
    valid_time rather than one file per member) -- one flat glob over
    input_dir, then infer_stacked_member_count() once per file (a cheap
    metadata-only shape read, never loads the full array) to discover how
    many members it holds, producing one SeriesEntry per (file, member index)
    with a distinct member_id and extra_dim_index.

    member_subdirs=False and stacked_members=False: one flat glob over
    input_dir, member_id=None (a single deterministic run).

    valid_time comes from load_model_netcdf()'s (or, for stacked_members,
    read_valid_time_only()'s) own flexible derivation -- never parsed from
    the filename.
    """
    loader = lambda fp, extra_dim_index=None: load_model_netcdf(
        fp,
        varname=var_name, lat_name=lat_name, lon_name=lon_name,
        init_attr=init_attr, lead_attr=lead_attr, lead_units=lead_units, init_format=init_format,
        valid_time_attr=valid_time_attr, valid_time_format=valid_time_format,
        extra_dim_index=extra_dim_index,
    )

    manifest: list[SeriesEntry] = []

    if member_subdirs:
        member_dirs = sorted(
            d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)
        )
        if not member_dirs:
            raise FileNotFoundError(f"member_subdirs=True but no subdirectories found under '{input_dir}'")
        for member_dir in member_dirs:
            member_id = os.path.basename(os.path.normpath(member_dir))
            files = sorted(glob.glob(os.path.join(member_dir, file_pattern)))
            if not files:
                raise FileNotFoundError(
                    f"No files matching '{file_pattern}' found under member directory '{member_dir}'"
                )
            for f in files:
                manifest.append(SeriesEntry(valid_time=loader(f).valid_time, filepath=f, member_id=member_id))
    elif stacked_members:
        files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
        if not files:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found under '{input_dir}'")
        for f in files:
            valid_time = read_valid_time_only(
                f,
                init_attr=init_attr, lead_attr=lead_attr, lead_units=lead_units, init_format=init_format,
                valid_time_attr=valid_time_attr, valid_time_format=valid_time_format,
            )
            n_members = infer_stacked_member_count(f, var_name)
            for idx in range(n_members):
                manifest.append(SeriesEntry(
                    valid_time=valid_time, filepath=f,
                    member_id=f"mem{idx:02d}", extra_dim_index=idx,
                ))
    else:
        files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
        if not files:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found under '{input_dir}'")
        for f in files:
            manifest.append(SeriesEntry(valid_time=loader(f).valid_time, filepath=f, member_id=None))

    return manifest, loader
