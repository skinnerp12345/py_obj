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

from py_obj.filename_time import parse_filename_times
from py_obj.regrid import (
    infer_stacked_member_count, load_model_grib2, load_model_netcdf, read_grib2_times, read_init_time_only,
    read_valid_time_only,
)

from .object_io import SeriesEntry


def _try_read_init_time(f, init_attr, init_format, valid_time_attr, valid_time_format, init_time_attr,
                        filename_time_template=None, init_time_offset_minutes=None):
    """init_time is a best-effort addition (file_grouping='init_snapshot' is
    the only current consumer) -- a caller not using that grouping shouldn't
    have their whole manifest build fail just because e.g. init_time_attr is
    absent from this particular file. Returns None (not an error) on any
    read failure; run_object_id_series's init_snapshot branch is what
    actually enforces init_time being present, for whoever asks for it.

    Exception: with filename_time_template set, a filename that doesn't match
    the template (or a template with no init slot) is a config error, not a
    best-effort miss -- it raises rather than silently returning None."""
    if filename_time_template is not None:
        return read_init_time_only(
            f, filename_time_template=filename_time_template, init_time_offset_minutes=init_time_offset_minutes,
        )
    try:
        return read_init_time_only(
            f, init_attr=init_attr or "initializationTime", init_format=init_format or "%Y%m%d%H",
            valid_time_attr=valid_time_attr, valid_time_format=valid_time_format,
            init_time_attr=init_time_attr,
        )
    except Exception:
        return None


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
    valid_time_var: str | None = None,
    member_subdir_pattern: str = "*",
    init_time_attr: str = "init_time",
    filename_time_template: str | None = None,
    init_time_offset_minutes: int | None = None,
    file_format: str = "netcdf",
    grib_selector=None,
) -> tuple[list[SeriesEntry], Callable[..., object]]:
    """Build the (member, time, filepath) manifest run_object_id_series (or
    the histogram driver) needs, plus a matching loader closure.

    member_subdirs=True: one member per immediate subdirectory of input_dir
    matching member_subdir_pattern (member_id = subdirectory basename) --
    e.g. an input_dir with `mem1/`, `mem2/` subdirectories, one ensemble
    member per subdirectory. Member id is never parsed from any filename
    convention, since a model's own ensemble-naming scheme
    is the caller's business, not this pipeline's. member_subdir_pattern
    (default "*", i.e. every subdirectory) exists because a real archive's
    input_dir can hold non-member sibling directories alongside the real
    per-member ones (confirmed real: an NCAR HPC MPAS archive with mem1-mem10
    plus an unrelated ens_mean_5mems directory at the same level) -- without
    a name filter, the first non-member subdirectory with no matching files
    raises FileNotFoundError.

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
    the filename. valid_time_var is an opt-in override for a source whose
    global valid_time attribute is known-unreliable (e.g. a real confirmed
    WoFSCast bug where it was a stale copy of init_time): when given, it
    takes precedence over valid_time_attr, decoding a CF-convention time
    coordinate variable instead (see regrid.io_grid._resolve_valid_time()'s
    full precedence order).

    init_time (the forecast's own initialization time, distinct from
    valid_time) is read once per file via read_init_time_only() and attached
    to every SeriesEntry that file produces -- needed to group/name output by
    forecast case (file_grouping="init_snapshot"). Best-effort: a file this
    can't be derived for (e.g. init_time_attr absent) gets init_time=None
    rather than failing the whole manifest build, since most callers
    (anything not using init_snapshot) never look at it.

    filename_time_template/init_time_offset_minutes (opt-in, default off):
    derive valid_time and init_time from each file's NAME instead of its
    metadata (see py_obj.filename_time) -- for sources whose internal
    time metadata is unreliable. init_time_offset_minutes is subtracted from
    the filename init time only (e.g. WoFSCast initialized 20 min after its
    parent WoFS run).

    file_format ("netcdf" default | "grib2"): the forecast files' format. For
    "grib2" (see py_obj.regrid.io_grib) the record is chosen by grib_selector
    or, if none, the var_name preset (e.g. "REFC"); the grid comes from the
    message (lat_name/lon_name unused); init/valid times come from the message
    keys (analDate/validDate) unless filename_time_template is set, which wins
    as it does for NetCDF. The NetCDF time-attribute options do not apply.
    stacked_members is NetCDF-only. Member/flat discovery, SeriesEntry
    contents and everything downstream are identical for both formats.
    """
    if file_format not in ("netcdf", "grib2"):
        raise ValueError(f"file_format must be 'netcdf' or 'grib2', got {file_format!r}")
    if file_format == "grib2":
        if stacked_members:
            raise ValueError("stacked_members is NetCDF-only (it reads an extra array dimension); not valid with file_format='grib2'")
        return _build_grib2_manifest(
            input_dir, file_pattern, member_subdirs, member_subdir_pattern, var_name, grib_selector,
            filename_time_template, init_time_offset_minutes,
        )

    loader = lambda fp, extra_dim_index=None: load_model_netcdf(
        fp,
        varname=var_name, lat_name=lat_name, lon_name=lon_name,
        init_attr=init_attr, lead_attr=lead_attr, lead_units=lead_units, init_format=init_format,
        valid_time_attr=valid_time_attr, valid_time_format=valid_time_format, valid_time_var=valid_time_var,
        filename_time_template=filename_time_template,
        extra_dim_index=extra_dim_index,
    )

    manifest: list[SeriesEntry] = []

    if member_subdirs:
        member_dirs = sorted(
            d for d in glob.glob(os.path.join(input_dir, member_subdir_pattern)) if os.path.isdir(d)
        )
        if not member_dirs:
            raise FileNotFoundError(
                f"member_subdirs=True but no subdirectories matching '{member_subdir_pattern}' found under '{input_dir}'"
            )
        for member_dir in member_dirs:
            member_id = os.path.basename(os.path.normpath(member_dir))
            files = sorted(glob.glob(os.path.join(member_dir, file_pattern)))
            if not files:
                raise FileNotFoundError(
                    f"No files matching '{file_pattern}' found under member directory '{member_dir}'"
                )
            for f in files:
                init_time = _try_read_init_time(
                    f, init_attr, init_format, valid_time_attr, valid_time_format, init_time_attr,
                    filename_time_template, init_time_offset_minutes,
                )
                manifest.append(SeriesEntry(
                    valid_time=loader(f).valid_time, filepath=f, member_id=member_id, init_time=init_time,
                ))
    elif stacked_members:
        files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
        if not files:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found under '{input_dir}'")
        for f in files:
            valid_time = read_valid_time_only(
                f,
                init_attr=init_attr, lead_attr=lead_attr, lead_units=lead_units, init_format=init_format,
                valid_time_attr=valid_time_attr, valid_time_format=valid_time_format, valid_time_var=valid_time_var,
                filename_time_template=filename_time_template,
            )
            init_time = _try_read_init_time(
                f, init_attr, init_format, valid_time_attr, valid_time_format, init_time_attr,
                filename_time_template, init_time_offset_minutes,
            )
            n_members = infer_stacked_member_count(f, var_name)
            for idx in range(n_members):
                manifest.append(SeriesEntry(
                    valid_time=valid_time, filepath=f,
                    member_id=f"mem{idx:02d}", extra_dim_index=idx, init_time=init_time,
                ))
    else:
        files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
        if not files:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found under '{input_dir}'")
        for f in files:
            init_time = _try_read_init_time(
                f, init_attr, init_format, valid_time_attr, valid_time_format, init_time_attr,
                filename_time_template, init_time_offset_minutes,
            )
            manifest.append(SeriesEntry(valid_time=loader(f).valid_time, filepath=f, member_id=None, init_time=init_time))

    return manifest, loader


def _discover(input_dir, file_pattern, member_subdirs, member_subdir_pattern) -> list[tuple[str | None, str]]:
    """[(member_id, filepath)] -- the same member_subdirs/flat discovery rules,
    and the same errors, as build_model_manifest's NetCDF branches."""
    if not member_subdirs:
        files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
        if not files:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found under '{input_dir}'")
        return [(None, f) for f in files]
    member_dirs = sorted(d for d in glob.glob(os.path.join(input_dir, member_subdir_pattern)) if os.path.isdir(d))
    if not member_dirs:
        raise FileNotFoundError(
            f"member_subdirs=True but no subdirectories matching '{member_subdir_pattern}' found under '{input_dir}'"
        )
    out = []
    for member_dir in member_dirs:
        files = sorted(glob.glob(os.path.join(member_dir, file_pattern)))
        if not files:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found under member directory '{member_dir}'")
        out.extend((os.path.basename(os.path.normpath(member_dir)), f) for f in files)
    return out


def _build_grib2_manifest(
    input_dir, file_pattern, member_subdirs, member_subdir_pattern, var_name, grib_selector,
    filename_time_template, init_time_offset_minutes,
):
    """GRIB2 counterpart of the flat/member_subdirs branches above: times are read
    from each message's header once (values are only decoded later, by the
    loader), and a filename_time_template overrides them exactly as for NetCDF."""
    def times(f):
        if filename_time_template is not None:
            init, valid = parse_filename_times(f, filename_time_template, init_time_offset_minutes)
            if valid is None:
                raise ValueError(f"filename_time_template {filename_time_template!r} has no '{{valid:...}}' slot")
            return init, valid
        return read_grib2_times(f, var_name=var_name, grib_selector=grib_selector)

    def loader(fp, extra_dim_index=None):
        if extra_dim_index is not None:
            raise ValueError("extra_dim_index is NetCDF-only (stacked_members); not valid for GRIB2 files")
        field = load_model_grib2(fp, var_name=var_name, grib_selector=grib_selector)
        if filename_time_template is not None:
            field.valid_time = times(fp)[1]
        return field

    manifest = []
    for member_id, f in _discover(input_dir, file_pattern, member_subdirs, member_subdir_pattern):
        init_time, valid_time = times(f)
        manifest.append(SeriesEntry(valid_time=valid_time, filepath=f, member_id=member_id, init_time=init_time))
    return manifest, loader
