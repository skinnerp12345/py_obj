"""Self-contained, flexible NetCDF object-file I/O.

One flexible schema handles all four file shapes (single member/time,
member-series, ensemble-snapshot, full) -- there is no separate "observation
format" vs "forecast format". Which of the optional `member`/`time` dimensions
are present in a given file is exactly how a reader tells which shape it's
looking at.

Deliberately replaces the legacy shelve (pickle-based, Python-version-fragile)
format with something self-describing and portable.
"""

from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Iterator

import netCDF4
import numpy as np

from .identify import StormObject

_TIME_UNITS = "seconds since 1970-01-01 00:00:00"

# StormObject fields written as one 1D array each along the `object` dimension.
_OBJECT_FLOAT_FIELDS = [
    "area_km2", "max_intensity", "mean_intensity", "major_axis_length",
    "minor_axis_length", "eccentricity", "orientation", "solidity",
    "centroid_lat", "centroid_lon", "centroid_x_km", "centroid_y_km",
    "centroid_row", "centroid_col",
]
# is_linear is no longer a boolean despite the name: 0=cellular, 1=mixed,
# 2=linear (see identify.py's identify_objects()) -- stored/read generically
# as a plain int like every other field here, no schema change needed.
_OBJECT_INT_FIELDS = ["id", "area_px", "is_linear"]


@dataclass
class SeriesEntry:
    """One manifest entry: a single (member, time) input to be identified."""
    valid_time: datetime
    filepath: str
    member_id: str | None = None
    extra_dim_index: int | None = None  # generic: which slice of a possibly-shared file to extract
    init_time: datetime | None = None  # the forecast's own init time (not valid_time) -- only
                                        # populated for model/forecast series; used to group
                                        # output by forecast case (file_grouping="init_snapshot")


@dataclass
class IdentificationResult:
    """One (member, time) identification result, ready to be written -- the
    unit write_object_file() operates on. member_id/valid_time are None only
    when genuinely inapplicable (never mixed: either every result in a file has
    a member_id or none do, and similarly for valid_time)."""
    labels: np.ndarray
    objects: list[StormObject]
    valid_time: datetime | None = None
    member_id: str | None = None
    init_time: datetime | None = None


@dataclass
class ObjectFileContents:
    lat2d: np.ndarray
    lon2d: np.ndarray
    labels: np.ndarray
    objects: list[StormObject]
    member_index: np.ndarray | None  # per-object, into member_ids; None if no member dim
    time_index: np.ndarray | None  # per-object, into valid_times; None if no time dim
    member_ids: list[str] | None
    valid_times: list[datetime] | None
    init_time: datetime | None
    tracked: bool
    thresh_1: float
    thresh_2: float
    area_thresh_km2: float
    track_bound_disp_km: float | None
    n_source_files: int  # count, not the file list itself -- a large ensemble/long
                          # forecast can consolidate hundreds of source files into one
                          # object file, and the full path list isn't otherwise used


def read_grid_shape(path: str) -> tuple[int, int]:
    """Just the (y, x) grid shape -- reads a netCDF variable's declared shape,
    not its data, so this is essentially free even for a file whose `labels`
    variable is multiple GB. Used to validate two files share a grid without
    paying for a full read of either."""
    with netCDF4.Dataset(path, "r") as ds:
        return tuple(ds.variables["lat"].shape)


def read_init_time_only(path: str) -> datetime | None:
    """Just the file's own init_time global attribute (None if absent, e.g. a
    truth/obs file that never has one) -- reads no variables at all, so this
    is essentially free. Used to name a consolidated (file_grouping=
    "init_snapshot") match file after its forecast case, without a full read."""
    with netCDF4.Dataset(path, "r") as ds:
        return datetime.fromisoformat(ds.init_time) if hasattr(ds, "init_time") else None


def _dt_to_seconds(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _seconds_to_dt(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


def write_object_file(
    path: str,
    init_time: datetime | None,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    results: list[IdentificationResult],
    n_source_files: int,
    thresh_1: float,
    thresh_2: float,
    area_thresh_km2: float,
    tracked: bool = False,
    track_bound_disp_km: float | None = None,
) -> None:
    """Write one object file from a list of per-(member,time) results.

    The file's shape (whether `member`/`time` dimensions exist) is derived
    automatically from how many DISTINCT member_id/valid_time values appear
    across `results` -- not passed explicitly. A dimension is only created
    when there's more than one distinct value to distinguish; a single shared
    value (or none, for member_id) is instead recorded as a global attribute,
    so e.g. a single-timestep single-member file gets plain 2D labels with no
    length-1 dimensions, matching the plain observation shape exactly. Passing
    many results with a shared member_id but many valid_times produces the
    member_series shape; many member_ids at one valid_time produces the
    ensemble_snapshot shape; many of both produces the full shape.
    """
    if not results:
        raise ValueError("write_object_file: results must be non-empty")
    if any(r.valid_time is None for r in results):
        raise ValueError("write_object_file: every result must have a valid_time")

    distinct_members = sorted({r.member_id for r in results if r.member_id is not None})
    distinct_times = sorted({r.valid_time for r in results})

    use_member_dim = len(distinct_members) > 1
    use_time_dim = len(distinct_times) > 1

    member_ids = distinct_members if use_member_dim else None
    valid_times = distinct_times if use_time_dim else None

    member_id_to_idx = {m: i for i, m in enumerate(member_ids)} if member_ids else {}
    time_to_idx = {t: i for i, t in enumerate(valid_times)} if valid_times else {}

    ny, nx = lat2d.shape

    # label_dims only -- NOT a pre-built consolidated array. The netCDF
    # `labels` variable is created below and each result's label array is
    # written directly into its own target slice as the loop reaches it,
    # rather than first stacking every result into one big in-memory array
    # and writing that. This matters at real scale: a case with many members
    # x many lead times (e.g. 5 members x 133 hourly lead times on a
    # full-CONUS grid) already needs several GB just to hold every result's
    # own label array (owned by the caller); a second, equally large
    # consolidated copy built here on top of that was a real, confirmed
    # cause of an OOM kill in production -- removing it roughly halves peak
    # memory for this function with no change in the file this produces.
    if use_member_dim and use_time_dim:
        label_dims = ("member", "time", "y", "x")
    elif use_member_dim:
        label_dims = ("member", "y", "x")
    elif use_time_dim:
        label_dims = ("time", "y", "x")
    else:
        if len(results) != 1:
            raise ValueError(
                "write_object_file: multiple results given but only one distinct "
                "member_id and one distinct valid_time -- nothing distinguishes them"
            )
        label_dims = ("y", "x")

    # flatten objects into a tidy table with member_index/time_index columns
    # (only present when the corresponding dimension exists)
    flat_objects: list[StormObject] = []
    flat_member_idx: list[int] = []
    flat_time_idx: list[int] = []
    for r in results:
        for obj in r.objects:
            flat_objects.append(obj)
            if use_member_dim:
                flat_member_idx.append(member_id_to_idx[r.member_id])
            if use_time_dim:
                flat_time_idx.append(time_to_idx[r.valid_time])

    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        ds.createDimension("object", len(flat_objects))
        if member_ids:
            ds.createDimension("member", len(member_ids))
        if valid_times:
            ds.createDimension("time", len(valid_times))

        lat_var = ds.createVariable("lat", "f8", ("y", "x"), zlib=True)
        lon_var = ds.createVariable("lon", "f8", ("y", "x"), zlib=True)
        lat_var[:, :] = lat2d
        lon_var[:, :] = lon2d

        labels_var = ds.createVariable("labels", "i4", label_dims, zlib=True)
        for r in results:
            if use_member_dim and use_time_dim:
                labels_var[member_id_to_idx[r.member_id], time_to_idx[r.valid_time], :, :] = r.labels
            elif use_member_dim:
                labels_var[member_id_to_idx[r.member_id], :, :] = r.labels
            elif use_time_dim:
                labels_var[time_to_idx[r.valid_time], :, :] = r.labels
            else:
                labels_var[:, :] = r.labels

        if member_ids:
            member_var = ds.createVariable("member_id", str, ("member",))
            for i, m in enumerate(member_ids):
                member_var[i] = m
        if valid_times:
            time_var = ds.createVariable("valid_time", "f8", ("time",))
            time_var.units = _TIME_UNITS
            time_var[:] = [_dt_to_seconds(t) for t in valid_times]

        for fname in _OBJECT_FLOAT_FIELDS:
            var = ds.createVariable(fname, "f8", ("object",), zlib=True)
            if fname == "centroid_row":
                var[:] = [o.centroid_rowcol[0] for o in flat_objects]
            elif fname == "centroid_col":
                var[:] = [o.centroid_rowcol[1] for o in flat_objects]
            else:
                var[:] = [getattr(o, fname) for o in flat_objects]
        for fname in _OBJECT_INT_FIELDS:
            var = ds.createVariable(fname, "i8", ("object",), zlib=True)
            var[:] = [getattr(o, fname) for o in flat_objects]

        if member_ids:
            obj_member_var = ds.createVariable("object_member_index", "i4", ("object",))
            obj_member_var[:] = flat_member_idx
        if valid_times:
            obj_time_var = ds.createVariable("object_time_index", "i4", ("object",))
            obj_time_var[:] = flat_time_idx

        if tracked:
            age_var = ds.createVariable("age_seconds", "f8", ("object",), zlib=True)
            age_var[:] = [o.age_seconds if o.age_seconds is not None else np.nan for o in flat_objects]
            track_id_var = ds.createVariable("track_id", "i8", ("object",), zlib=True)
            track_id_var[:] = [o.track_id if o.track_id is not None else -1 for o in flat_objects]

        if init_time is not None:
            ds.init_time = init_time.isoformat()
        ds.n_source_files = n_source_files
        ds.thresh_1 = thresh_1
        ds.thresh_2 = thresh_2
        ds.area_thresh_km2 = area_thresh_km2
        ds.tracked = int(tracked)
        if tracked and track_bound_disp_km is not None:
            ds.track_bound_disp_km = track_bound_disp_km

        # single shared value (not distinguished by a dimension) -> global attr
        if not use_time_dim:
            ds.valid_time = _dt_to_seconds(distinct_times[0])
        if not use_member_dim and distinct_members:
            ds.member_id = distinct_members[0]


def _read_member_ids(ds: netCDF4.Dataset) -> list[str] | None:
    # member_id is either a dimensioned variable (multiple distinct values) or
    # a single global attribute (one shared value) -- normalize both into a
    # plain list (or None) so callers have one interface.
    if "member_id" in ds.variables:
        return list(ds.variables["member_id"][:])
    elif hasattr(ds, "member_id"):
        return [ds.member_id]
    return None


def _read_valid_times(ds: netCDF4.Dataset) -> list[datetime]:
    if "valid_time" in ds.variables:
        return [_seconds_to_dt(t) for t in ds.variables["valid_time"][:]]
    return [_seconds_to_dt(ds.valid_time)]


def read_valid_time_range(path: str) -> tuple[datetime, datetime]:
    """Just the (min, max) valid_time spanned by a file -- reads only the
    `valid_time` variable/attribute (a handful of floats at most), never
    lat/lon/labels, so this is essentially free even for a file whose
    `labels` variable is multiple GB. Used to bound which truth files are
    even worth considering for a given forecast case, before touching any
    of the (potentially large) truth archive itself."""
    with netCDF4.Dataset(path, "r") as ds:
        times = _read_valid_times(ds)
    return min(times), max(times)


def _read_objects_table(ds: netCDF4.Dataset) -> list[StormObject]:
    """Read every StormObject field except the labels grid -- always cheap
    (one 1D array per field, length = number of objects), regardless of how
    large the file's `labels` variable is."""
    n_obj = ds.dimensions["object"].size
    tracked = bool(ds.tracked)

    objects = []
    for i in range(n_obj):
        kwargs = {}
        for fname in _OBJECT_FLOAT_FIELDS:
            if fname in ("centroid_row", "centroid_col"):
                continue
            kwargs[fname] = float(ds.variables[fname][i])
        centroid_rowcol = (float(ds.variables["centroid_row"][i]), float(ds.variables["centroid_col"][i]))
        age_seconds = None
        track_id = None
        if tracked:
            raw_age = float(ds.variables["age_seconds"][i])
            age_seconds = None if np.isnan(raw_age) else raw_age
            raw_tid = int(ds.variables["track_id"][i])
            track_id = None if raw_tid == -1 else raw_tid
        objects.append(
            StormObject(
                id=int(ds.variables["id"][i]),
                area_px=int(ds.variables["area_px"][i]),
                is_linear=int(ds.variables["is_linear"][i]),
                centroid_rowcol=centroid_rowcol,
                age_seconds=age_seconds,
                track_id=track_id,
                **kwargs,
            )
        )
    return objects


def _read_n_source_files(ds: netCDF4.Dataset, path: str) -> int:
    if hasattr(ds, "n_source_files"):
        return int(ds.n_source_files)
    elif "member_id" not in ds.variables and "valid_time" not in ds.variables:
        # "single" grouping shape (plain 2D labels, no member/time
        # dimension) is, by construction, always derived from exactly
        # one input file -- safe to infer for files written before
        # n_source_files existed (e.g. a real MRMS archive generated
        # with an older schema version), with no need to touch the old
        # source_files string attribute at all. Any other shape
        # (member_series/ensemble_snapshot/full/init_snapshot) could
        # genuinely have come from a different number of files, so
        # inference isn't safe there -- fail loud instead of guessing.
        return 1
    else:
        raise ValueError(
            f"'{path}' has no 'n_source_files' global attribute and is not a "
            "single-file-shape object file (has a member_id and/or valid_time "
            "dimension) -- cannot safely infer a source file count for this shape."
        )


def read_object_file(path: str) -> ObjectFileContents:
    with netCDF4.Dataset(path, "r") as ds:
        lat2d = np.asarray(ds.variables["lat"][:])
        lon2d = np.asarray(ds.variables["lon"][:])
        labels = np.asarray(ds.variables["labels"][:])

        member_ids = _read_member_ids(ds)
        valid_times = _read_valid_times(ds)
        tracked = bool(ds.tracked)
        objects = _read_objects_table(ds)

        member_index = np.asarray(ds.variables["object_member_index"][:]) if "object_member_index" in ds.variables else None
        time_index = np.asarray(ds.variables["object_time_index"][:]) if "object_time_index" in ds.variables else None
        n_source_files = _read_n_source_files(ds, path)

        return ObjectFileContents(
            lat2d=lat2d,
            lon2d=lon2d,
            labels=labels,
            objects=objects,
            member_index=member_index,
            time_index=time_index,
            member_ids=member_ids,
            valid_times=valid_times,
            init_time=datetime.fromisoformat(ds.init_time) if hasattr(ds, "init_time") else None,
            tracked=tracked,
            thresh_1=float(ds.thresh_1),
            thresh_2=float(ds.thresh_2),
            area_thresh_km2=float(ds.area_thresh_km2),
            track_bound_disp_km=float(ds.track_bound_disp_km) if hasattr(ds, "track_bound_disp_km") else None,
            n_source_files=n_source_files,
        )


def iter_object_slices(
    contents: ObjectFileContents,
) -> Iterator[tuple[str | None, datetime, np.ndarray, list[StormObject]]]:
    """Unpack an object file's contents (any of the four shapes) back into
    individual (member_id, valid_time, labels2d, objects) slices -- the
    inverse of what write_object_file's shape-collapsing does. Used by
    matching (which operates one (member, time) slice at a time regardless of
    how the source files happened to be grouped), but generically useful
    anywhere a caller needs to iterate per-slice without caring which of the
    four shapes it's reading.
    """
    has_member = contents.member_index is not None
    has_time = contents.time_index is not None

    if has_member and has_time:
        for mi, member_id in enumerate(contents.member_ids):
            for ti, vt in enumerate(contents.valid_times):
                labels2d = contents.labels[mi, ti]
                objects = [
                    o for o, m, t in zip(contents.objects, contents.member_index, contents.time_index)
                    if m == mi and t == ti
                ]
                yield member_id, vt, labels2d, objects
    elif has_member:
        for mi, member_id in enumerate(contents.member_ids):
            labels2d = contents.labels[mi]
            objects = [o for o, m in zip(contents.objects, contents.member_index) if m == mi]
            yield member_id, contents.valid_times[0], labels2d, objects
    elif has_time:
        for ti, vt in enumerate(contents.valid_times):
            labels2d = contents.labels[ti]
            objects = [o for o, t in zip(contents.objects, contents.time_index) if t == ti]
            yield None, vt, labels2d, objects
    else:
        member_id = contents.member_ids[0] if contents.member_ids else None
        yield member_id, contents.valid_times[0], contents.labels, contents.objects


def iter_object_slices_lazy(
    path: str,
) -> Iterator[tuple[str | None, datetime, np.ndarray, list[StormObject]]]:
    """Same contract/output as iter_object_slices(read_object_file(path)), but
    avoids read_object_file's single eager `labels_var[:]` read when both a
    member and a time dimension are present (the shape a large ensemble/
    long-forecast case takes under file_grouping="init_snapshot") -- confirmed
    by direct measurement, not assumed, that eagerly loading the whole
    `labels` array for a real 5-member x 133-lead-time x 1059x1799 case file
    peaks at ~10 GB transient memory (netCDF4/HDF5 decompression overhead on
    top of the final ~5 GB array), too much headroom to risk when processing
    many such files back-to-back on a memory-constrained machine.

    Reads one member's full (time, y, x) block at a time instead
    (`labels_var[mi, :, :, :]`) -- measured at ~2 GB peak per member with no
    extra wall-clock cost (same order as the eager read). Reading a single
    (member, time) 2D slice at a time is cheaper still (~tens of MB peak) but
    measured ~16x slower overall, because this file's native chunk layout
    tiles the spatial dimensions -- a lone full-domain 2D read touches many
    chunks and thrashes the HDF5 chunk cache, whereas one member's block
    matches the chunking's own time-major layout.

    For the other three shapes (no member+time combination) `labels` is
    already small (a few MB up to ~1 GB) -- no special-casing needed there,
    this just delegates to the normal eager read.
    """
    with netCDF4.Dataset(path, "r") as ds:
        has_member_dim = "member" in ds.dimensions
        has_time_dim = "time" in ds.dimensions
        needs_member_by_member_read = has_member_dim and has_time_dim

        if not needs_member_by_member_read:
            member_ids = None
            valid_times = None
            objects = None
            member_index = None
            time_index = None
            labels_var = None
        else:
            member_ids = _read_member_ids(ds)
            valid_times = _read_valid_times(ds)
            objects = _read_objects_table(ds)
            member_index = np.asarray(ds.variables["object_member_index"][:])
            time_index = np.asarray(ds.variables["object_time_index"][:])
            labels_var = ds.variables["labels"]

            for mi, member_id in enumerate(member_ids):
                member_labels = np.asarray(labels_var[mi, :, :, :])
                member_object_mask = member_index == mi
                for ti, vt in enumerate(valid_times):
                    slice_mask = member_object_mask & (time_index == ti)
                    slice_objects = [o for o, keep in zip(objects, slice_mask) if keep]
                    yield member_id, vt, member_labels[ti], slice_objects
                del member_labels

    if not needs_member_by_member_read:
        # small shape (no member+time combination) -- labels is already cheap
        # (a few MB up to ~1 GB), no special-casing needed, delegate normally
        yield from iter_object_slices(read_object_file(path))
