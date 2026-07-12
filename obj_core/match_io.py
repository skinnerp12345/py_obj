"""Self-contained, flexible NetCDF match-result-file I/O.

Mirrors object_io.py's flexible-dimension pattern exactly: `member`/`time`
dimensions exist only when more than one distinct value is present across the
results being written; a single shared value becomes a global attribute
instead. No grid/labels concept here (unlike object files) -- match results
are purely tabular, one row per truth/forecast object (matched or not).
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import netCDF4
import numpy as np

from .matching import MatchRecord

_TIME_UNITS = "seconds since 1970-01-01 00:00:00"

_MATCH_FLOAT_FIELDS = [
    "ti_score",
    "truth_area_km2", "truth_max_intensity", "truth_centroid_lat", "truth_centroid_lon",
    "forecast_area_km2", "forecast_max_intensity", "forecast_centroid_lat", "forecast_centroid_lon",
]
_MATCH_REQUIRED_INT_FIELDS = ["truth_id", "forecast_id"]  # always int, -1 = not applicable
# truth_is_linear/forecast_is_linear: 0=cellular, 1=mixed, 2=linear -- -1
# sentinel for None (not applicable) remains safe/distinct from all 3 values.
_MATCH_OPTIONAL_INT_FIELDS = ["truth_is_linear", "forecast_is_linear"]  # None when not applicable


@dataclass
class MatchResult:
    """One (member, time) match result -- ready to be written. Mirrors
    IdentificationResult's role in object_io.py."""
    records: list[MatchRecord]
    valid_time: datetime | None = None
    member_id: str | None = None


@dataclass
class MatchFileContents:
    records: list[MatchRecord]
    member_index: np.ndarray | None  # per-record, into member_ids; None if no member dim
    time_index: np.ndarray | None  # per-record, into valid_times; None if no time dim
    member_ids: list[str] | None
    valid_times: list[datetime] | None
    max_boundary_disp_km: float
    max_centroid_disp_km: float
    ti_threshold: float
    max_time_offset_minutes: float | None
    truth_source_files: list[str]
    forecast_source_files: list[str]


def _dt_to_seconds(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _seconds_to_dt(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


def write_match_file(
    path: str,
    results: list[MatchResult],
    truth_source_files: list[str],
    forecast_source_files: list[str],
    max_boundary_disp_km: float,
    max_centroid_disp_km: float,
    ti_threshold: float,
    max_time_offset_minutes: float | None = None,
) -> None:
    """Write one match file from a list of per-(member,time) results.

    Shape (whether `member`/`time` dimensions exist) is derived automatically
    from how many DISTINCT member_id/valid_time values appear across
    `results` -- exactly the same rule as write_object_file().
    """
    if not results:
        raise ValueError("write_match_file: results must be non-empty")
    if any(r.valid_time is None for r in results):
        raise ValueError("write_match_file: every result must have a valid_time")

    distinct_members = sorted({r.member_id for r in results if r.member_id is not None})
    distinct_times = sorted({r.valid_time for r in results})

    use_member_dim = len(distinct_members) > 1
    use_time_dim = len(distinct_times) > 1

    member_ids = distinct_members if use_member_dim else None
    valid_times = distinct_times if use_time_dim else None

    member_id_to_idx = {m: i for i, m in enumerate(member_ids)} if member_ids else {}
    time_to_idx = {t: i for i, t in enumerate(valid_times)} if valid_times else {}

    # flatten records into a tidy table with member_index/time_index columns
    # (only present when the corresponding dimension exists)
    flat_records: list[MatchRecord] = []
    flat_member_idx: list[int] = []
    flat_time_idx: list[int] = []
    for r in results:
        for rec in r.records:
            flat_records.append(rec)
            if use_member_dim:
                flat_member_idx.append(member_id_to_idx[r.member_id])
            if use_time_dim:
                flat_time_idx.append(time_to_idx[r.valid_time])

    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("match", len(flat_records))
        if member_ids:
            ds.createDimension("member", len(member_ids))
        if valid_times:
            ds.createDimension("time", len(valid_times))

        if member_ids:
            member_var = ds.createVariable("member_id", str, ("member",))
            for i, m in enumerate(member_ids):
                member_var[i] = m
        if valid_times:
            time_var = ds.createVariable("valid_time", "f8", ("time",))
            time_var.units = _TIME_UNITS
            time_var[:] = [_dt_to_seconds(t) for t in valid_times]

        category_var = ds.createVariable("category", str, ("match",))
        for i, rec in enumerate(flat_records):
            category_var[i] = rec.category

        for fname in _MATCH_REQUIRED_INT_FIELDS:
            var = ds.createVariable(fname, "i8", ("match",), zlib=True)
            var[:] = [getattr(rec, fname) for rec in flat_records]
        for fname in _MATCH_OPTIONAL_INT_FIELDS:
            var = ds.createVariable(fname, "i8", ("match",), zlib=True)
            var[:] = [getattr(rec, fname) if getattr(rec, fname) is not None else -1 for rec in flat_records]
        for fname in _MATCH_FLOAT_FIELDS:
            var = ds.createVariable(fname, "f8", ("match",), zlib=True)
            var[:] = [getattr(rec, fname) if getattr(rec, fname) is not None else np.nan for rec in flat_records]

        if member_ids:
            match_member_var = ds.createVariable("match_member_index", "i4", ("match",))
            match_member_var[:] = flat_member_idx
        if valid_times:
            match_time_var = ds.createVariable("match_time_index", "i4", ("match",))
            match_time_var[:] = flat_time_idx

        ds.truth_source_files = ";".join(truth_source_files)
        ds.forecast_source_files = ";".join(forecast_source_files)
        ds.max_boundary_disp_km = max_boundary_disp_km
        ds.max_centroid_disp_km = max_centroid_disp_km
        ds.ti_threshold = ti_threshold
        if max_time_offset_minutes is not None:
            ds.max_time_offset_minutes = max_time_offset_minutes

        # single shared value (not distinguished by a dimension) -> global attr
        if not use_time_dim:
            ds.valid_time = _dt_to_seconds(distinct_times[0])
        if not use_member_dim and distinct_members:
            ds.member_id = distinct_members[0]


def read_match_file(path: str) -> MatchFileContents:
    with netCDF4.Dataset(path, "r") as ds:
        if "member_id" in ds.variables:
            member_ids = list(ds.variables["member_id"][:])
        elif hasattr(ds, "member_id"):
            member_ids = [ds.member_id]
        else:
            member_ids = None

        if "valid_time" in ds.variables:
            valid_times = [_seconds_to_dt(t) for t in ds.variables["valid_time"][:]]
        else:
            valid_times = [_seconds_to_dt(ds.valid_time)]

        n_match = ds.dimensions["match"].size

        records = []
        for i in range(n_match):
            kwargs = {"category": str(ds.variables["category"][i])}
            for fname in _MATCH_REQUIRED_INT_FIELDS:
                kwargs[fname] = int(ds.variables[fname][i])
            for fname in _MATCH_OPTIONAL_INT_FIELDS:
                raw = int(ds.variables[fname][i])
                kwargs[fname] = None if raw == -1 else raw
            for fname in _MATCH_FLOAT_FIELDS:
                raw = float(ds.variables[fname][i])
                kwargs[fname] = None if np.isnan(raw) else raw
            records.append(MatchRecord(**kwargs))

        member_index = np.asarray(ds.variables["match_member_index"][:]) if "match_member_index" in ds.variables else None
        time_index = np.asarray(ds.variables["match_time_index"][:]) if "match_time_index" in ds.variables else None

        return MatchFileContents(
            records=records,
            member_index=member_index,
            time_index=time_index,
            member_ids=member_ids,
            valid_times=valid_times,
            max_boundary_disp_km=float(ds.max_boundary_disp_km),
            max_centroid_disp_km=float(ds.max_centroid_disp_km),
            ti_threshold=float(ds.ti_threshold),
            max_time_offset_minutes=float(ds.max_time_offset_minutes) if hasattr(ds, "max_time_offset_minutes") else None,
            truth_source_files=ds.truth_source_files.split(";"),
            forecast_source_files=ds.forecast_source_files.split(";"),
        )
