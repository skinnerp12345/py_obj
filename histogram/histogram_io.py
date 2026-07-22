"""Self-contained, flexible NetCDF histogram-file I/O.

Mirrors object_io.py's/match_io.py's tidy-table convention: one file holds a
flat table of "slices" (one row per input file/timestep the histogram was
built from), each carrying its own real valid_time and (when applicable)
lead_hours/member_id -- this per-slice granularity is what lets
histogram.aggregate later reconstruct subsets (by hour of day, by lead-time
bucket) that a single pre-collapsed histogram never could.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import netCDF4
import numpy as np

_TIME_UNITS = "seconds since 1970-01-01 00:00:00"
_NO_LEAD_HOURS = -1.0  # sentinel: this slice has no lead-time concept (pure obs)


@dataclass
class HistogramSlice:
    """One (file, [member]) contribution to a histogram file."""
    valid_time: datetime
    hist: np.ndarray  # raw bin counts, shape (n_bins,)
    lead_hours: float | None = None
    member_id: str | None = None


@dataclass
class HistogramFileContents:
    bins: np.ndarray  # bin edges, shape (n_bins + 1,)
    slices: list[HistogramSlice]
    varname: str
    edge_trim: int
    clip_negative_to_zero: bool
    n_source_files: int  # count, not the file list itself -- a large ensemble/long
                          # forecast can pool hundreds of source files into one
                          # histogram file, and the full path list isn't otherwise used


def _dt_to_seconds(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _seconds_to_dt(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


def write_histogram_file(
    path: str,
    bins: np.ndarray,
    slices: list[HistogramSlice],
    varname: str,
    edge_trim: int,
    clip_negative_to_zero: bool,
    n_source_files: int,
) -> None:
    if not slices:
        raise ValueError("write_histogram_file: slices must be non-empty")

    n_bins = len(bins) - 1
    for s in slices:
        if s.hist.shape != (n_bins,):
            raise ValueError(
                f"write_histogram_file: slice hist shape {s.hist.shape} does not match bins ({n_bins},)"
            )

    has_lead_hours = any(s.lead_hours is not None for s in slices)
    has_member_id = any(s.member_id is not None for s in slices)

    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("bin_edge", len(bins))
        ds.createDimension("bin", n_bins)
        ds.createDimension("slice", len(slices))

        bins_var = ds.createVariable("bins", "f8", ("bin_edge",), zlib=True)
        bins_var[:] = bins

        hist_var = ds.createVariable("hist", "i8", ("slice", "bin"), zlib=True)
        hist_var[:, :] = np.stack([s.hist for s in slices])

        valid_time_var = ds.createVariable("valid_time", "f8", ("slice",))
        valid_time_var.units = _TIME_UNITS
        valid_time_var[:] = [_dt_to_seconds(s.valid_time) for s in slices]

        if has_lead_hours:
            lead_var = ds.createVariable("lead_hours", "f8", ("slice",), zlib=True)
            lead_var[:] = [s.lead_hours if s.lead_hours is not None else _NO_LEAD_HOURS for s in slices]

        if has_member_id:
            member_var = ds.createVariable("member_id", str, ("slice",))
            for i, s in enumerate(slices):
                member_var[i] = s.member_id if s.member_id is not None else ""

        ds.varname = varname
        ds.edge_trim = edge_trim
        ds.clip_negative_to_zero = int(clip_negative_to_zero)
        ds.n_source_files = n_source_files


def read_histogram_file(path: str) -> HistogramFileContents:
    with netCDF4.Dataset(path, "r") as ds:
        bins = np.asarray(ds.variables["bins"][:])
        hist = np.asarray(ds.variables["hist"][:, :])
        valid_times = [_seconds_to_dt(t) for t in ds.variables["valid_time"][:]]

        n_slices = ds.dimensions["slice"].size
        lead_hours_arr = ds.variables["lead_hours"][:] if "lead_hours" in ds.variables else None
        member_ids_arr = list(ds.variables["member_id"][:]) if "member_id" in ds.variables else None

        slices = []
        for i in range(n_slices):
            lead_hours = None
            if lead_hours_arr is not None:
                raw = float(lead_hours_arr[i])
                lead_hours = None if raw == _NO_LEAD_HOURS else raw
            member_id = None
            if member_ids_arr is not None:
                raw_member = str(member_ids_arr[i])
                member_id = raw_member if raw_member != "" else None
            slices.append(HistogramSlice(
                valid_time=valid_times[i], hist=hist[i, :], lead_hours=lead_hours, member_id=member_id,
            ))

        return HistogramFileContents(
            bins=bins,
            slices=slices,
            varname=str(ds.varname),
            edge_trim=int(ds.edge_trim),
            clip_negative_to_zero=bool(ds.clip_negative_to_zero),
            n_source_files=int(ds.n_source_files),
        )
