"""Standalone driver: match many forecast CASE files (e.g. init_snapshot
files, one per forecast case -- many members x many lead times) against a
truth-object-file archive, one case at a time -- NOT one whole-directory
run_matching_series() call across everything, which is not viable when the
forecast archive holds many large, temporally-overlapping cases.

The real, motivating problem this solves (found via direct measurement, not
assumed): a large ensemble/long-forecast object file's decompressed labels
array can be multiple GB (confirmed real example: 5 members x 133 lead times
x 1059x1799 grid = 5.07 GB, transient peak 10.14 GB reading it in one shot).
Holding many such files' full label arrays simultaneously -- as a single
run_matching_series() call over a whole directory would try to do -- is not
viable on a memory-constrained machine. matching_pipeline.run_matching_series
already reads forecast files via object_io.iter_object_slices_lazy (one
member's block at a time, ~2 GB peak for the case above) and matches each
slice immediately rather than accumulating every slice first; this driver
adds the two remaining pieces needed to run that safely at real scale:

1. Per-case truth-file restriction. Loading the ENTIRE truth archive for
   every case wastes memory and time when only a small, real-time-bounded
   subset of it is ever relevant to any one case's own forecast valid-time
   span. Truth files are filtered by their own filename-encoded real
   valid_time (this library's established obj_obs_<YYYYMMDD>_<HHMMSS>.nc
   "single" grouping convention, see id_pipeline.py) against the case's own
   valid-time range (read cheaply via object_io.read_valid_time_range, never
   touching labels), with a buffer of max_time_offset_minutes on each side --
   exactly the tolerance run_matching_series itself applies when it later
   looks up each file's real valid_time, so this is a safe pre-filter (can
   only ever be too generous, never too strict), not a second, independent
   notion of "close enough".

2. One consolidated match file per case, not one per (case, valid_time) pair.
   run_matching_series's default file_grouping="per_time" names output purely
   by forecast valid_time (match_<valid_time>.nc), with no case/init-time key
   -- daily-issued, multi-day-long forecast cases have real, overlapping
   valid-time ranges (confirmed real example: consecutive daily 133-lead-hour
   MPAS cases overlap by ~4.5 of their 5.5-day span), so running every case's
   per-time output into one shared directory would silently overwrite most of
   it -- the same collision class file_grouping="init_snapshot" was built to
   fix for object files. This driver instead calls run_matching_series with
   file_grouping="init_snapshot": one match_init_<init_time>.nc file per case
   (every member x every lead time, mirroring the forecast object file's own
   init_snapshot shape exactly), named after the case's own unique init_time
   -- inherently collision-free across cases even in one shared flat
   output_dir, and subsettable back into individual (member, valid_time)
   slices the same way object files already are.

Configured via the shared MatchingConfig schema (max_boundary_disp_km,
max_centroid_disp_km, ti_threshold, max_time_offset_minutes,
truth_object_dir, forecast_object_dir, output_dir, file_pattern) -- no schema
changes needed; only the discovery/looping strategy differs from
run_matching.py's whole-directory approach.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/run_matching_per_case.py [path/to/config.yaml]

If no config path is given, uses python_obj/configs/config.yaml.
"""

import glob
import os
import re
import sys
from datetime import datetime, timedelta

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.config import load_config, require_section
from python_obj.obj_core import MatchingSummary, read_valid_time_range, run_matching_series

_FILENAME_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")


def discover_forecast_cases(forecast_object_dir: str, file_pattern: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(forecast_object_dir, "**", file_pattern), recursive=True))
    if not files:
        raise FileNotFoundError(f"No forecast case files matching '{file_pattern}' found under '{forecast_object_dir}'")
    return files


def _parse_filename_timestamp(path: str) -> datetime | None:
    """Best-effort: parse a YYYYMMDD_HHMMSS timestamp out of a filename, per
    this library's own established naming convention (id_pipeline.py's
    obj_obs_<YYYYMMDD>_<HHMMSS>.nc). Only used as a cheap filename-based
    PRE-filter (see module docstring) -- never the sole source of truth for
    which files actually get matched, so a non-conforming filename just makes
    the pre-filter conservative (falls back to "always include"), not
    incorrect.
    """
    m = _FILENAME_TIMESTAMP_RE.search(os.path.basename(path))
    if m is None:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def discover_truth_files_near_window(
    truth_object_dir: str, file_pattern: str,
    window_start: datetime, window_end: datetime, buffer_minutes: float,
) -> list[str]:
    """Cheap (filename-only, no file I/O) pre-filter of a large truth-file
    archive down to just the files whose own real valid_time (parsed from the
    filename) falls within [window_start - buffer, window_end + buffer].
    Safe by construction: run_matching_series applies the same
    max_time_offset_minutes tolerance itself when it reads each file's real
    valid_time, so this can only ever include a few extra files near the
    boundary, never exclude a file that would actually have been used.
    """
    buffer = timedelta(minutes=buffer_minutes)
    lo, hi = window_start - buffer, window_end + buffer

    all_files = sorted(glob.glob(os.path.join(truth_object_dir, "**", file_pattern), recursive=True))
    if not all_files:
        raise FileNotFoundError(f"No truth files matching '{file_pattern}' found under '{truth_object_dir}'")

    kept = [f for f in all_files if (ts := _parse_filename_timestamp(f)) is None or lo <= ts <= hi]
    if not kept:
        raise FileNotFoundError(
            f"No truth files found within [{lo}, {hi}] under '{truth_object_dir}' "
            f"(window derived from a forecast case spanning [{window_start}, {window_end}])"
        )
    return kept


def run_one_case(config_path: str, case_file: str) -> MatchingSummary:
    cfg = load_config(config_path)
    match = require_section(cfg.matching, "matching", config_path)

    window_start, window_end = read_valid_time_range(case_file)
    truth_files = discover_truth_files_near_window(
        match.truth_object_dir, match.file_pattern, window_start, window_end, match.max_time_offset_minutes,
    )

    case_key = os.path.splitext(os.path.basename(case_file))[0]

    print(
        f"[{case_key}] forecast valid-time span: {window_start} to {window_end}; "
        f"{len(truth_files)} truth files selected (of the full archive) -> {match.output_dir}"
    )

    return run_matching_series(
        truth_files, [case_file],
        max_boundary_disp_km=match.max_boundary_disp_km,
        max_centroid_disp_km=match.max_centroid_disp_km,
        ti_threshold=match.ti_threshold,
        output_dir=match.output_dir,
        max_time_offset_minutes=match.max_time_offset_minutes,
        file_grouping="init_snapshot",
    )


def run_all_cases_sequential(config_path: str) -> dict[str, MatchingSummary]:
    """Run every discovered forecast case sequentially, in one process -- the
    recommended mode on a memory-constrained machine (see module docstring).
    Use run_matching_per_case_batch.py's run_cases_in_parallel(..., n_workers=N)
    instead only after confirming your machine has enough RAM for N cases'
    peak memory simultaneously."""
    cfg = load_config(config_path)
    match = require_section(cfg.matching, "matching", config_path)
    case_files = discover_forecast_cases(match.forecast_object_dir, match.file_pattern)
    print(f"Found {len(case_files)} forecast case files under '{match.forecast_object_dir}'")

    results = {}
    for i, case_file in enumerate(case_files, 1):
        print(f"--- case {i}/{len(case_files)}: {os.path.basename(case_file)} ---")
        results[case_file] = run_one_case(config_path, case_file)
    return results


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
    results = run_all_cases_sequential(config_path)
    total_written = sum(len(s.output_paths) for s in results.values())
    total_skipped = sum(len(s.skipped_forecast_times) for s in results.values())
    print(f"\nDone: {len(results)} cases, {total_written} match files written, {total_skipped} forecast times skipped total")


if __name__ == "__main__":
    main()
