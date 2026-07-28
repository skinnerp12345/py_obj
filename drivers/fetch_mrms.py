"""Standalone script: fetch MRMS observations from the public noaa-mrms-pds
AWS S3 archive, in either of two independent modes.

Model-driven mode (the original mode): for each file in a directory of model
output (e.g. WoFS, which has no local matching MRMS data), derives its
valid_time (via python_obj.regrid.read_valid_time_only's flexible mechanism
-- a ready-made valid_time string attribute, or init+lead arithmetic,
depending on the model), lists that day's MRMS files in the public archive
(one HTTPS request per distinct day, cached), finds the nearest available
MRMS timestamp within a tolerance, and downloads it.

Date-driven mode: given 'dates' (an explicit list) or 'date_range' (an
inclusive [start, end] pair, expanded to daily strings -- same convention as
batch_config.py's cases: section), fetches EVERY MRMS file found for each
requested day, no model files or valid-time matching involved at all.

Both modes reuse the same listing (_list_mrms_day)/download (_download_file)
machinery and write fetched files in the exact directory/filename convention
already used by test_mrms/ and already consumed unmodified by
discover_mrms_files()/interpolate_mrms.py -- <output_dir>/<YYYYMMDD>/
<original S3 filename>, no renaming -- so fetched output can be used
directly as an 'interpolation.raw_mrms_dir' with zero downstream changes,
regardless of which mode produced it.

The bucket is public; no AWS credentials or SDK needed, just plain HTTPS
(the `requests` library). Configured entirely via the shared
python_obj/configs/config.yaml (its 'fetch_mrms:' section).

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/fetch_mrms.py [path/to/config.yaml]

If no config path is given, uses python_obj/configs/config.yaml.
"""

import glob
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import requests

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.config import FetchMrmsConfig, load_config, require_section
from python_obj.regrid import read_valid_time_only
from python_obj.time_utils import nearest_within_tolerance

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_DATE_FORMAT = "%Y%m%d"  # matches batch_config.py's _DEFAULT_DATE_FORMAT


@dataclass
class FetchFileResult:
    model_input_path: str | None  # None in date-driven mode (see source_date instead)
    model_valid_time: datetime | None
    mrms_key: str | None
    mrms_local_path: str | None
    status: str  # "downloaded" | "already_exists" | "no_match_within_tolerance" | "failed"
    error: str | None = None
    source_date: str | None = None  # set (YYYYMMDD) in date-driven mode; None in model-driven mode


@dataclass
class FetchSummary:
    n_total: int
    results: list = field(default_factory=list)  # list[FetchFileResult]

    def __str__(self) -> str:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        lines = [f"FetchSummary: {self.n_total} total -- " + ", ".join(f"{v} {k}" for k, v in counts.items())]
        for r in self.results:
            if r.status not in ("downloaded", "already_exists"):
                label = r.model_input_path if r.model_input_path is not None else f"date={r.source_date}"
                lines.append(f"  {r.status.upper()}: {label}: {r.error or '(no MRMS time in tolerance)'}")
        return "\n".join(lines)


def _expand_date_range(start_str: str, end_str: str, date_format: str = _DATE_FORMAT) -> list[str]:
    """Inclusive [start, end] -> daily list of date strings -- identical
    convention to batch_config.py's own date_range expansion (same format,
    same timedelta(days=1) step, same end-before-start check)."""
    start = datetime.strptime(start_str, date_format)
    end = datetime.strptime(end_str, date_format)
    if end < start:
        raise ValueError(f"date_range: end ({end_str}) is before start ({start_str})")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime(date_format))
        current += timedelta(days=1)
    return dates


def _list_mrms_day(bucket: str, product: str, day: str) -> list[tuple[datetime, str, int]]:
    """List every MRMS file for one YYYYMMDD day-prefix via the bucket's
    public HTTPS REST API (no credentials needed). Pages through
    continuation tokens if a day ever exceeds one LIST page (not expected
    at ~720 files/day for this product, but handled rather than silently
    truncated)."""
    prefix = f"CONUS/{product}/{day}/"
    entries: list[tuple[datetime, str, int]] = []
    continuation_token = None

    while True:
        params = {"list-type": "2", "prefix": prefix}
        if continuation_token:
            params["continuation-token"] = continuation_token
        resp = requests.get(f"https://{bucket}.s3.amazonaws.com/", params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        for contents in root.findall(f"{_S3_NS}Contents"):
            key = contents.findtext(f"{_S3_NS}Key")
            size = int(contents.findtext(f"{_S3_NS}Size"))
            basename = os.path.basename(key)
            file_prefix = f"MRMS_{product}_{day}-"
            if not (basename.startswith(file_prefix) and basename.endswith(".grib2.gz")):
                continue
            hhmmss = basename[len(file_prefix):-len(".grib2.gz")]
            timestamp = datetime.strptime(day + hhmmss, "%Y%m%d%H%M%S")
            entries.append((timestamp, key, size))

        is_truncated = root.findtext(f"{_S3_NS}IsTruncated") == "true"
        if not is_truncated:
            break
        continuation_token = root.findtext(f"{_S3_NS}NextContinuationToken")

    return entries


def _download_file(bucket: str, key: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    part_path = local_path + ".part"
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(part_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    os.rename(part_path, local_path)


def _run_model_driven(fm: FetchMrmsConfig) -> FetchSummary:
    model_files = sorted(glob.glob(os.path.join(fm.model_input_dir, fm.file_pattern)))
    if not model_files:
        raise FileNotFoundError(
            f"No files matching '{fm.file_pattern}' found under '{fm.model_input_dir}'"
        )
    if fm.max_files is not None:
        model_files = model_files[:fm.max_files]
    print(f"Found {len(model_files)} model files under '{fm.model_input_dir}'")

    day_cache: dict[str, list[tuple[datetime, str, int]]] = {}
    results: list[FetchFileResult] = []

    for fp in model_files:
        try:
            valid_time = read_valid_time_only(
                fp,
                init_attr=fm.init_attr or "initializationTime",
                lead_attr=fm.lead_attr or "forecastHour",
                lead_units=fm.lead_units,
                init_format=fm.init_format or "%Y%m%d%H",
                valid_time_attr=fm.valid_time_attr,
                valid_time_format=fm.valid_time_format,
            )
        except Exception as exc:  # noqa: BLE001 -- one file's failure never aborts the run
            results.append(FetchFileResult(
                model_input_path=fp, model_valid_time=None, mrms_key=None, mrms_local_path=None,
                status="failed", error=f"{type(exc).__name__}: {exc}",
            ))
            continue

        day = valid_time.strftime("%Y%m%d")
        if day not in day_cache:
            print(f"Listing MRMS archive for {day} ...")
            try:
                day_cache[day] = _list_mrms_day(fm.s3_bucket, fm.mrms_product, day)
            except Exception as exc:  # noqa: BLE001
                results.append(FetchFileResult(
                    model_input_path=fp, model_valid_time=valid_time, mrms_key=None, mrms_local_path=None,
                    status="failed", error=f"listing {day} failed: {type(exc).__name__}: {exc}",
                ))
                continue

        entries = day_cache[day]
        nearest = nearest_within_tolerance(valid_time, [t for t, _, _ in entries], fm.tolerance_minutes)
        if nearest is None:
            results.append(FetchFileResult(
                model_input_path=fp, model_valid_time=valid_time, mrms_key=None, mrms_local_path=None,
                status="no_match_within_tolerance",
            ))
            continue

        _, key, size = next(e for e in entries if e[0] == nearest)
        out_subdir = os.path.join(fm.output_dir, day) if fm.mirror_subdirs else fm.output_dir
        local_path = os.path.join(out_subdir, os.path.basename(key))

        if fm.skip_existing and os.path.exists(local_path) and os.path.getsize(local_path) == size:
            results.append(FetchFileResult(
                model_input_path=fp, model_valid_time=valid_time, mrms_key=key, mrms_local_path=local_path,
                status="already_exists",
            ))
            continue

        try:
            _download_file(fm.s3_bucket, key, local_path)
            results.append(FetchFileResult(
                model_input_path=fp, model_valid_time=valid_time, mrms_key=key, mrms_local_path=local_path,
                status="downloaded",
            ))
        except Exception as exc:  # noqa: BLE001
            results.append(FetchFileResult(
                model_input_path=fp, model_valid_time=valid_time, mrms_key=key, mrms_local_path=local_path,
                status="failed", error=f"{type(exc).__name__}: {exc}",
            ))

    return FetchSummary(n_total=len(results), results=results)


def _run_date_driven(fm: FetchMrmsConfig) -> FetchSummary:
    dates = list(fm.dates) if fm.dates is not None else _expand_date_range(*fm.date_range)
    print(f"Fetching every MRMS file for {len(dates)} date(s): {dates[0]}"
          + (f" - {dates[-1]}" if len(dates) > 1 else ""))

    results: list[FetchFileResult] = []
    # (date, key, size) across every requested day, listed up front so
    # max_files can cap the TOTAL across all dates combined, not per-day.
    all_entries: list[tuple[str, str, int]] = []
    for day in dates:
        print(f"Listing MRMS archive for {day} ...")
        try:
            entries = _list_mrms_day(fm.s3_bucket, fm.mrms_product, day)
        except Exception as exc:  # noqa: BLE001 -- one day's failure never aborts the run
            results.append(FetchFileResult(
                model_input_path=None, model_valid_time=None, mrms_key=None, mrms_local_path=None,
                status="failed", error=f"listing {day} failed: {type(exc).__name__}: {exc}", source_date=day,
            ))
            continue
        all_entries.extend((day, key, size) for _, key, size in entries)

    if fm.max_files is not None:
        all_entries = all_entries[:fm.max_files]

    for day, key, size in all_entries:
        out_subdir = os.path.join(fm.output_dir, day) if fm.mirror_subdirs else fm.output_dir
        local_path = os.path.join(out_subdir, os.path.basename(key))

        if fm.skip_existing and os.path.exists(local_path) and os.path.getsize(local_path) == size:
            results.append(FetchFileResult(
                model_input_path=None, model_valid_time=None, mrms_key=key, mrms_local_path=local_path,
                status="already_exists", source_date=day,
            ))
            continue

        try:
            _download_file(fm.s3_bucket, key, local_path)
            results.append(FetchFileResult(
                model_input_path=None, model_valid_time=None, mrms_key=key, mrms_local_path=local_path,
                status="downloaded", source_date=day,
            ))
        except Exception as exc:  # noqa: BLE001
            results.append(FetchFileResult(
                model_input_path=None, model_valid_time=None, mrms_key=key, mrms_local_path=local_path,
                status="failed", error=f"{type(exc).__name__}: {exc}", source_date=day,
            ))

    return FetchSummary(n_total=len(results), results=results)


def run_one_case(config_path: str) -> FetchSummary:
    cfg = load_config(config_path)
    fm = require_section(cfg.fetch_mrms, "fetch_mrms", config_path)

    if fm.model_input_dir is not None:
        summary = _run_model_driven(fm)
    else:
        summary = _run_date_driven(fm)

    print(summary)
    return summary


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
    run_one_case(config_path)


if __name__ == "__main__":
    main()
