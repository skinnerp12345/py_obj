"""Standalone script: fetch MRMS observations matching a directory of model
files' valid times from the public noaa-mrms-pds AWS S3 archive.

For each model file (e.g. WoFS, which has no local matching MRMS data),
derives its valid_time (via python_obj.regrid.read_valid_time_only's
flexible mechanism -- a ready-made valid_time string attribute, or
init+lead arithmetic, depending on the model), lists that day's MRMS files
in the public archive (one HTTPS request per distinct day, cached), finds
the nearest available MRMS timestamp within a tolerance, and downloads it.

Fetched files are written in the exact directory/filename convention
already used by test_mrms/ and already consumed unmodified by
discover_mrms_files()/interpolate_mrms.py -- <output_dir>/<YYYYMMDD>/
<original S3 filename>, no renaming -- so the fetched output can be used
directly as an 'interpolation.raw_mrms_dir' with zero downstream changes.

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
from datetime import datetime

import requests

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.config import load_config, require_section
from python_obj.regrid import read_valid_time_only
from python_obj.time_utils import nearest_within_tolerance

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


@dataclass
class FetchFileResult:
    model_input_path: str
    model_valid_time: datetime | None
    mrms_key: str | None
    mrms_local_path: str | None
    status: str  # "downloaded" | "already_exists" | "no_match_within_tolerance" | "failed"
    error: str | None = None


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
                lines.append(f"  {r.status.upper()}: {r.model_input_path}: {r.error or '(no MRMS time in tolerance)'}")
        return "\n".join(lines)


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


def run_one_case(config_path: str) -> FetchSummary:
    cfg = load_config(config_path)
    fm = require_section(cfg.fetch_mrms, "fetch_mrms", config_path)

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

    summary = FetchSummary(n_total=len(results), results=results)
    print(summary)
    return summary


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
    run_one_case(config_path)


if __name__ == "__main__":
    main()
