"""Standalone script: match two already-existing directories of Step 4 object
files (a truth series, a forecast series -- from any source, produced
however) into hits/misses/false_alarms/truth_extras/forecast_extras.

No identification/tracking here (see identify_track_mrms.py or
identify_track_model.py for that); this only ever reads pre-existing object
files and matches between them. A thin driver over
python_obj.obj_core.run_matching_series; does not modify anything else in
python_obj/. Configured entirely via the shared python_obj/configs/config.yaml (its
'matching:' section).

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/run_matching.py [path/to/config.yaml]

If no config path is given, uses python_obj/configs/config.yaml.
"""

import glob
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.config import load_config, require_section
from python_obj.obj_core import MatchingSummary, run_matching_series


def _discover(directory: str, pattern: str, label: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(directory, "**", pattern), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No {label} object files matching '{pattern}' found under '{directory}'"
        )
    return files


def run_one_case(config_path: str) -> MatchingSummary:
    cfg = load_config(config_path)
    match = require_section(cfg.matching, "matching", config_path)

    truth_files = _discover(match.truth_object_dir, match.file_pattern, "truth")
    forecast_files = _discover(match.forecast_object_dir, match.file_pattern, "forecast")
    print(f"Found {len(truth_files)} truth files under '{match.truth_object_dir}'")
    print(f"Found {len(forecast_files)} forecast files under '{match.forecast_object_dir}'")
    print(
        f"Matching: max_boundary_disp_km={match.max_boundary_disp_km}, "
        f"max_centroid_disp_km={match.max_centroid_disp_km}, ti_threshold={match.ti_threshold}, "
        f"max_time_offset_minutes={match.max_time_offset_minutes}"
    )

    return run_matching_series(
        truth_files, forecast_files,
        max_boundary_disp_km=match.max_boundary_disp_km,
        max_centroid_disp_km=match.max_centroid_disp_km,
        ti_threshold=match.ti_threshold,
        output_dir=match.output_dir,
        max_time_offset_minutes=match.max_time_offset_minutes,
    )


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
    summary = run_one_case(config_path)
    print(
        f"\nDone: wrote {len(summary.output_paths)} match files. "
        f"Skipped forecast times (no truth within tolerance): {len(summary.skipped_forecast_times)}"
    )
    if summary.skipped_forecast_times:
        for t in summary.skipped_forecast_times:
            print(f"  skipped: {t}")


if __name__ == "__main__":
    main()
