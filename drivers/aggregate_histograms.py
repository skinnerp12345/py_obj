"""Standalone script: demonstrates building SUBSET histograms out of the
per-day (MRMS)/per-forecast (model) output files build_histogram_mrms.py/
build_histogram_model.py produce -- an hour-of-day MRMS climatology subset
and a forecast-lead-hour-range ("day N of the forecast") model subset -- then
a real matched-percentile-threshold computation between the two full
distributions (the actual end goal: "what model dBZ value corresponds to the
same percentile as MRMS's 40 dBZ?").

Not a from-scratch pipeline stage -- this is a thin driver over
python_obj.histogram.aggregate, showing the reusable functions there working
against real output. Reuses the same 'histogram_observations:'/
'histogram_model:' config sections build_histogram_mrms.py/
build_histogram_model.py already read (their own output_dir fields tell this
script where to find their output).

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/aggregate_histograms.py [path/to/config.yaml] [source_threshold_dbz]

If no config path is given, uses python_obj/configs/config.yaml.
source_threshold_dbz defaults to 40.0 if omitted.
"""

import glob
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.config import load_config, require_section
from python_obj.histogram import (
    by_hour_of_day,
    by_lead_hours_range,
    match_percentile_threshold,
    read_histogram_file,
    sum_histograms,
)


def run_one_case(config_path: str, source_threshold_dbz: float = 40.0) -> dict:
    cfg = load_config(config_path)
    obs_cfg = require_section(cfg.histogram_observations, "histogram_observations", config_path)
    model_cfg = require_section(cfg.histogram_model, "histogram_model", config_path)

    mrms_files = sorted(glob.glob(os.path.join(obs_cfg.output_dir, "hist_mrms_*.nc")))
    model_files = sorted(glob.glob(os.path.join(model_cfg.output_dir, "hist_model_*.nc")))
    if not mrms_files:
        raise FileNotFoundError(f"No MRMS histogram files found under '{obs_cfg.output_dir}' -- run build_histogram_mrms.py first.")
    if not model_files:
        raise FileNotFoundError(f"No model histogram files found under '{model_cfg.output_dir}' -- run build_histogram_model.py first.")
    print(f"Found {len(mrms_files)} MRMS histogram file(s), {len(model_files)} model histogram file(s)")

    # --- full distributions (no subsetting) ---
    mrms_bins, mrms_full = sum_histograms(mrms_files)
    model_bins, model_full = sum_histograms(model_files)
    print(f"MRMS full distribution: {mrms_full.sum()} total counts")
    print(f"Model full distribution: {model_full.sum()} total counts")

    # --- hour-of-day MRMS subset: use whichever hour the first available
    # slice actually has, so this demo always matches real data regardless
    # of exactly which hours are present in the bundled/real files ---
    first_slice_hour = read_histogram_file(mrms_files[0]).slices[0].valid_time.hour
    _, mrms_hour_subset = sum_histograms(mrms_files, predicate=by_hour_of_day(first_slice_hour))
    print(f"MRMS subset (hour={first_slice_hour:02d}Z climatology): {mrms_hour_subset.sum()} counts")

    # --- lead-hour-bucketed model subset: "day 1" (lead 0-24h) vs whatever
    # lead times actually exist beyond that, if any ---
    _, model_day1 = sum_histograms(model_files, predicate=by_lead_hours_range(0.0, 24.0))
    print(f"Model subset (lead 0-24h, 'day 1'): {model_day1.sum()} counts")

    # --- matched-percentile threshold: the actual stated end goal ---
    source_pct, target_value = match_percentile_threshold(
        mrms_bins, mrms_full, source_threshold_dbz, model_bins, model_full,
    )
    print(
        f"\nMatched-percentile threshold: MRMS {source_threshold_dbz:.1f} dBZ is the "
        f"{source_pct * 100:.1f}th percentile of the MRMS distribution -- "
        f"the model's {source_pct * 100:.1f}th percentile is {target_value:.1f} dBZ"
    )

    return {
        "mrms_full": mrms_full, "model_full": model_full,
        "mrms_hour_subset": mrms_hour_subset, "model_day1_subset": model_day1,
        "source_percentile": source_pct, "target_value": target_value,
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
    source_threshold_dbz = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
    run_one_case(config_path, source_threshold_dbz)


if __name__ == "__main__":
    main()
