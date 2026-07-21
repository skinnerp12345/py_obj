"""Standalone script: build one composite-reflectivity (or any configured
variable) distribution histogram per YYYYMMDD day of already-interpolated
MRMS. Generalizes python_base/mrms_dz_histogram_base.py -- configurable bins/
variable, and (new) preserves one histogram slice per input file (tagged
with its real valid_time) inside each day's output file, rather than
collapsing straight to one flat total -- this is what lets
python_obj.histogram.aggregate later rebuild subsets (e.g. an hour-of-day
climatology) that the original script's output could never support.

A thin driver over python_obj.histogram; does not modify anything else in
python_obj/. Configured entirely via the shared python_obj/configs/config.yaml
(its 'histogram_observations:' section).

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/build_histogram_mrms.py [path/to/config.yaml]

If no config path is given, uses python_obj/configs/config.yaml.
"""

import glob
import os
import sys
from collections import defaultdict

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.config import HistogramObservationConfig, load_config, require_section
from python_obj.histogram import HistogramSlice, compute_histogram, default_bin_edges, write_histogram_file
from python_obj.obj_core import conus_mask, conus_mask_east
from python_obj.regrid import load_mrms_netcdf


def _discover_by_day(interp_mrms_dir: str) -> dict[str, list[str]]:
    files = sorted(glob.glob(os.path.join(interp_mrms_dir, "**", "*.nc"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No interpolated MRMS files found under '{interp_mrms_dir}'")
    by_day: dict[str, list[str]] = defaultdict(list)
    for f in files:
        day = os.path.basename(os.path.dirname(f))
        by_day[day].append(f)
    return by_day


def run_one_case(config_path: str) -> list[str]:
    cfg = load_config(config_path)
    hist_cfg: HistogramObservationConfig = require_section(cfg.histogram_observations, "histogram_observations", config_path)

    bins = default_bin_edges(hist_cfg.bin_min, hist_cfg.bin_max, hist_cfg.bin_width)
    by_day = _discover_by_day(hist_cfg.interp_mrms_dir)
    n_total = sum(len(v) for v in by_day.values())
    print(
        f"Found {n_total} interpolated MRMS files across {len(by_day)} day(s) under '{hist_cfg.interp_mrms_dir}'"
    )
    print(
        f"Building histograms: var_name={hist_cfg.var_name}, bins=[{hist_cfg.bin_min}, {hist_cfg.bin_max}] "
        f"by {hist_cfg.bin_width}, edge_trim={hist_cfg.edge_trim}, clip_negative_to_zero={hist_cfg.clip_negative_to_zero}"
    )

    # mask is computed once from the first file's own lat/lon (a fixed grid
    # across the whole series, same assumption identify_track_mrms.py makes)
    # and applied as NaN, NOT the 0.0 the object-ID pipeline uses -- a masked
    # cell must be excluded from the distribution entirely, not counted as a
    # fake clear-air reading (compute_histogram already excludes NaN via its
    # existing np.isfinite filter, so no core histogram code changes needed).
    mask = None
    if hist_cfg.mask != "none":
        first_day = sorted(by_day)[0]
        first = load_mrms_netcdf(sorted(by_day[first_day])[0], varname=hist_cfg.var_name, lat_name=hist_cfg.lat_name, lon_name=hist_cfg.lon_name)
        mask_fn = conus_mask_east if hist_cfg.mask == "conus_east" else conus_mask
        mask = mask_fn(first.lat2d, first.lon2d)
        print(f"Mask '{hist_cfg.mask}' excludes {mask.mean() * 100:.1f}% of grid cells")

    os.makedirs(hist_cfg.output_dir, exist_ok=True)
    out_paths = []
    for day in sorted(by_day):
        files = sorted(by_day[day])
        slices = []
        for f in files:
            field = load_mrms_netcdf(f, varname=hist_cfg.var_name, lat_name=hist_cfg.lat_name, lon_name=hist_cfg.lon_name)
            data = field.data if mask is None else np.where(mask, np.nan, field.data)
            # out-of-range real values are now clamped into the nearest edge
            # bin rather than dropped (see compute_histogram's docstring), so
            # MRMS's -999 "no coverage" sentinel must be excluded explicitly
            # here -- it's a real finite value in field.data, not NaN, and
            # would otherwise be silently clamped in as a fake -20 dBZ reading.
            counts = compute_histogram(
                data, bins, edge_trim=hist_cfg.edge_trim, clip_negative_to_zero=hist_cfg.clip_negative_to_zero,
                missing_value=field.missing_value,
            )
            slices.append(HistogramSlice(valid_time=field.valid_time, hist=counts))

        out_path = os.path.join(hist_cfg.output_dir, f"hist_mrms_{day}.nc")
        write_histogram_file(
            out_path, bins, slices, hist_cfg.var_name, hist_cfg.edge_trim, hist_cfg.clip_negative_to_zero, files,
        )
        out_paths.append(out_path)
        print(f"  wrote {out_path} ({len(slices)} slices, {sum(s.hist.sum() for s in slices)} total counts)")

    return out_paths


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
    out_paths = run_one_case(config_path)
    print(f"\nDone: wrote {len(out_paths)} histogram files (one per day).")


if __name__ == "__main__":
    main()
