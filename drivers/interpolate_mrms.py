"""Standalone script: interpolate raw native-grid MRMS to a fixed target grid.

Interpolation only, does not identify or track objects (see
identify_track_mrms.py for that). A thin driver over
python_obj.regrid.run_batch_interpolation; does not modify anything else in
python_obj/. Configured entirely via the shared python_obj/configs/config.yaml (only
its 'interpolation:' section is used).

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/interpolate_mrms.py [path/to/config.yaml]

If no config path is given, uses python_obj/configs/config.yaml.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.config import load_config, require_section
from python_obj.regrid import BatchSummary, build_corner_spacing_grid, run_batch_interpolation


def run_one_case(config_path: str) -> BatchSummary:
    cfg = load_config(config_path)
    interp = require_section(cfg.interpolation, "interpolation", config_path)

    print(f"Interpolating MRMS files from '{interp.raw_mrms_dir}' -> '{interp.interp_mrms_dir}'")
    print(f"File pattern: '{interp.file_pattern}'")
    if interp.date_range:
        print(f"Date range: {interp.date_range[0]} - {interp.date_range[1]}")
    if interp.max_files:
        print(f"max_files: {interp.max_files} (smoke-test/dry-run cap)")

    if interp.target_grid_file is not None:
        print(f"Target grid: '{interp.target_grid_file}' (lat='{interp.target_lat_name}', lon='{interp.target_lon_name}')")
        return run_batch_interpolation(
            input_dir=interp.raw_mrms_dir,
            output_dir=interp.interp_mrms_dir,
            target_grid_file=interp.target_grid_file,
            target_lat_name=interp.target_lat_name,
            target_lon_name=interp.target_lon_name,
            weight_cache_dir=interp.weight_cache_dir,
            n_workers=interp.n_workers,
            date_range=interp.date_range,
            max_files=interp.max_files,
            file_pattern=interp.file_pattern,
        )

    print(
        f"Target grid: computed, sw_corner=({interp.target_grid_sw_lat}, {interp.target_grid_sw_lon}), "
        f"dx_km={interp.target_grid_dx_km}, dy_km={interp.target_grid_dy_km or interp.target_grid_dx_km}, "
        f"nx={interp.target_grid_nx}, ny={interp.target_grid_ny}"
    )
    target_grid = build_corner_spacing_grid(
        sw_lat=interp.target_grid_sw_lat,
        sw_lon=interp.target_grid_sw_lon,
        dx_km=interp.target_grid_dx_km,
        dy_km=interp.target_grid_dy_km,
        nx=interp.target_grid_nx,
        ny=interp.target_grid_ny,
        true_lat_1=interp.target_grid_true_lat_1,
        true_lat_2=interp.target_grid_true_lat_2,
        cen_lat=interp.target_grid_cen_lat,
        cen_lon=interp.target_grid_cen_lon,
    )
    return run_batch_interpolation(
        input_dir=interp.raw_mrms_dir,
        output_dir=interp.interp_mrms_dir,
        target_grid=target_grid,
        weight_cache_dir=interp.weight_cache_dir,
        n_workers=interp.n_workers,
        date_range=interp.date_range,
        max_files=interp.max_files,
        file_pattern=interp.file_pattern,
    )


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
    summary = run_one_case(config_path)

    print(f"\nDone: {summary.n_success}/{summary.n_total} files interpolated, {summary.n_failed} failed.")
    if summary.failures:
        print("Failures:")
        for fail in summary.failures:
            print(f"  {fail.input_path}: {fail.error}")


if __name__ == "__main__":
    main()
