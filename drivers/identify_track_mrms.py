"""Standalone script: identify (and optionally track) thunderstorm objects in
already-interpolated MRMS data.

Reads the interpolated MRMS NetCDF files produced by interpolate_mrms.py
(same format as python_obj's own batch interpolation) and writes
self-contained object files, one per available output time by default
(file_grouping -- the natural shape for observations, which have no
member/ensemble concept). A thin driver over
python_obj.obj_core.run_object_id_series; does not modify anything else in
python_obj/. Configured entirely via the shared python_obj/configs/config.yaml (its
'observations:' and 'linear_classification:' sections).

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/identify_track_mrms.py [path/to/config.yaml]

If no config path is given, uses python_obj/configs/config.yaml.
Requires observations.interp_mrms_dir to already be populated (run
interpolate_mrms.py first).
"""

import glob
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.config import load_config, require_section
from python_obj.obj_core import SeriesEntry, conus_mask, conus_mask_east, run_object_id_series
from python_obj.regrid import load_mrms_netcdf


def run_one_case(config_path: str) -> list[str]:
    cfg = load_config(config_path)
    obs = require_section(cfg.observations, "observations", config_path)
    linear = require_section(cfg.linear_classification, "linear_classification", config_path)

    files = sorted(glob.glob(os.path.join(obs.interp_mrms_dir, "**", "*.nc"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No interpolated MRMS files found under '{obs.interp_mrms_dir}' -- run interpolate_mrms.py first."
        )
    print(f"Found {len(files)} interpolated MRMS files under '{obs.interp_mrms_dir}'")

    loader = lambda fp: load_mrms_netcdf(fp)
    manifest = [SeriesEntry(valid_time=loader(f).valid_time, filepath=f, member_id=None) for f in files]

    mask = None
    if obs.mask != "none":
        first = loader(files[0])
        mask_fn = conus_mask_east if obs.mask == "conus_east" else conus_mask
        mask = mask_fn(first.lat2d, first.lon2d)
        print(f"Mask '{obs.mask}' excludes {mask.mean() * 100:.1f}% of grid cells")

    print(
        f"Identifying objects: boundary_threshold={obs.boundary_threshold}, "
        f"max_value_threshold={obs.max_value_threshold}, area_threshold_km2={obs.area_threshold_km2}, "
        f"track={obs.track}, file_grouping={obs.file_grouping}"
    )

    return run_object_id_series(
        manifest, lambda entry: loader(entry.filepath),
        thresh_1=obs.boundary_threshold, thresh_2=obs.max_value_threshold, area_thresh_km2=obs.area_threshold_km2,
        output_dir=obs.object_output_dir,
        file_grouping=obs.file_grouping,
        track_in_time=obs.track,
        track_bound_disp_km=obs.track_distance_km,
        mask=mask,
        linear_eccentricity_thresh=linear.linear_eccentricity_threshold,
        linear_length_thresh_km=linear.linear_length_threshold_km,
        mixed_eccentricity_thresh=linear.mixed_eccentricity_threshold,
        mixed_length_thresh_km=linear.mixed_length_threshold_km,
    )


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
    out_paths = run_one_case(config_path)
    obs = require_section(load_config(config_path).observations, "observations", config_path)
    print(f"\nDone: wrote {len(out_paths)} object files to '{obs.object_output_dir}'")


if __name__ == "__main__":
    main()
