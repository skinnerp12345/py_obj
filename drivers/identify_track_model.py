"""Standalone script: identify (and optionally track) thunderstorm objects in
already-gridded model/forecast output -- a single deterministic run or a
multi-member ensemble.

No interpolation step here (model output is assumed already on its own
target grid); no matching against truth (see run_matching.py for that). A
thin driver over python_obj.obj_core.run_object_id_series; does not modify
anything else in python_obj/. Configured entirely via the shared
python_obj/configs/config.yaml (its 'model:' and 'linear_classification:' sections).

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/identify_track_model.py [path/to/config.yaml]

If no config path is given, uses python_obj/configs/config.yaml.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.config import ModelConfig, load_config, require_section
from python_obj.obj_core import build_model_manifest, conus_mask, conus_mask_east, run_object_id_series


def build_manifest(model: ModelConfig):
    """Thin adapter over the shared python_obj.obj_core.build_model_manifest
    (promoted there so the histogram-building driver can reuse the same file-
    discovery logic without duplicating it)."""
    return build_model_manifest(
        input_dir=model.input_dir, file_pattern=model.file_pattern,
        member_subdirs=model.member_subdirs, stacked_members=model.stacked_members,
        var_name=model.var_name, lat_name=model.lat_name, lon_name=model.lon_name,
        init_attr=model.init_attr, lead_attr=model.lead_attr, lead_units=model.lead_units,
        init_format=model.init_format,
        valid_time_attr=model.valid_time_attr, valid_time_format=model.valid_time_format,
        member_subdir_pattern=model.member_subdir_pattern,
    )


def run_one_case(config_path: str) -> list[str]:
    cfg = load_config(config_path)
    model = require_section(cfg.model, "model", config_path)
    linear = require_section(cfg.linear_classification, "linear_classification", config_path)

    manifest, loader = build_manifest(model)

    print(
        f"Found {len(manifest)} manifest entries under '{model.input_dir}' "
        f"(member_subdirs={model.member_subdirs}, stacked_members={model.stacked_members})"
    )

    mask = None
    if model.mask != "none":
        first = loader(manifest[0].filepath, extra_dim_index=manifest[0].extra_dim_index)
        mask_fn = conus_mask_east if model.mask == "conus_east" else conus_mask
        mask = mask_fn(first.lat2d, first.lon2d)
        print(f"Mask '{model.mask}' excludes {mask.mean() * 100:.1f}% of grid cells")

    print(
        f"Identifying objects: boundary_threshold={model.boundary_threshold}, "
        f"max_value_threshold={model.max_value_threshold}, area_threshold_km2={model.area_threshold_km2}, "
        f"track={model.track}, file_grouping={model.file_grouping}"
    )

    return run_object_id_series(
        manifest, lambda entry: loader(entry.filepath, extra_dim_index=entry.extra_dim_index),
        thresh_1=model.boundary_threshold, thresh_2=model.max_value_threshold,
        area_thresh_km2=model.area_threshold_km2,
        output_dir=model.object_output_dir,
        file_grouping=model.file_grouping,
        track_in_time=model.track,
        track_bound_disp_km=model.track_distance_km,
        mask=mask,
        linear_eccentricity_thresh=linear.linear_eccentricity_threshold,
        linear_length_thresh_km=linear.linear_length_threshold_km,
        mixed_eccentricity_thresh=linear.mixed_eccentricity_threshold,
        mixed_length_thresh_km=linear.mixed_length_threshold_km,
    )


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
    out_paths = run_one_case(config_path)
    model = require_section(load_config(config_path).model, "model", config_path)
    print(f"\nDone: wrote {len(out_paths)} object files to '{model.object_output_dir}'")


if __name__ == "__main__":
    main()
