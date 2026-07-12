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

import glob
import os
import sys
from typing import Callable

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.config import ModelConfig, load_config, require_section
from python_obj.obj_core import SeriesEntry, conus_mask, conus_mask_east, run_object_id_series
from python_obj.regrid import infer_stacked_member_count, load_model_netcdf, read_valid_time_only


def _make_loader(model: ModelConfig) -> Callable[..., object]:
    return lambda fp, extra_dim_index=None: load_model_netcdf(
        fp,
        varname=model.var_name,
        lat_name=model.lat_name,
        lon_name=model.lon_name,
        init_attr=model.init_attr,
        lead_attr=model.lead_attr,
        lead_units=model.lead_units,
        init_format=model.init_format,
        valid_time_attr=model.valid_time_attr,
        valid_time_format=model.valid_time_format,
        extra_dim_index=extra_dim_index,
    )


def build_manifest(model: ModelConfig) -> tuple[list[SeriesEntry], Callable[..., object]]:
    """Build the (member, time, filepath) manifest run_object_id_series needs.

    member_subdirs=True: one member per immediate subdirectory of input_dir
    (member_id = subdirectory basename), mirroring the real test_mpas/mem1/,
    test_mpas/mem2/ layout already used elsewhere in this repo -- not parsed
    from any filename convention, since a model's own ensemble-naming scheme
    is the caller's business, not this pipeline's.

    stacked_members=True: each file contains ALL members stacked as a real
    array dimension (e.g. WoFS's comp_dz(ne=18, lat, lon), one file per
    valid_time rather than one file per member) -- one flat glob over
    input_dir, then infer_stacked_member_count() once per file (a cheap
    metadata-only shape read, never loads the full array) to discover how
    many members it holds, producing one SeriesEntry per (file, member index)
    with a distinct member_id and extra_dim_index.

    member_subdirs=False and stacked_members=False: one flat glob over
    input_dir, member_id=None (a single deterministic run).

    valid_time comes from load_model_netcdf()'s (or, for stacked_members,
    read_valid_time_only()'s) own flexible derivation -- never parsed from
    the filename.
    """
    loader = _make_loader(model)
    manifest: list[SeriesEntry] = []

    if model.member_subdirs:
        member_dirs = sorted(
            d for d in glob.glob(os.path.join(model.input_dir, "*")) if os.path.isdir(d)
        )
        if not member_dirs:
            raise FileNotFoundError(
                f"member_subdirs=True but no subdirectories found under '{model.input_dir}'"
            )
        for member_dir in member_dirs:
            member_id = os.path.basename(os.path.normpath(member_dir))
            files = sorted(glob.glob(os.path.join(member_dir, model.file_pattern)))
            if not files:
                raise FileNotFoundError(
                    f"No files matching '{model.file_pattern}' found under member directory '{member_dir}'"
                )
            for f in files:
                manifest.append(SeriesEntry(valid_time=loader(f).valid_time, filepath=f, member_id=member_id))
    elif model.stacked_members:
        files = sorted(glob.glob(os.path.join(model.input_dir, model.file_pattern)))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{model.file_pattern}' found under '{model.input_dir}'"
            )
        for f in files:
            valid_time = read_valid_time_only(
                f,
                init_attr=model.init_attr, lead_attr=model.lead_attr, lead_units=model.lead_units,
                init_format=model.init_format,
                valid_time_attr=model.valid_time_attr, valid_time_format=model.valid_time_format,
            )
            n_members = infer_stacked_member_count(f, model.var_name)
            for idx in range(n_members):
                manifest.append(SeriesEntry(
                    valid_time=valid_time, filepath=f,
                    member_id=f"mem{idx:02d}", extra_dim_index=idx,
                ))
    else:
        files = sorted(glob.glob(os.path.join(model.input_dir, model.file_pattern)))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{model.file_pattern}' found under '{model.input_dir}'"
            )
        for f in files:
            manifest.append(SeriesEntry(valid_time=loader(f).valid_time, filepath=f, member_id=None))

    return manifest, loader


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
