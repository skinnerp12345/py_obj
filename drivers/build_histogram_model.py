"""Standalone script: build one composite-reflectivity (or any configured
variable) distribution histogram for one whole model/forecast run (every
lead time, every member if an ensemble). Generalizes
python_base/wofs_dz_histogram_base.py / wofs_dz_histogram_wofscast.py --
configurable bins/variable, and (new) preserves one histogram slice per
(member, lead-time) combination -- tagged with its real valid_time,
lead_hours, and member_id -- inside the one output file, rather than
collapsing everything to one flat total. This is what lets
python_obj.histogram.aggregate later build a "day N of the forecast" subset
from a single multi-day run, which the original script's output could never
support.

A thin driver over python_obj.histogram; does not modify anything else in
python_obj/. Configured entirely via the shared python_obj/configs/config.yaml
(its 'histogram_model:' section). File discovery reuses
python_obj.obj_core.build_model_manifest -- the same manifest builder
identify_track_model.py uses.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/build_histogram_model.py [path/to/config.yaml]

If no config path is given, uses python_obj/configs/config.yaml.
"""

import os
import sys
from datetime import datetime

import netCDF4
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.config import HistogramModelConfig, load_config, require_section
from python_obj.histogram import HistogramSlice, compute_histogram, default_bin_edges, write_histogram_file
from python_obj.obj_core import build_model_manifest, conus_mask, conus_mask_east

_LEAD_UNITS_TO_HOURS = {"hours": 1.0, "minutes": 1.0 / 60.0, "seconds": 1.0 / 3600.0}


def _compute_lead_hours(filepath: str, hist_cfg: HistogramModelConfig) -> float | None:
    """lead_hours for one file, using whichever time-derivation mode the
    config is set up for. init_attr/lead_attr mode reads the lead-time
    number directly (no reconstruction needed); valid_time_attr mode reads
    both the init_time-equivalent and valid_time string attributes (same
    format) and takes their difference. Returns None (not an error) if
    neither is derivable for this file -- the slice is still valid, it just
    won't match any by_lead_hours_range() predicate later."""
    with netCDF4.Dataset(filepath, "r") as ds:
        if hist_cfg.lead_attr is not None and hasattr(ds, hist_cfg.lead_attr):
            lead_value = float(getattr(ds, hist_cfg.lead_attr))
            return lead_value * _LEAD_UNITS_TO_HOURS[hist_cfg.lead_units]

        if hist_cfg.valid_time_attr is not None and hasattr(ds, hist_cfg.valid_time_attr) and hasattr(ds, hist_cfg.init_time_attr):
            init_dt = datetime.strptime(getattr(ds, hist_cfg.init_time_attr), hist_cfg.valid_time_format)
            valid_dt = datetime.strptime(getattr(ds, hist_cfg.valid_time_attr), hist_cfg.valid_time_format)
            return (valid_dt - init_dt).total_seconds() / 3600.0

    return None


def run_one_case(config_path: str) -> str:
    cfg = load_config(config_path)
    hist_cfg: HistogramModelConfig = require_section(cfg.histogram_model, "histogram_model", config_path)

    bins = default_bin_edges(hist_cfg.bin_min, hist_cfg.bin_max, hist_cfg.bin_width)
    manifest, loader = build_model_manifest(
        input_dir=hist_cfg.input_dir, file_pattern=hist_cfg.file_pattern,
        member_subdirs=hist_cfg.member_subdirs, stacked_members=hist_cfg.stacked_members,
        var_name=hist_cfg.var_name, lat_name=hist_cfg.lat_name, lon_name=hist_cfg.lon_name,
        init_attr=hist_cfg.init_attr, lead_attr=hist_cfg.lead_attr, lead_units=hist_cfg.lead_units,
        init_format=hist_cfg.init_format,
        valid_time_attr=hist_cfg.valid_time_attr, valid_time_format=hist_cfg.valid_time_format,
        member_subdir_pattern=hist_cfg.member_subdir_pattern,
    )
    print(
        f"Found {len(manifest)} manifest entries under '{hist_cfg.input_dir}' "
        f"(member_subdirs={hist_cfg.member_subdirs}, stacked_members={hist_cfg.stacked_members})"
    )
    print(
        f"Building histogram: var_name={hist_cfg.var_name}, bins=[{hist_cfg.bin_min}, {hist_cfg.bin_max}] "
        f"by {hist_cfg.bin_width}, edge_trim={hist_cfg.edge_trim}, clip_negative_to_zero={hist_cfg.clip_negative_to_zero}"
    )

    # mask is computed once from the first manifest entry's own lat/lon (a
    # fixed grid across the whole series, same assumption identify_track_model.py
    # makes) and applied as NaN, NOT the 0.0 the object-ID pipeline uses -- a
    # masked cell must be excluded from the distribution entirely, not
    # counted as a fake clear-air reading (compute_histogram already excludes
    # NaN via its existing np.isfinite filter, so no core histogram code
    # changes needed).
    mask = None
    if hist_cfg.mask != "none":
        first = loader(manifest[0].filepath, extra_dim_index=manifest[0].extra_dim_index)
        mask_fn = conus_mask_east if hist_cfg.mask == "conus_east" else conus_mask
        mask = mask_fn(first.lat2d, first.lon2d)
        print(f"Mask '{hist_cfg.mask}' excludes {mask.mean() * 100:.1f}% of grid cells")

    slices = []
    source_files = sorted({entry.filepath for entry in manifest})
    lead_hours_by_file: dict[str, float | None] = {
        f: _compute_lead_hours(f, hist_cfg) for f in source_files
    }
    for entry in manifest:
        field = loader(entry.filepath, extra_dim_index=entry.extra_dim_index)
        data = field.data if mask is None else np.where(mask, np.nan, field.data)
        counts = compute_histogram(
            data, bins, edge_trim=hist_cfg.edge_trim, clip_negative_to_zero=hist_cfg.clip_negative_to_zero,
            missing_value=field.missing_value,
        )
        slices.append(HistogramSlice(
            valid_time=entry.valid_time, hist=counts,
            lead_hours=lead_hours_by_file[entry.filepath], member_id=entry.member_id,
        ))

    os.makedirs(hist_cfg.output_dir, exist_ok=True)
    anchor_time = min(s.valid_time for s in slices)
    out_path = os.path.join(hist_cfg.output_dir, f"hist_model_{anchor_time:%Y%m%d_%H%M%S}.nc")
    write_histogram_file(
        out_path, bins, slices, hist_cfg.var_name, hist_cfg.edge_trim, hist_cfg.clip_negative_to_zero, source_files,
    )
    print(f"  wrote {out_path} ({len(slices)} slices, {sum(s.hist.sum() for s in slices)} total counts)")

    return out_path


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
    out_path = run_one_case(config_path)
    print(f"\nDone: wrote 1 histogram file for this forecast -> '{out_path}'")


if __name__ == "__main__":
    main()
