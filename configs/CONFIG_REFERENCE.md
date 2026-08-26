# Config reference

Every `python_obj/configs/*.yaml` file is parsed by `python_obj.config.load_config()`
into a `Config` object with several independently **optional** top-level
sections. Each section here functions like a **namelist** familiar from NWP
models (WRF, MPAS): one flat set of named parameters, no code changes needed
to adjust a run.

Populate only the section(s) the driver(s) you're running actually need — a
driver calls `require_section()` on exactly the section(s) it reads and
raises a clear, named error only if one of those is missing; an unpopulated
section irrelevant to that driver is never an error. See
`config_example_*.yaml` for one single-purpose example config per driver, or
`config.yaml` for a fully-populated, chained example spanning every section.

Path-shaped fields (any directory/file path) are resolved relative to the
**config file's own directory**, not the current working directory a driver
happens to be run from.

---

## `interpolation:` — read by `interpolate_mrms.py`

Interpolates raw native-grid MRMS composite reflectivity onto a fixed target
grid.

| Field | Type | Default | Description |
|---|---|---|---|
| `raw_mrms_dir` | str | *required* | Directory of raw native-grid MRMS GRIB2 files, `YYYYMMDD/` subdirectories. |
| `interp_mrms_dir` | str | *required* | Output directory for interpolated NetCDF files. |
| `target_grid_file` | str \| null | `null` | A ready-made model-grid NetCDF file to interpolate onto. Mutually exclusive with the corner+spacing+dims fields below — give exactly one target-grid mode. |
| `target_lat_name` | str | `"latitude"` | Latitude variable name inside `target_grid_file`. |
| `target_lon_name` | str | `"longitude"` | Longitude variable name inside `target_grid_file`. |
| `target_grid_sw_lat` | float \| null | `null` | Southwest-corner latitude for a computed grid (corner+spacing+dims mode). |
| `target_grid_sw_lon` | float \| null | `null` | Southwest-corner longitude for a computed grid. |
| `target_grid_dx_km` | float \| null | `null` | East-west grid spacing (km) for a computed grid. |
| `target_grid_dy_km` | float \| null | `target_grid_dx_km` | North-south grid spacing (km); defaults to square cells if omitted. |
| `target_grid_nx` | int \| null | `null` | Number of columns for a computed grid. |
| `target_grid_ny` | int \| null | `null` | Number of rows for a computed grid. |
| `target_grid_true_lat_1` | float \| null | `null` | Optional LCC projection override (auto-derived from grid extent if omitted). Computed-grid mode only. |
| `target_grid_true_lat_2` | float \| null | `null` | Same as above, second standard parallel. |
| `target_grid_cen_lat` | float \| null | `null` | Optional LCC projection center latitude override. |
| `target_grid_cen_lon` | float \| null | `null` | Optional LCC projection center longitude override. |
| `n_workers` | int | `8` | Number of parallel worker processes for file-level interpolation. |
| `weight_cache_dir` | str | `"output/weight_cache"` | Directory for cached regridding weights (keyed by a content hash of both grids). |
| `date_range` | [str, str] \| null | `null` | Inclusive `[YYYYMMDD, YYYYMMDD]` range restricting discovered files; `null` = everything found. |
| `max_files` | int \| null | `null` | Caps the file count after discovery/date filtering; `null` = no cap. |
| `file_pattern` | str | `"**/*.grib2*"` | Glob restricting discovery to one MRMS product when a date directory holds more than one (e.g. `"**/*MergedReflectivityQCComposite*"`). |

Exactly one target-grid mode is required: `target_grid_file`, **or** all five
of `target_grid_sw_lat`/`target_grid_sw_lon`/`target_grid_dx_km`/`target_grid_nx`/`target_grid_ny`.

---

## `observations:` — read by `identify_track_mrms.py`

Identifies (and optionally tracks) thunderstorm objects in already-
interpolated MRMS data. Paired with `linear_classification:` (see below) --
`identify_track_mrms.py` requires both sections together.

| Field | Type | Default | Description |
|---|---|---|---|
| `file_format` | str | *required* | `netcdf` \| `grib2` — selects the loader. |
| `var_name` | str | *required* | Reflectivity variable name to read. |
| `lat_name` | str | *required* | Latitude variable name. |
| `lon_name` | str | *required* | Longitude variable name. |
| `boundary_threshold` | float | *required* | Initial object-boundary identification threshold (dBZ). |
| `max_value_threshold` | float | *required* | Stricter threshold an object's peak value must exceed to be retained (dBZ). |
| `area_threshold_km2` | float | *required* | Minimum true physical area (km²) for an object to be retained. |
| `interp_mrms_dir` | str | *required* | Directory of already-interpolated MRMS files (typically `interpolate_mrms.py`'s own output). |
| `mask` | str | `"none"` | `none` \| `conus` \| `conus_east` — spatial domain restriction. |
| `track` | bool | `false` | Whether to track objects in time (assigns `track_id`/`branch_id`/`age_seconds`). |
| `track_distance_km` | float | `0.0` | Buffer distance for tracking linkage; `0.0` = touching/overlapping only. |
| `file_grouping` | str | `"single"` | `single` \| `member_series` \| `ensemble_snapshot` \| `full` \| `init_snapshot` — output file shape. |
| `object_output_dir` | str | `"output/obj_mrms"` | Output directory for object files. |
| `file_pattern` | str | `"**/*.nc"` | Glob restricting discovery when `interp_mrms_dir` contains non-data files alongside real ones. |

---

## `model:` — read by `identify_track_model.py`

Identifies (and optionally tracks) thunderstorm objects in already-gridded
model/forecast output — a single deterministic run or a multi-member
ensemble. Paired with `linear_classification:` (below) -- both sections
required together.

| Field | Type | Default | Description |
|---|---|---|---|
| `file_format` | str | *required* | `netcdf` \| `grib2` — selects the loader. |
| `var_name` | str | *required* | Reflectivity variable name to read. |
| `lat_name` | str | *required* | Latitude variable name. |
| `lon_name` | str | *required* | Longitude variable name. |
| `boundary_threshold` | float | *required* | Initial object-boundary identification threshold (dBZ). |
| `max_value_threshold` | float | *required* | Stricter threshold an object's peak value must exceed to be retained (dBZ). |
| `area_threshold_km2` | float | *required* | Minimum true physical area (km²) for an object to be retained. |
| `input_dir` | str | *required* | Directory of model output files. |
| `init_attr` | str \| null | `null` | Global attribute holding the forecast init time (arithmetic time-derivation mode). |
| `lead_attr` | str \| null | `null` | Global attribute holding the forecast lead-time number (arithmetic mode). |
| `init_format` | str \| null | `null` | `strptime` format for `init_attr`'s value (arithmetic mode). |
| `valid_time_attr` | str \| null | `null` | Global attribute holding a ready-made valid_time string (string time-derivation mode, e.g. WoFS). |
| `valid_time_format` | str \| null | `null` | `strptime` format for `valid_time_attr`'s value (string mode). |
| `valid_time_var` | str \| null | `null` | Name of a CF-convention time coordinate variable to decode instead — takes precedence over `valid_time_attr` when a source's global attribute is known-unreliable (real case: WoFSCast). |
| `mask` | str | `"none"` | `none` \| `conus` \| `conus_east` — spatial domain restriction. |
| `track` | bool | `false` | Whether to track objects in time. |
| `track_distance_km` | float | `0.0` | Buffer distance for tracking linkage; `0.0` = touching/overlapping only. |
| `file_grouping` | str | `"single"` | `single` \| `member_series` \| `ensemble_snapshot` \| `full` \| `init_snapshot` — output file shape. |
| `lead_units` | str | `"hours"` | `hours` \| `minutes` \| `seconds` — the unit `lead_attr`'s raw number is already in (arithmetic mode). |
| `member_subdirs` | bool | `false` | `true` = one ensemble member per immediate subdirectory of `input_dir`. |
| `member_subdir_pattern` | str | `"*"` | Restricts which subdirectories count as members when `member_subdirs=true` and `input_dir` has non-member siblings (e.g. `"mem[0-9]*"`). |
| `stacked_members` | bool | `false` | `true` = all ensemble members stacked inside each file as a real array dimension (e.g. WoFS `comp_dz(ne=18,...)`), not one file per member. |
| `file_pattern` | str | `"*.nc"` | Glob restricting file discovery. |
| `object_output_dir` | str | `"output/obj_model"` | Output directory for object files. |
| `init_time_attr` | str | `"init_time"` | Only used for `file_grouping="init_snapshot"` in string time-mode: the file's own init-time string attribute (read with `valid_time_format`). Unused in arithmetic mode (init_time comes from `init_attr` directly there). |

Exactly one time-derivation mode is required: `valid_time_attr`+`valid_time_format`,
**or** `init_attr`+`lead_attr`+`init_format`, **or** `valid_time_var`.

---

## `matching:` — read by `run_matching.py`, `run_matching_per_case.py`

Matches truth vs. forecast object files via a Total Interest (TI) score.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_boundary_disp_km` | float | *required* | Maximum boundary displacement (km) between candidate truth/forecast objects. |
| `max_centroid_disp_km` | float | *required* | Maximum centroid displacement (km) between candidate objects. |
| `ti_threshold` | float | *required* | Minimum spatial Total Interest score for a pair to be a match candidate. |
| `truth_object_dir` | str | *required* | Directory of truth (e.g. MRMS) object files. |
| `forecast_object_dir` | str | *required* | Directory of forecast (e.g. model) object files. |
| `max_time_offset_minutes` | float | `5.0` | Maximum allowed truth/forecast valid_time mismatch when pairing files. |
| `output_dir` | str | `"output/matches"` | Output directory for match files. |
| `file_pattern` | str | `"*.nc"` | Glob restricting object-file discovery in both directories. |

---

## `linear_classification:` — read by `identify_track_mrms.py`, `identify_track_model.py`

Shared, not per-source: object shape/linearity is a property of the storm's
own footprint, not a per-source dBZ calibration value. Two independent
tiers, checked strict-first: an object meeting the `linear_*` thresholds is
"linear" (`is_linear=2`); failing that but meeting the `mixed_*` thresholds
is "mixed" (`is_linear=1`); meeting neither is "cellular" (`is_linear=0`).

| Field | Type | Default | Description |
|---|---|---|---|
| `linear_eccentricity_threshold` | float | *required* | Minimum eccentricity for the "linear" tier. |
| `linear_length_threshold_km` | float | *required* | Minimum major-axis length (km) for the "linear" tier. |
| `mixed_eccentricity_threshold` | float | *required* | Minimum eccentricity for the "mixed" tier. |
| `mixed_length_threshold_km` | float | *required* | Minimum major-axis length (km) for the "mixed" tier. |
| `storm_mode_classification` | bool | `false` | **v2.** When `true`, objects sharing one connected region at `system_boundary_threshold` (a third, lower dBZ threshold) are merged into a "system" before classification: linear/mixed/cellular is decided from the UNION of the constituent objects' own already-identified pixel footprints (not the full system-boundary connected component), then applied to every constituent object, which also gets a shared `system_id`. Fixes spurious loss of temporal continuity when a linear system's leading edge fragments into several individually-too-small/round objects. Default `false` reproduces v1 behavior byte-for-byte. |
| `system_boundary_threshold` | float \| null | `null` | **v2.** Required whenever `storm_mode_classification` is `true`; must be a LOWER dBZ value than `boundary_threshold` (in `observations:`/`model:`). |

---

## `fetch_mrms:` — read by `fetch_mrms.py`

Fetches MRMS observations from the public `noaa-mrms-pds` AWS S3 archive.
Two top-level modes, mutually exclusive: **model-driven** (fetch the nearest
MRMS match to each file in a directory of model output) or **date-driven**
(fetch everything found for given date(s), no model files involved).

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | str | *required* | Output directory for fetched MRMS files. |
| `model_input_dir` | str \| null | `null` | Model-driven mode: directory of model files whose valid times to match. Mutually exclusive with `dates`/`date_range`. |
| `file_pattern` | str | `"*.nc"` | Glob restricting model-file discovery (model-driven mode only). |
| `valid_time_attr` | str \| null | `null` | Model-driven mode, string time-derivation: global attribute holding a ready-made valid_time string. |
| `valid_time_format` | str \| null | `null` | `strptime` format for `valid_time_attr`'s value. |
| `init_attr` | str \| null | `null` | Model-driven mode, arithmetic time-derivation: global attribute holding the init time. |
| `lead_attr` | str \| null | `null` | Arithmetic mode: global attribute holding the lead-time number. |
| `lead_units` | str | `"hours"` | `hours` \| `minutes` \| `seconds` — unit of `lead_attr`'s raw number (arithmetic mode). |
| `init_format` | str \| null | `null` | `strptime` format for `init_attr`'s value (arithmetic mode). |
| `tolerance_minutes` | float | `5.0` | Maximum allowed gap between a model file's valid_time and the nearest available MRMS timestamp. |
| `dates` | list[str] \| null | `null` | Date-driven mode: explicit list of `YYYYMMDD` strings. Mutually exclusive with `date_range` and `model_input_dir`. |
| `date_range` | [str, str] \| null | `null` | Date-driven mode: inclusive `[YYYYMMDD, YYYYMMDD]` range, expanded to daily strings. |
| `s3_bucket` | str | `"noaa-mrms-pds"` | S3 bucket name. |
| `mrms_product` | str | `"MergedReflectivityQCComposite_00.50"` | MRMS product path within the bucket. |
| `mirror_subdirs` | bool | `true` | Mirror the source's `YYYYMMDD/` subdirectory structure in `output_dir`. |
| `skip_existing` | bool | `true` | Skip re-downloading a file already present locally with a matching size. |
| `max_files` | int \| null | `null` | Model-driven: caps model files processed. Date-driven: caps TOTAL files across every requested date combined (not a per-day cap). |

Model-driven mode additionally requires exactly one time-derivation mode
(same rule as `model:` above): `valid_time_attr`+`valid_time_format`, or
`init_attr`+`lead_attr`+`init_format`.

---

## `histogram_observations:` — read by `build_histogram_mrms.py`

Builds one reflectivity-value-distribution histogram file per `YYYYMMDD` day
of already-interpolated MRMS. Self-contained (no thresholds/tracking fields
— those are object-ID concepts, irrelevant to a raw-value distribution).

| Field | Type | Default | Description |
|---|---|---|---|
| `interp_mrms_dir` | str | *required* | Parent directory of `YYYYMMDD/` day-subdirectories of already-interpolated MRMS. |
| `var_name` | str | `"refl_consv"` | Reflectivity variable name. |
| `lat_name` | str | `"lat"` | Latitude variable name. |
| `lon_name` | str | `"lon"` | Longitude variable name. |
| `output_dir` | str | `"output/hist_mrms"` | Output directory for histogram files. |
| `mask` | str | `"none"` | `none` \| `conus` \| `conus_east` — spatial domain restriction. Masked cells are excluded (set to NaN before histogramming), never zeroed. |
| `bin_min` | float | `-20.0` | Lower edge of the histogram bin range (dBZ). |
| `bin_max` | float | `80.0` | Upper edge of the histogram bin range (dBZ). |
| `bin_width` | float | `0.2` | Bin width (dBZ). |
| `edge_trim` | int | `7` | Pixels trimmed from each grid edge before histogramming. |
| `clip_negative_to_zero` | bool | `false` | If `true`, negative values are floored to 0 before histogramming (off by default — real values outside `[bin_min, bin_max]` are clamped to the nearest edge bin, never dropped, so every valid pixel is guaranteed to land in some bin). |
| `file_pattern` | str | `"**/*.nc"` | Glob restricting discovery when `interp_mrms_dir` contains non-data files alongside real ones. |

---

## `histogram_model:` — read by `build_histogram_model.py`

Builds one reflectivity-value-distribution histogram file for one whole
forecast (every lead time, every member if ensemble). Self-contained, same
rationale as `histogram_observations:` above.

| Field | Type | Default | Description |
|---|---|---|---|
| `input_dir` | str | *required* | Directory of model output files. |
| `var_name` | str | `"refl10cm_max"` | Reflectivity variable name. |
| `lat_name` | str | `"latitude"` | Latitude variable name. |
| `lon_name` | str | `"longitude"` | Longitude variable name. |
| `member_subdirs` | bool | `false` | `true` = one ensemble member per immediate subdirectory of `input_dir`. |
| `member_subdir_pattern` | str | `"*"` | Restricts which subdirectories count as members (see `model:`'s field of the same name). |
| `stacked_members` | bool | `false` | `true` = all members stacked inside each file as a real array dimension. |
| `file_pattern` | str | `"*.nc"` | Glob restricting file discovery. |
| `init_attr` | str \| null | `null` | Arithmetic time-derivation mode: global attribute holding the init time. |
| `lead_attr` | str \| null | `null` | Arithmetic mode: global attribute holding the lead-time number. |
| `lead_units` | str | `"hours"` | `hours` \| `minutes` \| `seconds` — unit of `lead_attr`'s raw number. |
| `init_format` | str \| null | `null` | `strptime` format for `init_attr`'s value. |
| `valid_time_attr` | str \| null | `null` | String time-derivation mode: global attribute holding a ready-made valid_time string. |
| `valid_time_format` | str \| null | `null` | `strptime` format for `valid_time_attr`'s value. |
| `valid_time_var` | str \| null | `null` | CF-convention time coordinate variable override (see `model:`'s field of the same name). |
| `init_time_attr` | str | `"init_time"` | Only used to derive `lead_hours` in string time-mode: the file's own init-time string attribute, read with `valid_time_format`. Unused in arithmetic mode. |
| `output_dir` | str | `"output/hist_model"` | Output directory for histogram files. |
| `mask` | str | `"none"` | `none` \| `conus` \| `conus_east` — spatial domain restriction (excluded, not zeroed). |
| `bin_min` | float | `-20.0` | Lower edge of the histogram bin range (dBZ). |
| `bin_max` | float | `80.0` | Upper edge of the histogram bin range (dBZ). |
| `bin_width` | float | `0.2` | Bin width (dBZ). |
| `edge_trim` | int | `7` | Pixels trimmed from each grid edge before histogramming. |
| `clip_negative_to_zero` | bool | `false` | See `histogram_observations:`'s field of the same name. |

Exactly one time-derivation mode is required: `valid_time_attr`+`valid_time_format`,
**or** `init_attr`+`lead_attr`+`init_format`, **or** `valid_time_var`.

---

## Batch expansion: `cases:` — read by `python_obj.batch_config.expand_batch_config()`

Not a `Config` section itself — a separate top-level block recognized only
by the template-expansion mechanism (`expand_batch_config()`), which
materializes one ordinary config file per case from a template plus this
block, substituting `{date}`/`{init_time}` placeholders elsewhere in the
template. See `config_batch_template.yaml` and
`config_batch_template_init_times.yaml` for worked examples.

| Field | Type | Default | Description |
|---|---|---|---|
| `dates` | list[str] \| null | `null` | Explicit list of `YYYYMMDD` strings. Mutually exclusive with `date_range`. |
| `date_range` | [str, str] \| null | `null` | Inclusive `[YYYYMMDD, YYYYMMDD]` range. Mutually exclusive with `dates`. |
| `date_format` | str | `"%Y%m%d"` | Format used when substituting `{date}` into path fields. |
| `init_times` | list[str] \| null | `null` | Explicit list of `HHMM`-style init-time strings — a second, optional expansion axis for archives nesting multiple forecast initializations under one date directory. Mutually exclusive with `init_time_range`. |
| `init_time_range` | [str, str] \| null | `null` | `[start, end]` `HHMM` range, expanded contiguously by `init_time_step_minutes`. Mutually exclusive with `init_times`. |
| `init_time_step_minutes` | int | *required with `init_time_range`* | Step size (minutes) for `init_time_range`'s generator. |
| `init_time_format` | str | `"%H%M"` | Format used when substituting `{init_time}` into path fields. |
