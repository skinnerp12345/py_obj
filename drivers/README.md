# python_obj/drivers/

Standalone, independently-runnable driver scripts built on top of the rest
of `python_obj/` (nothing here modifies `regrid/`, `obj_core/`, or
`config.py`). Each driver reads only the section(s) of the **one shared
config file** it actually needs — a user populates `python_obj/configs/config.yaml`
with whichever sections are relevant to their problem, then chooses which
driver(s) to run. Omitting a section a given driver doesn't use is never an
error; using a driver whose required section is missing raises a clear,
named error telling you which section to add.

The `pysteps_env` conda environment is required for all of these:
`/opt/anaconda3/envs/pysteps_env/bin/python`.

## The config file

`python_obj/configs/config.yaml` has independently optional top-level sections:

| Section | Used by | Required fields |
|---|---|---|
| `interpolation:` | `interpolate_mrms.py` | `raw_mrms_dir`, `interp_mrms_dir`, `target_grid_file` |
| `observations:` | `identify_track_mrms.py` | `file_format`, `var_name`, `lat_name`, `lon_name`, `boundary_threshold`, `max_value_threshold`, `area_threshold_km2`, `interp_mrms_dir` |
| `model:` | `identify_track_model.py` | same common fields as `observations:` plus `init_attr`, `lead_attr`, `init_format`, `input_dir` |
| `matching:` | `run_matching.py` | `max_boundary_disp_km`, `max_centroid_disp_km`, `ti_threshold`, `truth_object_dir`, `forecast_object_dir` |
| `linear_classification:` | `identify_track_mrms.py`, `identify_track_model.py` | all four threshold fields |
| `fetch_mrms:` | `fetch_mrms.py` | `model_input_dir`, `output_dir`, and either `valid_time_attr`+`valid_time_format` or `init_attr`+`lead_attr`+`init_format` |
| `histogram_observations:` | `build_histogram_mrms.py`, `aggregate_histograms.py` | `interp_mrms_dir` |
| `histogram_model:` | `build_histogram_model.py`, `aggregate_histograms.py` | `input_dir`, and either `valid_time_attr`+`valid_time_format` or `init_attr`+`lead_attr`+`init_format` |

Every other field has a documented default — see the comments in
`python_obj/configs/config.yaml` for the full field list. Paths are resolved relative
to **the config file's own directory**, not your current directory, so the
same file behaves identically no matter where you run a driver from.

`python_obj/configs/config.yaml` as shipped populates all five sections and chains
together end to end (interpolation's output feeds observations' input;
observations'/model's outputs feed matching's inputs) — a complete,
self-consistent walkthrough if you run all four drivers against it in
sequence. `python_obj/configs/config_smoketest.yaml` (interpolation+observations only,
4 files) and `python_obj/configs/config_ensemble.yaml` (model+linear_classification
only, 2-member MPAS ensemble) are smaller examples that populate only the
sections their own scenario needs — a concrete demonstration that omitting
irrelevant sections is normal, not an error.

## `interpolate_mrms.py`

Interpolates raw native-grid MRMS composite reflectivity onto a fixed target
grid. Requires `interpolation:`.

```bash
/opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/interpolate_mrms.py [path/to/config.yaml]
```

Writes one NetCDF file per input MRMS file under `interpolation.interp_mrms_dir`.
The first run for a given (MRMS grid, target grid) pair builds and caches
ESMF regridding weights under `weight_cache_dir`; later runs reuse the cache.
Warnings like "N of M target cells had no valid source coverage" are
expected wherever your target grid extends beyond MRMS's observed radar
coverage — those cells are padded with the fill value, not an error.

## `identify_track_mrms.py`

Identifies (and optionally tracks) storm objects in already-interpolated
MRMS data. Requires `observations:` + `linear_classification:` (needs
`interpolate_mrms.py` to have already populated `observations.interp_mrms_dir`).

```bash
/opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/identify_track_mrms.py [path/to/config.yaml]
```

Writes object files under `observations.object_output_dir`, shaped per
`observations.file_grouping` (defaults to `single` — one file per output
time, the natural shape for observations, which have no member/ensemble
concept). `observations.track: true` links objects across consecutive times
(`age_seconds`/`track_id`).

## `identify_track_model.py`

Identifies (and optionally tracks) storm objects in already-gridded
model/forecast output — a single deterministic run or a multi-member
ensemble. No interpolation (model output is assumed already on its own
target grid) and no truth-vs-forecast matching. Requires `model:` +
`linear_classification:`.

```bash
/opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/identify_track_model.py [path/to/config.yaml]
```

`model.member_subdirs: false` (default) treats `input_dir` as one flat
series. Set `true` for an ensemble: `input_dir` must then contain one
immediate subdirectory per member (e.g. `mem1/`, `mem2/`, matching the real
`test_mpas/mem1/`, `test_mpas/mem2/` convention) — the subdirectory name
becomes each member's `member_id`, nothing is parsed from filenames.
`model.member_subdir_pattern` (default `"*"`, every subdirectory) restricts
*which* subdirectories count as members when `member_subdirs: true` and
`input_dir` has non-member siblings — confirmed real on an NCAR HPC MPAS
archive with `mem1`-`mem10` alongside an unrelated `ens_mean_5mems`
directory at the same level: the default discovers every subdirectory, so
the first non-member one with no matching files raises `FileNotFoundError`.
Set e.g. `member_subdir_pattern: "mem[0-9]*"` to restrict discovery to only
directories whose name starts with `mem` followed by a digit.
`model.lead_units` is the unit `lead_attr`'s raw stored number is already
in, NOT a description of your model's output cadence (MPAS's hourly output
happens to store `forecastHour` in hours; a 5-minute-cadence model isn't
required to use `minutes`). `model.file_grouping`: `single` (one file per
time), `member_series` (one file per member), `ensemble_snapshot` (one file
per time, all members), or `full` (one file, everything).

## `run_matching.py`

Matches two already-existing directories of object files (a truth series, a
forecast series — from any source) into `hit`/`miss`/`false_alarm`/
`truth_extra`/`forecast_extra` categories via the Total Interest score. No
identification/tracking here. Requires `matching:`.

```bash
/opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/run_matching.py [path/to/config.yaml]
```

`matching.max_time_offset_minutes` is the tolerance for aligning each
forecast valid_time to the nearest truth valid_time (still one truth time
per forecast time, not a fuzzy multi-way match) — a forecast time with no
truth time within tolerance is skipped and reported, never silently
dropped. Writes one match file per distinct forecast valid_time under
`matching.output_dir`.

## `fetch_mrms.py`

Fetches MRMS observations matching a directory of model files' valid times
from the public `noaa-mrms-pds` AWS S3 archive (no credentials needed, plain
HTTPS) — for a model (e.g. WoFS) that has no local matching MRMS data yet.
Requires `fetch_mrms:`.

```bash
/opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/fetch_mrms.py [path/to/config.yaml]
```

For each model file, derives its valid_time (via
`python_obj.regrid.read_valid_time_only`'s flexible mechanism — either a
ready-made `valid_time_attr`+`valid_time_format` string, e.g. WoFS's
`valid_time="20260518_230000"`, or `init_attr`+`lead_attr`+`init_format`
arithmetic, e.g. MPAS's), lists that day's MRMS archive (one HTTPS request
per distinct day, cached across files), and downloads the nearest MRMS file
within `tolerance_minutes`. A model time with no MRMS file within tolerance
is reported as `no_match_within_tolerance`, never silently dropped.
`skip_existing: true` (default) skips re-downloading a file whose local size
already matches the archive's — safe to re-run.

Fetched files land at `<output_dir>/<YYYYMMDD>/<original filename>`, the
same layout `test_mrms/` and `discover_mrms_files()`/`interpolate_mrms.py`
already use — point `interpolation.raw_mrms_dir` at the fetched
`output_dir` to interpolate them with zero further changes.

## `build_histogram_mrms.py`

Builds a composite-reflectivity (or any configured variable) distribution
histogram for **each day** of already-interpolated MRMS -- generalizes
`python_base/mrms_dz_histogram_base.py` (configurable bins/variable instead
of hardcoded). Requires `histogram_observations:`.

```bash
/opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/build_histogram_mrms.py [path/to/config.yaml]
```

Groups `histogram_observations.interp_mrms_dir` by its `YYYYMMDD`
day-subdirectories and writes **one histogram file per day** under
`histogram_observations.output_dir`. Each file preserves one histogram
*slice* per input file (tagged with that file's real `valid_time`) rather
than collapsing straight to one flat total -- this is what lets
`aggregate_histograms.py` subset by hour of day afterward. Default bins:
-20 to 80 dBZ by 0.2; default `edge_trim=7` pixels off each border (avoids
regridding-edge artifacts, ported from the original script's hardcoded
`edge=7`); `clip_negative_to_zero` is off by default (the original WoFS-side
scripts did this silently -- here it's an explicit opt-in).

Bins are a fixed range with no clear-air/no-echo bin excluded: every valid
pixel is counted, including clear air, and any real value outside
`[bin_min, bin_max]` is clamped into the nearest edge bin rather than
dropped -- so two histograms built from the same grid/`edge_trim` always
carry equal total gridpoint counts, regardless of where a given source's
own clear-air floor happens to sit (confirmed to vary by source: ~0.0 dBZ
for MRMS, exactly -35.0 dBZ for MPAS). MRMS's `-999` "no coverage" sentinel
is still excluded outright (never clamped in as fake data) via the file's
own `_FillValue`, which this driver passes through automatically.

## `build_histogram_model.py`

Builds one distribution histogram for **one whole forecast** (every lead
time, every member if an ensemble) -- generalizes
`python_base/wofs_dz_histogram_base.py`/`wofs_dz_histogram_wofscast.py`.
Requires `histogram_model:`. File discovery reuses the same
`python_obj.obj_core.build_model_manifest` `identify_track_model.py` itself
uses (same `member_subdirs`/`stacked_members`/`file_pattern` semantics).

```bash
/opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/build_histogram_model.py [path/to/config.yaml]
```

Writes **one histogram file for the whole forecast** under
`histogram_model.output_dir`, with one slice per (member, lead-time)
combination -- tagged with real `valid_time`, `lead_hours`, and `member_id`.
`lead_hours` is read directly from `lead_attr` (already a lead-time number)
in the `init_attr`/`lead_attr` time-mode, or computed from the difference
between `init_time_attr` and `valid_time_attr` (same string format) in the
`valid_time_attr` time-mode.

## `aggregate_histograms.py`

Demonstrates subsetting the output of the two drivers above -- an
hour-of-day MRMS climatology, a forecast-lead-hour-bucketed ("day N of the
forecast") model subset -- then a real matched-percentile-threshold
computation between the two full distributions: generalizes
`python_base/wofs_dz_histogram_plotter.py`'s threshold-matching method
(find a source value's percentile, find the target distribution's value at
that same percentile) into a reusable function
(`python_obj.histogram.match_percentile_threshold`). Requires
`histogram_observations:` + `histogram_model:` (reads their own
`output_dir`s to find the histogram files to aggregate).

```bash
/opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/aggregate_histograms.py [path/to/config.yaml] [source_threshold_dbz]
```

For programmatic/notebook use, `python_obj.histogram.sum_histograms(paths,
predicate=...)` with `by_hour_of_day(hours)`/`by_lead_hours_range(min, max)`
is the reusable building block this driver is a thin wrapper over -- see
`notebooks/histogram_tutorial.ipynb`.

## Running many cases in parallel

Each driver has a `_batch.py` companion (`interpolate_mrms_batch.py`,
`identify_track_mrms_batch.py`, `identify_track_model_batch.py`,
`run_matching_batch.py`, `fetch_mrms_batch.py`) that runs its sibling's
`run_one_case(config_path)`
across a list of per-case config files in parallel, via
`python_obj.batch_runner.run_cases_in_parallel`. Edit the `CASE_CONFIGS`
list at the top of the script to point at your own config paths — cases are
never auto-discovered:

```bash
/opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/identify_track_model_batch.py
```

Prints a `BatchCaseSummary` (succeeded/failed count, each failing case's own
error) — one bad case never stops the others.

## Reading the output

```python
from python_obj.regrid import load_mrms_netcdf
from python_obj.obj_core import read_object_file, read_match_file

field = load_mrms_netcdf("python_obj/configs/output/interp_mrms/20230401/interp_mrms_20230401_010041.nc")
# field.lat2d, field.lon2d, field.data, field.valid_time

objs = read_object_file("python_obj/configs/output/obj_mrms/obj_obs_20230401_010041.nc")
# objs.objects -> list[StormObject] (centroid, area_km2, max_value, is_linear, track_id, ...)
# objs.member_ids, objs.member_index -> filter objects down to one member (ensemble files)

matches = read_match_file("python_obj/configs/output/matches/match_20230501_030000.nc")
for r in matches.records:
    print(r.category, r.truth_id, r.forecast_id, r.ti_score)

from python_obj.histogram import read_histogram_file, sum_histograms, by_hour_of_day

hist = read_histogram_file("python_obj/configs/output/hist_mrms/hist_mrms_20230501.nc")
for s in hist.slices:
    print(s.valid_time, s.lead_hours, s.member_id, s.hist.sum())  # lead_hours/member_id are None for MRMS

bins, total = sum_histograms(["hist_mrms_20230501.nc", "hist_mrms_20230502.nc"], predicate=by_hour_of_day(18))
# total -> summed bin counts for every slice valid at 18Z across both days
```
