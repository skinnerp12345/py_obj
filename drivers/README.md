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

`python_obj/configs/config.yaml` has five independently optional top-level sections:

| Section | Used by | Required fields |
|---|---|---|
| `interpolation:` | `interpolate_mrms.py` | `raw_mrms_dir`, `interp_mrms_dir`, `target_grid_file` |
| `observations:` | `identify_track_mrms.py` | `file_format`, `var_name`, `lat_name`, `lon_name`, `boundary_threshold`, `max_value_threshold`, `area_threshold_km2`, `interp_mrms_dir` |
| `model:` | `identify_track_model.py` | same common fields as `observations:` plus `init_attr`, `lead_attr`, `init_format`, `input_dir` |
| `matching:` | `run_matching.py` | `max_boundary_disp_km`, `max_centroid_disp_km`, `ti_threshold`, `truth_object_dir`, `forecast_object_dir` |
| `linear_classification:` | `identify_track_mrms.py`, `identify_track_model.py` | all four threshold fields |
| `fetch_mrms:` | `fetch_mrms.py` | `model_input_dir`, `output_dir`, and either `valid_time_attr`+`valid_time_format` or `init_attr`+`lead_attr`+`init_format` |

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
```
