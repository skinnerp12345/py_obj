# py_obj

A small, standalone Python library for object-based thunderstorm verification
— identifying storm objects (from composite reflectivity), optionally
tracking them in time, matching forecast objects against truth observations
via a Total Interest score, and building reflectivity-distribution
histograms for identifying matched-percentile object-ID thresholds across
sources. Based on the method in Skinner et al. 2025 (WAF-D-24-0238), rebuilt
from an earlier duplicated implementation into one small, configurable,
testable library.

Every user-facing decision — remap MRMS onto a model grid or not, mask the
domain or not, track objects in time or not, single-member or ensemble input,
a small storm-scale moving domain (e.g. WoFS) vs. a full-CONUS static domain
(e.g. a next-day CAM ensemble) — is an independent, orthogonal, tunable
parameter, not a hardcoded mode switch. Two example model conventions (WoFS,
MPAS) are bundled as concrete configurations, not as separate code paths.

## What's here

```
py_obj/
  regrid/          conservative MRMS-to-model-grid regridding (xesmf/ESMF)
  obj_core/        object identification, tracking, matching, CONUS masking
  histogram/       reflectivity-distribution histograms + matched-percentile thresholds
  drivers/         standalone, independently-runnable CLI scripts (see drivers/README.md)
  notebooks/       three step-by-step tutorial notebooks (see notebooks/README.md)
  configs/         one shared YAML config schema + example/sample config files
  sample_data/     small, real, bundled MRMS/MPAS/WoFS data for the tests and tutorials
  tests/           pytest suite, self-contained (uses only sample_data/)
  config.py        the config schema + loader
  batch_runner.py  generic parallel case-runner shared by every driver's _batch.py companion
  time_utils.py    shared time-tolerance matching helper
```

`py_obj/` has no dependency on anything outside itself — it can be copied
out of its original repository and used standalone.

## Installation

```bash
git clone https://github.com/skinnerp12345/py_obj.git
cd py_obj

conda env create -f environment.yml
conda activate py_obj_env
```

Every module in this library imports itself as `py_obj.xxx` (e.g. `from
py_obj.config import load_config`), matching both this GitHub repo's own
name and the folder name a plain `git clone` produces — so nothing further
is required to run any script directly from within the repo (e.g. `python
drivers/interpolate_mrms.py ...`), since Python resolves `py_obj` as a
regular package via the parent directory already being on the path.

To use `py_obj` from a script or notebook OUTSIDE this repo (i.e. not run
from inside it), either add the parent of this checkout to `sys.path`
first, or install it as a real package once per environment:

```bash
pip install -e . --no-deps
```

## Quick start

Run the bundled smoke test (interpolates 3 real MRMS files onto a small real
MPAS target grid, then identifies storm objects in them):

```bash
python drivers/interpolate_mrms.py configs/config_smoketest.yaml
python drivers/identify_track_mrms.py configs/config_smoketest.yaml
```

Or open one of the three tutorial notebooks (`notebooks/wofs_tutorial.ipynb`,
`notebooks/mpas_tutorial.ipynb`, `notebooks/histogram_tutorial.ipynb`) for a
full step-by-step walkthrough with real, already-executed output — see
`notebooks/README.md`.

Run the test suite:

```bash
pytest tests/
```

Everything above works out of a fresh clone with **zero external data** —
`sample_data/` bundles small, real (not synthetic) MRMS/MPAS/WoFS files
trimmed down to just what the library reads (see `sample_data/README.md` for
exact provenance). `configs/config.yaml` (a multi-day MRMS range, a 2-member
MPAS ensemble) is the exception: it intentionally demonstrates a scenario too
large to bundle, and is meant as a "bring your own larger dataset" reference
— point its paths at your own data to run it. The single-purpose
`configs/config_example_*.yaml` files (one per driver — see "Configuration"
below) use the same larger-dataset paths as `config.yaml`.

## Configuration

Each YAML config file here functions like a **namelist** familiar from NWP
models (WRF, MPAS) — one flat set of named parameters per section, no code
changes needed to adjust a run. One shared schema, independently optional
top-level sections (`interpolation`, `observations`, `model`, `matching`,
`linear_classification`, `fetch_mrms`, `histogram_observations`,
`histogram_model`). Populate only the sections your problem needs; each
driver script reads only the section(s) it requires and raises a clear error
naming which section is missing, rather than guessing.

For a single-purpose example matching exactly one driver's own required
section(s), see `configs/config_example_<driver_name>.yaml` (e.g.
`config_example_interpolate_mrms.yaml`, `config_example_run_matching.yaml`).
`configs/config.yaml` is instead a fully-populated, chained example spanning
every section end to end (each section's output feeding the next's input),
and doubles as every driver's own default config path when none is given on
the command line. See `drivers/README.md` for the full section/field
reference and `configs/CONFIG_REFERENCE.md` for a field-by-field reference of
every config option.

## License

MIT — see `LICENSE`.
