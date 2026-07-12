# python_obj

A small, standalone Python library for object-based thunderstorm verification
— identifying storm objects (from composite reflectivity), optionally
tracking them in time, and matching forecast objects against truth
observations via a Total Interest score. Based on the method in Skinner et
al. 2025 (WAF-D-24-0238), rebuilt from an earlier duplicated implementation
into one small, configurable, testable library.

Every user-facing decision — remap MRMS onto a model grid or not, mask the
domain or not, track objects in time or not, single-member or ensemble input,
a small storm-scale moving domain (e.g. WoFS) vs. a full-CONUS static domain
(e.g. a next-day CAM ensemble) — is an independent, orthogonal, tunable
parameter, not a hardcoded mode switch. Two example model conventions (WoFS,
MPAS) are bundled as concrete configurations, not as separate code paths.

## What's here

```
python_obj/
  regrid/          conservative MRMS-to-model-grid regridding (xesmf/ESMF)
  obj_core/        object identification, tracking, matching, CONUS masking
  drivers/         standalone, independently-runnable CLI scripts (see drivers/README.md)
  notebooks/       two step-by-step tutorial notebooks (see notebooks/README.md)
  configs/         one shared YAML config schema + example/sample config files
  sample_data/     small, real, bundled MRMS/MPAS/WoFS data for the tests and tutorials
  tests/           pytest suite, self-contained (uses only sample_data/)
  config.py        the config schema + loader
  batch_runner.py  generic parallel case-runner shared by every driver's _batch.py companion
  time_utils.py    shared time-tolerance matching helper
```

`python_obj/` has no dependency on anything outside itself — it can be copied
out of its original repository and used standalone.

## Installation

```bash
conda env create -f environment.yml
conda activate python_obj_env
```

## Quick start

Run the bundled smoke test (interpolates 3 real MRMS files onto a small real
MPAS target grid, then identifies storm objects in them):

```bash
python drivers/interpolate_mrms.py configs/config_smoketest.yaml
python drivers/identify_track_mrms.py configs/config_smoketest.yaml
```

Or open one of the two tutorial notebooks (`notebooks/wofs_tutorial.ipynb`,
`notebooks/mpas_tutorial.ipynb`) for a full step-by-step walkthrough with
real, already-executed output — see `notebooks/README.md`.

Run the test suite:

```bash
pytest tests/
```

Everything above works out of a fresh clone with **zero external data** —
`sample_data/` bundles small, real (not synthetic) MRMS/MPAS/WoFS files
trimmed down to just what the library reads (see `sample_data/README.md` for
exact provenance). Two of the example configs (`configs/config.yaml`,
`configs/config_ensemble.yaml`) are the exception: they intentionally
demonstrate scenarios (a multi-day MRMS range, a 2-member MPAS ensemble) too
large to bundle, and are meant as "bring your own larger dataset" references
— point their paths at your own data to run them.

## Configuration

One shared YAML config file, five independently optional top-level sections
(`interpolation`, `observations`, `model`, `matching`, `linear_classification`,
plus `fetch_mrms`). Populate only the sections your problem needs; each
driver script reads only the section(s) it requires and raises a clear error
naming which section is missing, rather than guessing. See
`drivers/README.md` for the full section/field reference and
`configs/config.yaml` for a fully-populated, chained example.

## License

MIT — see `LICENSE`.
