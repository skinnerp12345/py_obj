# python_obj notebooks

Three paired, step-by-step walkthroughs of the pipeline, each with its own
"sample config" (`python_obj/configs/config_sample_*.yaml`), all running
against the small, real data bundled in `python_obj/sample_data/` -- no
external test data needed (see `sample_data/README.md` for provenance):

- **`wofs_tutorial.ipynb`** -- the full pipeline end to end: fetching
  matching MRMS observations from AWS, interpolating them onto the model
  grid, identifying objects in both the model and truth series, tracking
  the truth series, and matching them. Run against the real bundled WoFS
  case (`sample_data/wofs_case/`, an 18-member ensemble). Uses
  `configs/config_sample_wofs.yaml`.
- **`mpas_tutorial.ipynb`** -- object ID + matching only, using data that's
  already available on disk (a single-member MPAS forecast in
  `sample_data/mpas_case/mpas_mem1/`, f001-f003, matched against MRMS
  already interpolated onto the MPAS grid in
  `sample_data/mpas_case/interp_mrms/`) -- no fetch/interpolate steps
  needed. Both sides are identified without tracking. Uses
  `configs/config_sample_mpas.yaml`.
- **`histogram_tutorial.ipynb`** -- builds a per-day MRMS reflectivity-
  distribution histogram and a per-forecast MPAS histogram from the same
  bundled `sample_data/mpas_case/`, demonstrates subsetting them (by hour
  of day, by forecast lead time), and computes a real matched-percentile
  threshold between the two (e.g. "what MPAS dBZ value is the same
  percentile as MRMS's 40 dBZ?"). Uses `configs/config_sample_histogram.yaml`.

## Prerequisites

- The `pysteps_env` conda environment (`/opt/anaconda3/envs/pysteps_env/bin/python`),
  which has `jupyter`/`nbformat` plus every dependency `python_obj/` itself needs
  (`xesmf`, `netCDF4`, `pyproj`, `matplotlib`, etc).
- For `wofs_tutorial.ipynb` only: internet access (Step 1 fetches real files
  from the public `noaa-mrms-pds` AWS S3 bucket -- no credentials needed).
  `mpas_tutorial.ipynb`/`histogram_tutorial.ipynb` need no internet access.

## Running them

```
/opt/anaconda3/envs/pysteps_env/bin/jupyter notebook wofs_tutorial.ipynb
/opt/anaconda3/envs/pysteps_env/bin/jupyter notebook mpas_tutorial.ipynb
/opt/anaconda3/envs/pysteps_env/bin/jupyter notebook histogram_tutorial.ipynb
```

or, to execute any of them non-interactively end to end:

```
/opt/anaconda3/envs/pysteps_env/bin/jupyter nbconvert --to notebook --execute --inplace wofs_tutorial.ipynb
```

All outputs (fetched/interpolated files, object files, match files,
histogram files) are written under `python_obj/configs/output/`, per the
paths configured in each notebook's sample config. To adapt any notebook to
a different model or case, copy it and change the `CONFIG_PATH` variable in
its Setup cell to point at a different config file -- see
`python_obj/drivers/README.md` for the full config-field reference and
other example configs under `python_obj/configs/`.
