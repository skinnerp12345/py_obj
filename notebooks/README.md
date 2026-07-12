# python_obj notebooks

Two paired, step-by-step walkthroughs of the pipeline, each with its own
"sample config" (`python_obj/configs/config_sample_*.yaml`):

- **`wofs_tutorial.ipynb`** -- the full pipeline end to end: fetching
  matching MRMS observations from AWS, interpolating them onto the model
  grid, identifying objects in both the model and truth series, tracking
  the truth series, and matching them. Run against the real example WoFS
  case bundled with this repo (`test_wofs/`, an 18-member ensemble). Uses
  `configs/config_sample_wofs.yaml`.
- **`mpas_tutorial.ipynb`** -- object ID + matching only, using data that's
  already available on disk (a single-member MPAS forecast in
  `test_mpas/mem1/`, f000-f012, matched against MRMS already interpolated
  onto the MPAS grid in `interp_mrms/20230501/`) -- no fetch/interpolate
  steps needed. Both sides are identified without tracking. Uses
  `configs/config_sample_mpas.yaml`.

## Prerequisites

- The `pysteps_env` conda environment (`/opt/anaconda3/envs/pysteps_env/bin/python`),
  which has `jupyter`/`nbformat` plus every dependency `python_obj/` itself needs
  (`xesmf`, `netCDF4`, `pyproj`, `matplotlib`, etc).
- For `wofs_tutorial.ipynb`: internet access (Step 1 fetches real files from
  the public `noaa-mrms-pds` AWS S3 bucket -- no credentials needed) and
  `test_wofs/` present at the repo root, unmodified.
- For `mpas_tutorial.ipynb`: `test_mpas/mem1/` and `interp_mrms/20230501/`
  present at the repo root, unmodified. No internet access needed.

## Running them

```
/opt/anaconda3/envs/pysteps_env/bin/jupyter notebook wofs_tutorial.ipynb
/opt/anaconda3/envs/pysteps_env/bin/jupyter notebook mpas_tutorial.ipynb
```

or, to execute either non-interactively end to end:

```
/opt/anaconda3/envs/pysteps_env/bin/jupyter nbconvert --to notebook --execute --inplace wofs_tutorial.ipynb
/opt/anaconda3/envs/pysteps_env/bin/jupyter nbconvert --to notebook --execute --inplace mpas_tutorial.ipynb
```

All outputs (fetched/interpolated files, object files, match files) are
written under `python_obj/configs/output/`, per the paths configured in each
notebook's sample config. To adapt either notebook to a different model or
case, copy it and change the `CONFIG_PATH` variable in its Setup cell to
point at a different config file -- see `python_obj/drivers/README.md` for
the full config-field reference and other example configs under
`python_obj/configs/`.
