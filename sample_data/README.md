# Sample data

Small, **real** (not synthetic) data bundled with the library so the test
suite and the two tutorial notebooks (`python_obj/notebooks/`) can run with
zero external data -- just clone and go.

## Provenance

- **`mpas_case/`** -- a real MPAS 3-km CONUS forecast (init 2023-05-01 00Z,
  member 1), lead times f001-f003 (the first lead time with any real
  reflectivity signal is f001 -- f000 is empty). Cropped to a fixed
  250x250-pixel window (~750x750 km, Upper Midwest / Great Lakes region)
  containing a real, evolving storm cluster (3.9% / 33.1% / 39.1% of the
  window >=20 dBZ across f001/f002/f003) chosen to be fully within MRMS's
  real radar coverage (confirmed empirically: ~0% "no coverage" cells in the
  corresponding MRMS footprint -- an earlier candidate crop near the
  Atlantic coast had ~45% offshore no-coverage cells and was rejected),
  trimmed to just `refl10cm_max`/`latitude`/`longitude` + the
  `initializationTime`/`forecastHour` attributes `load_model_netcdf()` reads.
  Paired with 3 real raw MRMS `.grib2.gz` files (2023-05-01 01Z-03Z) -- these
  needed no trimming, native MRMS files are already only ~1.7-1.9 MB each.
- **`wofs_case/`** -- 3 consecutive real WoFS 5-min forecast files (init
  2026-05-18 23:00Z, valid 2300/2305/2310Z), all `NE=18` ensemble members kept
  intact (per-member identification is a real feature this sample
  demonstrates), trimmed to just `comp_dz`/`xlat`/`xlon` +
  `valid_time`/`init_time`. Paired with the 3 real MRMS `.grib2.gz` files
  nearest each WoFS valid time, already fetched from the public
  `noaa-mrms-pds` AWS archive in an earlier session (see
  `python_obj/drivers/fetch_mrms.py`).

Both source models' full files are 234-281 MB each -- almost entirely unused
WRF diagnostic variables this library never reads. Trimming down to just the
needed fields (+ cropping MPAS's much larger native grid) shrinks them by
>99.9% with zero loss of fidelity for anything this library actually does:
MPAS lead times are ~0.3 MB each here (was ~280 MB), WoFS files are
~1.7-1.8 MB each (was ~235 MB). Total bundle size: ~17 MB.

## Regenerating

`build_sample_data.py` is the exact script that produced everything in this
directory from this developer's own local copies of `test_mrms/`, `test_mpas/`,
`test_wofs/`, and `python_obj/configs/output/fetched_mrms_wofs/`. It is not run
automatically by anything else -- it's a provenance record, kept in case the
sample ever needs to be regenerated or extended (e.g. a different crop window,
more lead times). Requires those larger source directories to exist locally;
not runnable from a fresh clone that only has this trimmed output.

## What's NOT bundled here

`config.yaml` and `config_ensemble.yaml` (the two example configs that
demonstrate a multi-day MRMS range and a 2-member MPAS ensemble) intentionally
still point at the larger, non-bundled `test_mrms/`/`test_mpas/` directories --
those two configs are "bring your own larger dataset" reference examples, not
meant to run out of a fresh clone. See the top-level README for details.
