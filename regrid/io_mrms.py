"""Loaders for MRMS composite reflectivity fields.

Returns a common `GriddedField` regardless of source file format, so downstream
regridding code never needs to know whether the data came from GRIB2 or NetCDF --
and, since a model output field is structurally the same thing (lat/lon + one
data field + a valid time), this same type is reused for model loaders too
(see `io_grid.py`'s `load_model_netcdf`).
"""

import gzip
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class GriddedField:
    lat2d: np.ndarray
    lon2d: np.ndarray
    data: np.ndarray
    valid_time: datetime
    missing_value: float | None = None


# Kept as an alias so Step 1/1b code and tests (written before this was
# generalized) keep working unmodified.
MRMSField = GriddedField


def load_mrms(filepath: str, file_format: str | None = None, **kwargs) -> MRMSField:
    """Dispatch to a format-specific loader.

    file_format: "grib2" or "netcdf". If None, sniffed from the filename
    (".grib2"/".grib2.gz" -> grib2, ".nc" -> netcdf).
    """
    if file_format is None:
        lower = filepath.lower()
        if ".grib2" in lower or lower.endswith(".grb2") or lower.endswith(".grb2.gz"):
            file_format = "grib2"
        elif lower.endswith(".nc"):
            file_format = "netcdf"
        else:
            raise ValueError(
                f"Could not infer MRMS file format from filename '{filepath}'; "
                "pass file_format explicitly ('grib2' or 'netcdf')."
            )

    if file_format == "grib2":
        return load_mrms_grib2(filepath, **kwargs)
    elif file_format == "netcdf":
        return load_mrms_netcdf(filepath, **kwargs)
    else:
        raise ValueError(f"Unknown MRMS file_format '{file_format}'")


# MRMS composite reflectivity products carry two distinct large-magnitude negative
# sentinel values, confirmed against real test data and by the user (domain expert,
# 2026-07-06) -- NOT trust the GRIB2 header's own `missingValue` key for this product
# family (it reads a meaningless 9999 on real MRMS_MergedReflectivityQCComposite files):
#
#   -999  "no observational coverage" (_FillValue convention) -- must be EXCLUDED from
#          any averaging/regridding, not treated as a data value.
#   -99   "a real observation exists, at or below the minimum detectable/near-zero
#          reflectivity" (missing_value convention, despite the name) -- must be
#          REPLACED with a physically-reasonable floor (0 dBZ) before use, not
#          excluded, since excluding it would bias averages high by dropping all the
#          legitimate low/no-precip observations.
MRMS_MISSING_VALUE = -999.0
MRMS_NEAR_ZERO_VALUE = -99.0
MRMS_NEAR_ZERO_FLOOR = 0.0


def clip_near_zero_sentinel(
    data: np.ndarray,
    sentinel: float = MRMS_NEAR_ZERO_VALUE,
    floor: float = MRMS_NEAR_ZERO_FLOOR,
) -> np.ndarray:
    """Replace MRMS's -99 'valid, near/below-zero reflectivity' sentinel with a
    physically-reasonable floor value (0 dBZ by default).

    A standalone, explicit preprocessing step (not applied automatically inside
    the loader) so a caller who wants the untouched raw field still can -- but for
    any conservative averaging/regridding, this must be applied before calling
    regrid_field, since regrid_field's `missing_value` handling only excludes the
    true -999 "no coverage" sentinel, not -99.
    """
    return np.where(data == sentinel, floor, data)


def load_mrms_grib2(filepath: str, grib_selector: dict | None = None) -> MRMSField:
    """Load a single-field MRMS GRIB2 message (e.g. MergedReflectivityQCComposite).

    Transparently gunzips `.grib2.gz` inputs to a temp file, since pygrib requires
    a real file path (cannot stream a gzip object directly).
    """
    import pygrib

    tmp_path = None
    try:
        if filepath.endswith(".gz"):
            fd, tmp_path = tempfile.mkstemp(suffix=".grib2")
            os.close(fd)
            with gzip.open(filepath, "rb") as f_in, open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            read_path = tmp_path
        else:
            read_path = filepath

        grbs = pygrib.open(read_path)
        try:
            if grib_selector:
                matches = grbs.select(**grib_selector)
                if not matches:
                    raise ValueError(
                        f"No GRIB2 message in '{filepath}' matched selector {grib_selector}"
                    )
                grb = matches[0]
            else:
                grb = grbs.message(1)

            lat2d, lon2d = grb.latlons()
            data = np.asarray(grb.values, dtype=np.float64)
            valid_time = grb.validDate
        finally:
            grbs.close()
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return MRMSField(
        lat2d=np.asarray(lat2d, dtype=np.float64),
        lon2d=np.asarray(lon2d, dtype=np.float64),
        data=data,
        valid_time=valid_time,
        missing_value=MRMS_MISSING_VALUE,
    )


def load_mrms_netcdf(
    filepath: str,
    varname: str = "refl_consv",
    lat_name: str = "lat",
    lon_name: str = "lon",
    valid_time: datetime | None = None,
) -> MRMSField:
    """Load pre-interpolated MRMS NetCDF -- e.g. `python_obj.regrid.batch_interpolate`'s
    own output (Step 1b), which also matches the variable naming convention
    python_base's legacy `load_mrms_old`/`load_mrms_new` already expect (`lat`,
    `lon`, `refl_consv`).

    valid_time: if not given, read from the file's own `valid_time` global
    attribute (written by `write_interpolated_mrms_netcdf`). If that attribute
    is absent, raises rather than guessing from the filename -- callers of a
    file produced by some other means must supply valid_time explicitly.
    """
    import netCDF4

    with netCDF4.Dataset(filepath, "r") as ds:
        lat2d = np.asarray(ds.variables[lat_name][:], dtype=np.float64)
        lon2d = np.asarray(ds.variables[lon_name][:], dtype=np.float64)
        data = np.asarray(ds.variables[varname][:], dtype=np.float64)

        data_var = ds.variables[varname]
        missing_value = float(data_var._FillValue) if hasattr(data_var, "_FillValue") else MRMS_MISSING_VALUE

        if valid_time is None:
            if hasattr(ds, "valid_time"):
                valid_time = datetime.fromisoformat(ds.valid_time)
            else:
                raise ValueError(
                    f"'{filepath}' has no 'valid_time' global attribute; pass valid_time "
                    "explicitly rather than guessing it from the filename."
                )

    return MRMSField(lat2d=lat2d, lon2d=lon2d, data=data, valid_time=valid_time, missing_value=missing_value)
