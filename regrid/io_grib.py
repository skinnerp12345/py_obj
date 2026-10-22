"""Load gridded model forecast fields from GRIB2 files -- the GRIB2 counterpart of
io_grid.load_model_netcdf()/read_valid_time_only()/read_init_time_only().

A GRIB2 file holds many records ("messages"), so the field to read is chosen by
matching message keys rather than by a variable name:

- `grib_selector`: pygrib key/value pairs (e.g. {"discipline": 0,
  "parameterCategory": 16, "parameterNumber": 196}), or a LIST of such dicts
  meaning "any one of these encodings".
- `var_name`: a name from GRIB2_FIELD_PRESETS, used when no grib_selector is
  given. The same physical field can be encoded differently by different
  models' GRIB2 tables -- composite reflectivity is 0/16/196 (NCEP local table)
  in HRRR output but 0/16/5 (WMO standard) in RRFS output -- so a preset lists
  every verified encoding of one field.

Exactly one message must match (across all alternatives together); zero or more
than one raises a ValueError naming the file, the selector and what was found,
rather than silently picking one.

The grid comes from the message itself (`latlons()`), and the times from its own
keys: `analDate` (init) and `validDate` (valid).
"""

from datetime import datetime

import numpy as np

from py_obj.grib2_fields import GRIB2_FIELD_PRESETS, resolve_grib_selector  # noqa: F401 -- re-exported

from .io_mrms import GriddedField


def select_grib2_message(grbs, alternatives: list[dict], filepath: str):
    """The single message matching any of `alternatives` in an open pygrib file."""
    matches = []
    for alt in alternatives:
        grbs.rewind()
        try:
            matches.extend(grbs.select(**alt))
        except ValueError:  # pygrib raises when no message matches (incl. an unknown key)
            pass
    unique = {m.messagenumber: m for m in matches}
    if len(unique) != 1:
        found = [f"#{m.messagenumber} {m.discipline}/{m.parameterCategory}/{m.parameterNumber} "
                 f"{m.typeOfLevel} {m.level}" for m in unique.values()]
        raise ValueError(
            f"'{filepath}': expected exactly 1 GRIB2 message matching {alternatives}, "
            f"found {len(unique)}" + (f": {found}" if found else "")
        )
    return next(iter(unique.values()))


def load_model_grib2(filepath: str, var_name: str | None = "REFC", grib_selector=None) -> GriddedField:
    """One GRIB2 message as a GriddedField: lat/lon from the message's own grid,
    data as float64 (bitmap-masked points -> NaN), valid_time = validDate."""
    import pygrib

    alternatives = resolve_grib_selector(var_name, grib_selector)
    grbs = pygrib.open(filepath)
    try:
        msg = select_grib2_message(grbs, alternatives, filepath)
        lat2d, lon2d = msg.latlons()
        values = msg.values
        data = np.ma.filled(np.ma.asarray(values, dtype=np.float64), np.nan)
        valid_time = msg.validDate
    finally:
        grbs.close()
    return GriddedField(lat2d=np.asarray(lat2d, dtype=np.float64), lon2d=np.asarray(lon2d, dtype=np.float64),
                        data=data, valid_time=valid_time, missing_value=None)


def read_grib2_times(filepath: str, var_name: str | None = "REFC", grib_selector=None) -> tuple[datetime, datetime]:
    """(init_time, valid_time) of the selected message, from its analDate/validDate
    keys -- header only, the data values are never decoded."""
    import pygrib

    alternatives = resolve_grib_selector(var_name, grib_selector)
    grbs = pygrib.open(filepath)
    try:
        msg = select_grib2_message(grbs, alternatives, filepath)
        return msg.analDate, msg.validDate
    finally:
        grbs.close()


def load_grib2_grid(filepath: str, var_name: str | None = "REFC", grib_selector=None):
    """(lat2d, lon2d) of the selected message's grid, without decoding its values."""
    import pygrib

    alternatives = resolve_grib_selector(var_name, grib_selector)
    grbs = pygrib.open(filepath)
    try:
        lat2d, lon2d = select_grib2_message(grbs, alternatives, filepath).latlons()
    finally:
        grbs.close()
    return np.asarray(lat2d, dtype=np.float64), np.asarray(lon2d, dtype=np.float64)
