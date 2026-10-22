"""GRIB2 record identification shared by the GRIB2 reader (regrid/io_grib.py) and
config validation (config.py). Kept top-level, like filename_time.py, so
config.py can validate selectors without importing py_obj.regrid (which pulls in
xesmf/ESMF).

The same physical field can be encoded differently by different models' GRIB2
tables -- composite reflectivity is 0/16/196 (NCEP local table) in HRRR output
but 0/16/5 (WMO standard) in RRFS output -- so a preset lists every verified
encoding of one field, and the reader requires exactly one message to match.
"""

# Verified encodings only (each confirmed on real files; see the module docstring).
# Add a field or encoding here only after confirming it on real data; anything
# else should be passed explicitly as grib_selector.
GRIB2_FIELD_PRESETS: dict[str, list[dict]] = {
    "REFC": [  # composite reflectivity (wgrib2 name "REFC")
        {"discipline": 0, "parameterCategory": 16, "parameterNumber": 196},  # NCEP local table (e.g. HRRR)
        {"discipline": 0, "parameterCategory": 16, "parameterNumber": 5},    # WMO standard (e.g. RRFS)
    ],
}


def resolve_grib_selector(var_name: str | None, grib_selector=None) -> list[dict]:
    """The list of alternative key dicts to match: an explicit grib_selector
    (dict or list of dicts) wins; otherwise var_name must be a preset name."""
    if grib_selector is not None:
        alternatives = [grib_selector] if isinstance(grib_selector, dict) else list(grib_selector)
        if not alternatives or not all(isinstance(a, dict) and a for a in alternatives):
            raise ValueError(f"grib_selector must be a non-empty dict or a list of non-empty dicts, got {grib_selector!r}")
        return alternatives
    if var_name in GRIB2_FIELD_PRESETS:
        return GRIB2_FIELD_PRESETS[var_name]
    raise ValueError(
        f"GRIB2 field {var_name!r} has no built-in preset (known: {sorted(GRIB2_FIELD_PRESETS)}); "
        "pass grib_selector with the pygrib keys that identify the record"
    )
