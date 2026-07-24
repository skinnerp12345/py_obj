"""Load a target grid definition, or a full gridded data field, from a model NetCDF file."""

from datetime import datetime, timedelta
from typing import Callable

import netCDF4
import numpy as np

from .grid_spec import GridSpec
from .io_mrms import GriddedField


def _squeeze_leading(arr: np.ndarray) -> np.ndarray:
    """Drop leading singleton dims (e.g. a length-1 time dim on lat/lon/data)."""
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _squeeze_singleton_leading(arr: np.ndarray) -> np.ndarray:
    """Like _squeeze_leading, but only drops a leading dim if it's genuinely
    size 1 -- never silently discards a real, non-singleton leading axis
    (e.g. WoFS's comp_dz(ne=18, lat, lon), where naively taking arr[0] would
    keep only one ensemble member and silently drop the other 17)."""
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _select_data_slice(
    data: np.ndarray,
    filepath: str,
    varname: str,
    extra_dim_index: int | None,
    extra_dim_selector_fn: Callable[[np.ndarray], np.ndarray] | None,
) -> np.ndarray:
    """Reduce a data variable's array down to a single 2D field, for models
    whose data variable carries a real extra axis beyond (lat, lon) -- e.g.
    an ensemble-member dimension. Precedence mirrors _resolve_valid_time()'s
    layered design exactly:

      1. extra_dim_selector_fn(data) -- escape hatch for any convention that
         doesn't fit the simple "index into the leading extra axis" case
         below (a different axis position, a reduction instead of an index,
         etc.); must return a 2D array or this raises.
      2. extra_dim_index -- index into the first real axis remaining after
         best-effort singleton squeeze (the common case: one real extra
         axis, e.g. WoFS's comp_dz(ne, lat, lon)).
      3. neither given -- best-effort singleton squeeze only; if more than 2
         real dimensions remain, RAISE (naming the shape) instead of
         silently keeping index 0 -- this is the fix for a real bug found
         against test_wofs/*.nc, where the old _squeeze_leading silently
         discarded 17 of 18 real ensemble members.
    """
    if extra_dim_selector_fn is not None:
        result = np.asarray(extra_dim_selector_fn(data))
        if result.ndim != 2:
            raise ValueError(
                f"extra_dim_selector_fn for '{varname}' in '{filepath}' returned "
                f"ndim={result.ndim} (shape {result.shape}); must return a 2D array."
            )
        return result

    squeezed = _squeeze_singleton_leading(data)

    if extra_dim_index is not None:
        if squeezed.ndim < 3:
            raise ValueError(
                f"extra_dim_index={extra_dim_index} given for '{varname}' in '{filepath}', "
                f"but it has only {squeezed.ndim} real dim(s) (shape {squeezed.shape}) -- nothing to index."
            )
        sliced = _squeeze_singleton_leading(squeezed[extra_dim_index])
        if sliced.ndim != 2:
            raise ValueError(
                f"After extra_dim_index={extra_dim_index}, '{varname}' in '{filepath}' "
                f"is still shape {sliced.shape}, not 2D."
            )
        return sliced

    if squeezed.ndim > 2:
        raise ValueError(
            f"'{varname}' in '{filepath}' has {squeezed.ndim} real (non-singleton) dimensions "
            f"after best-effort squeeze (shape {squeezed.shape}), and neither extra_dim_index nor "
            f"extra_dim_selector_fn was given. Pass one -- silently keeping index 0 would discard "
            f"real data (e.g. WoFS's comp_dz(ne, lat, lon), ne=18 real ensemble members)."
        )
    return squeezed


def infer_stacked_member_count(filepath: str, varname: str) -> int:
    """Cheap metadata-only shape read (never touches array data, same pattern
    as read_valid_time_only) -- discovers how many members are stacked along
    a data variable's one real extra leading axis, e.g. WoFS's
    comp_dz(ne=18, lat, lon). Chosen over an explicit n_members config value
    since deriving it from the file itself can never drift out of sync with
    the real data.
    """
    with netCDF4.Dataset(filepath, "r") as ds:
        shape = ds.variables[varname].shape
    squeezed = shape
    while len(squeezed) > 2 and squeezed[0] == 1:
        squeezed = squeezed[1:]
    if len(squeezed) != 3:
        raise ValueError(
            f"'{varname}' in '{filepath}' has shape {shape} (real shape {squeezed} after "
            "singleton-squeeze); infer_stacked_member_count() requires exactly one real "
            "extra leading axis beyond (lat, lon)."
        )
    return squeezed[0]


def load_target_grid(example_file: str, lat_name: str, lon_name: str) -> GridSpec:
    """Read 2D (or 1D, broadcast to 2D) lat/lon coordinate arrays from a model file.

    Works for any grid size, including domains larger than or only partially
    overlapping MRMS's native coverage -- this function only reads coordinates,
    it does not assume anything about data availability.
    """
    with netCDF4.Dataset(example_file, "r") as ds:
        lat = np.asarray(ds.variables[lat_name][...], dtype=np.float64)
        lon = np.asarray(ds.variables[lon_name][...], dtype=np.float64)

    # some files carry a leading time dim on 2D lat/lon (e.g. the MPAS test file's
    # latitude(time,lat,lon)); squeeze any leading singleton dims down to 2D.
    lat = _squeeze_leading(lat)
    lon = _squeeze_leading(lon)

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        lat2d, lon2d = lat, lon
    else:
        raise ValueError(
            f"Unsupported lat/lon dimensionality in '{example_file}': "
            f"lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
        )

    return GridSpec(lat2d=lat2d, lon2d=lon2d)


def _resolve_valid_time(
    ds: netCDF4.Dataset,
    filepath: str,
    valid_time: datetime | None,
    init_attr: str,
    lead_attr: str,
    lead_units: str,
    init_format: str,
    valid_time_attr: str | None,
    valid_time_format: str | None,
    valid_time_fn: Callable[[netCDF4.Dataset], datetime] | None,
) -> datetime:
    """Shared valid_time derivation for load_model_netcdf()/read_valid_time_only().

    Precedence (first that applies wins), reflecting that different models
    store valid/init time in genuinely different conventions -- this is
    deliberately layered rather than a single hardcoded mechanism:

      1. explicit `valid_time` -- caller already knows it, nothing to derive.
      2. `valid_time_fn(ds)` -- caller-supplied escape hatch for any
         convention that fits neither of the two built-in modes below.
         Exceptions raised inside it propagate uncaught (it's the caller's
         own logic; not this function's job to guess what went wrong).
      3. `valid_time_attr`+`valid_time_format` -- a ready-made datetime
         STRING global attribute (e.g. WoFS's `valid_time="20260518_230000"`,
         format "%Y%m%d_%H%M%S") -- no init+lead arithmetic needed at all.
      4. `init_attr`+`lead_attr`+`lead_units`+`init_format` -- the original
         MPAS-style convention: init time + a lead-time NUMBER needing
         `timedelta(**{lead_units: lead_value})` arithmetic. Stays the
         default fallback so existing MPAS callers are unaffected by the
         two newer modes above.
    """
    if valid_time is not None:
        return valid_time

    if valid_time_fn is not None:
        return valid_time_fn(ds)

    if valid_time_attr is not None or valid_time_format is not None:
        if valid_time_attr is None or valid_time_format is None:
            raise ValueError(
                "valid_time_attr and valid_time_format must both be given, or neither "
                f"(got valid_time_attr={valid_time_attr!r}, valid_time_format={valid_time_format!r})"
            )
        if not hasattr(ds, valid_time_attr):
            raise ValueError(f"'{filepath}' is missing the '{valid_time_attr}' global attribute.")
        return datetime.strptime(getattr(ds, valid_time_attr), valid_time_format)

    if not (hasattr(ds, init_attr) and hasattr(ds, lead_attr)):
        raise ValueError(
            f"'{filepath}' is missing the '{init_attr}'/'{lead_attr}' global "
            "attributes; pass valid_time (or valid_time_attr/valid_time_fn) explicitly "
            "rather than guessing it."
        )
    if lead_units not in ("hours", "minutes", "seconds"):
        raise ValueError(f"lead_units must be 'hours', 'minutes', or 'seconds', got '{lead_units}'")
    init_time = datetime.strptime(getattr(ds, init_attr), init_format)
    lead_value = float(getattr(ds, lead_attr))
    return init_time + timedelta(**{lead_units: lead_value})


def load_model_netcdf(
    filepath: str,
    varname: str,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    init_attr: str = "initializationTime",
    lead_attr: str = "forecastHour",
    lead_units: str = "hours",
    init_format: str = "%Y%m%d%H",
    valid_time: datetime | None = None,
    valid_time_attr: str | None = None,
    valid_time_format: str | None = None,
    valid_time_fn: Callable[[netCDF4.Dataset], datetime] | None = None,
    extra_dim_index: int | None = None,
    extra_dim_selector_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> GriddedField:
    """Load one gridded data field from a model output NetCDF file with 2D lat/lon
    coordinates (validated against test_mpas/*.nc's `refl10cm_max`).

    valid_time derivation (see _resolve_valid_time() for the full precedence
    order): if not given explicitly, tries valid_time_fn, then
    valid_time_attr/valid_time_format (a ready-made datetime string, e.g.
    WoFS's `valid_time="20260518_230000"`), then falls back to the original
    init_attr+lead_attr arithmetic (e.g. MPAS test files carry
    `initializationTime="2023050100"` and `forecastHour="12"`). Not parsed
    from the filename in any mode -- more robust, and generalizes to any
    similarly-structured model output.

    lead_units ("hours"|"minutes"|"seconds"): only relevant to the
    init_attr/lead_attr fallback mode. The unit `lead_attr`'s raw stored
    number is already in -- NOT a description of the model's output cadence.
    MPAS's hourly output happens to store `forecastHour` in hours (lead_units=
    "hours" is correct there), but a 5-minute-cadence model isn't required to
    use "minutes" either -- match whatever unit that model's own attribute
    actually holds (e.g. a fractional-hour value would still need "hours").

    extra_dim_index/extra_dim_selector_fn: for a data variable with a real
    extra axis beyond (lat, lon) -- e.g. WoFS's comp_dz(ne=18, lat, lon), an
    ensemble-member dimension stacked inside one file. See
    _select_data_slice() for the full precedence; if the variable genuinely
    has more than 2 real dimensions and neither is given, this raises rather
    than silently keeping index 0. lat/lon are assumed shared across
    whatever this extra axis represents (e.g. one grid for all ensemble
    members) and never indexed by it.
    """
    with netCDF4.Dataset(filepath, "r") as ds:
        lat2d = _squeeze_singleton_leading(np.asarray(ds.variables[lat_name][...], dtype=np.float64))
        lon2d = _squeeze_singleton_leading(np.asarray(ds.variables[lon_name][...], dtype=np.float64))
        if lat2d.ndim != 2 or lon2d.ndim != 2:
            raise ValueError(
                f"'{lat_name}'/'{lon_name}' in '{filepath}' have shapes "
                f"{lat2d.shape}/{lon2d.shape} after singleton-squeeze, not 2D."
            )
        raw_data = np.asarray(ds.variables[varname][...], dtype=np.float64)
        data = _select_data_slice(raw_data, filepath, varname, extra_dim_index, extra_dim_selector_fn)

        data_var = ds.variables[varname]
        missing_value = float(data_var._FillValue) if hasattr(data_var, "_FillValue") else None

        valid_time = _resolve_valid_time(
            ds, filepath, valid_time, init_attr, lead_attr, lead_units, init_format,
            valid_time_attr, valid_time_format, valid_time_fn,
        )

    return GriddedField(lat2d=lat2d, lon2d=lon2d, data=data, valid_time=valid_time, missing_value=missing_value)


def read_valid_time_only(
    filepath: str,
    init_attr: str = "initializationTime",
    lead_attr: str = "forecastHour",
    lead_units: str = "hours",
    init_format: str = "%Y%m%d%H",
    valid_time_attr: str | None = None,
    valid_time_format: str | None = None,
    valid_time_fn: Callable[[netCDF4.Dataset], datetime] | None = None,
) -> datetime:
    """Read just a model file's valid_time, without loading its lat/lon grid
    or data variable -- for callers (e.g. a fetch/discovery script) that only
    need each file's timestamp, not its full field. Cheap regardless of file
    size, since netCDF4 only reads array data when a variable is sliced;
    opening a Dataset and reading global attributes never touches variable
    data. Shares _resolve_valid_time()'s precedence with load_model_netcdf(),
    so any new time-derivation mode is added once, not twice.
    """
    with netCDF4.Dataset(filepath, "r") as ds:
        return _resolve_valid_time(
            ds, filepath, None, init_attr, lead_attr, lead_units, init_format,
            valid_time_attr, valid_time_format, valid_time_fn,
        )


def read_init_time_only(
    filepath: str,
    init_attr: str = "initializationTime",
    init_format: str = "%Y%m%d%H",
    valid_time_attr: str | None = None,
    valid_time_format: str | None = None,
    init_time_attr: str = "init_time",
) -> datetime:
    """Read just a model file's forecast INIT time (not valid_time) -- needed
    to group/name output by forecast case (see obj_core's
    file_grouping='init_snapshot'). Cheap, attribute-only read, mirroring
    read_valid_time_only()'s own convention.

    Two modes, matching _resolve_valid_time()'s two attribute-based
    conventions (the escape-hatch valid_time/valid_time_fn modes have no
    init_time equivalent -- a caller using those must derive init_time some
    other way):
      - init_attr+init_format (MPAS-style arithmetic mode): init_time is
        read directly from init_attr -- no lead-time arithmetic needed for
        this value specifically.
      - valid_time_attr given (WoFS-style string mode): init_time is read
        from init_time_attr, parsed with the SAME format as valid_time_attr's
        own string (matching this convention's established "a same-format
        init_time alongside valid_time" pattern -- see
        HistogramModelConfig.init_time_attr / build_histogram_model.py's
        _compute_lead_hours(), which already relies on this exact
        assumption).
    """
    with netCDF4.Dataset(filepath, "r") as ds:
        if valid_time_attr is not None:
            if not hasattr(ds, init_time_attr):
                raise ValueError(f"'{filepath}' is missing the '{init_time_attr}' global attribute.")
            return datetime.strptime(getattr(ds, init_time_attr), valid_time_format)
        if not hasattr(ds, init_attr):
            raise ValueError(f"'{filepath}' is missing the '{init_attr}' global attribute.")
        return datetime.strptime(getattr(ds, init_attr), init_format)
