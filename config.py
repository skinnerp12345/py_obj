"""Unified config layer for the whole pipeline family (interpolation, MRMS
object ID/tracking, model object ID/tracking, matching) plus the shared
linear-classification thresholds.

Optional, additive convenience layer -- identify_objects()/run_object_id_series()/
masking functions keep taking plain arguments and stay independently callable from
a notebook or one-off script. This module only turns a YAML file into plain
dataclasses; it never calls into obj_core itself.

One config file, five independently OPTIONAL top-level sections
(interpolation/observations/model/matching/linear_classification). A user
populates only the sections relevant to their problem; each driver script
(python_obj/drivers/) reads only the section(s) it needs and calls
require_section() to fail loudly if a section IT needs is missing -- a
section absent from the YAML is not itself an error, since a different
driver may not need it at all.

Deliberately flat within each section, not per-model-named: `observations`/
`model` each hold exactly one active recipe, not a dict of named presets (no
"mpas:"/"wofs:" sub-keys). To verify a different model, edit the values in
place or point load_config() at a different file -- the schema doesn't
encode per-model special-casing.

Path-shaped fields (any directory/file path) are resolved relative to the
CONFIG FILE'S OWN directory, not the caller's current working directory --
so the same config.yaml behaves identically regardless of where a script
invoking it happens to be run from.
"""

import os
from dataclasses import dataclass

import yaml

_VALID_MASKS = ("none", "conus", "conus_east")
_VALID_FILE_GROUPINGS = ("single", "member_series", "ensemble_snapshot", "full")
_VALID_LEAD_UNITS = ("hours", "minutes", "seconds")

_COMMON_REQUIRED = (
    "file_format", "var_name", "lat_name", "lon_name",
    "boundary_threshold", "max_value_threshold", "area_threshold_km2",
)
_COMMON_OPTIONAL_DEFAULTS = {
    "mask": "none",
    "track": False,
    "track_distance_km": 0.0,
    "file_grouping": "single",
}

_INTERPOLATION_PATH_FIELDS = ("raw_mrms_dir", "interp_mrms_dir", "target_grid_file", "weight_cache_dir")
_OBSERVATIONS_PATH_FIELDS = ("interp_mrms_dir", "object_output_dir")
_MODEL_PATH_FIELDS = ("input_dir", "object_output_dir")
_MATCHING_PATH_FIELDS = ("truth_object_dir", "forecast_object_dir", "output_dir")
_FETCH_MRMS_PATH_FIELDS = ("model_input_dir", "output_dir")
_HISTOGRAM_OBSERVATIONS_PATH_FIELDS = ("interp_mrms_dir", "output_dir")
_HISTOGRAM_MODEL_PATH_FIELDS = ("input_dir", "output_dir")


@dataclass
class InterpolationConfig:
    raw_mrms_dir: str
    interp_mrms_dir: str
    target_grid_file: str
    target_lat_name: str = "latitude"
    target_lon_name: str = "longitude"
    n_workers: int = 8
    weight_cache_dir: str = "output/weight_cache"
    date_range: tuple[str, str] | None = None
    max_files: int | None = None
    file_pattern: str = "**/*.grib2*"  # restricts discovery to one MRMS product when
                                        # a date directory holds more than one (e.g.
                                        # "**/*MergedReflectivityQCComposite*" to skip
                                        # MESH/RotationTrack files sitting alongside it


@dataclass
class ObservationConfig:
    file_format: str
    var_name: str
    lat_name: str
    lon_name: str
    boundary_threshold: float
    max_value_threshold: float
    area_threshold_km2: float
    interp_mrms_dir: str  # Step 2's input dir (identify_track_mrms.py's glob root)
    mask: str = "none"
    track: bool = False
    track_distance_km: float = 0.0
    file_grouping: str = "single"
    object_output_dir: str = "output/obj_mrms"


@dataclass
class ModelConfig:
    file_format: str
    var_name: str
    lat_name: str
    lon_name: str
    boundary_threshold: float
    max_value_threshold: float
    area_threshold_km2: float
    input_dir: str
    # Exactly one of these two time-derivation modes must be given (see
    # _validate_model_time_mode()): a ready-made valid_time STRING attribute
    # (e.g. WoFS's valid_time="20260518_230000"), or init+lead arithmetic
    # (e.g. MPAS's initializationTime+forecastHour).
    init_attr: str | None = None
    lead_attr: str | None = None
    init_format: str | None = None
    valid_time_attr: str | None = None
    valid_time_format: str | None = None
    mask: str = "none"
    track: bool = False
    track_distance_km: float = 0.0
    file_grouping: str = "single"
    lead_units: str = "hours"
    member_subdirs: bool = False
    member_subdir_pattern: str = "*"  # restricts which subdirectories count as members when
                                       # member_subdirs=True and input_dir has non-member siblings
                                       # (e.g. "mem[0-9]*" to skip a sibling like ens_mean_5mems)
    stacked_members: bool = False
    file_pattern: str = "*.nc"
    object_output_dir: str = "output/obj_model"


@dataclass
class MatchingConfig:
    max_boundary_disp_km: float
    max_centroid_disp_km: float
    ti_threshold: float
    truth_object_dir: str
    forecast_object_dir: str
    max_time_offset_minutes: float = 5.0
    output_dir: str = "output/matches"
    file_pattern: str = "*.nc"


@dataclass
class FetchMrmsConfig:
    model_input_dir: str
    output_dir: str
    file_pattern: str = "*.nc"
    # Exactly one of these two time-derivation modes must be given (see
    # _validate_time_mode()): a ready-made valid_time STRING
    # attribute (e.g. WoFS's valid_time="20260518_230000"), or init+lead
    # arithmetic (e.g. MPAS's initializationTime+forecastHour).
    valid_time_attr: str | None = None
    valid_time_format: str | None = None
    init_attr: str | None = None
    lead_attr: str | None = None
    lead_units: str = "hours"
    init_format: str | None = None
    tolerance_minutes: float = 5.0
    s3_bucket: str = "noaa-mrms-pds"
    mrms_product: str = "MergedReflectivityQCComposite_00.50"
    mirror_subdirs: bool = True
    skip_existing: bool = True
    max_files: int | None = None


@dataclass
class HistogramObservationConfig:
    """Recipe for build_histogram_mrms.py: builds one reflectivity-
    distribution histogram file per YYYYMMDD day of already-interpolated
    MRMS. Self-contained (not reusing ObservationConfig), since most
    object-ID fields (thresholds/tracking) are irrelevant here -- mask is
    the one exception, reused as-is (none|conus|conus_east) since it
    describes the same spatial domain restriction either way."""
    interp_mrms_dir: str
    var_name: str = "refl_consv"
    lat_name: str = "lat"
    lon_name: str = "lon"
    output_dir: str = "output/hist_mrms"
    mask: str = "none"
    bin_min: float = -20.0
    bin_max: float = 80.0
    bin_width: float = 0.2
    edge_trim: int = 7
    clip_negative_to_zero: bool = False


@dataclass
class HistogramModelConfig:
    """Recipe for build_histogram_model.py: builds one reflectivity-
    distribution histogram file for one whole forecast (every lead time,
    every member if ensemble). Self-contained (not reusing ModelConfig), for
    the same reason as HistogramObservationConfig above."""
    input_dir: str
    var_name: str = "refl10cm_max"
    lat_name: str = "latitude"
    lon_name: str = "longitude"
    member_subdirs: bool = False
    member_subdir_pattern: str = "*"  # see ModelConfig's field of the same name
    stacked_members: bool = False
    file_pattern: str = "*.nc"
    init_attr: str | None = None
    lead_attr: str | None = None
    lead_units: str = "hours"
    init_format: str | None = None
    valid_time_attr: str | None = None
    valid_time_format: str | None = None
    # Only used to derive lead_hours in the valid_time_attr time-mode (the
    # init_attr/lead_attr mode already has a direct lead-time number to read
    # instead) -- the name of the file's own init-time string attribute,
    # read with the same valid_time_format (e.g. WoFS/NowcastNet/StormScope
    # all carry a same-format init_time="20260518_230000" alongside valid_time).
    init_time_attr: str = "init_time"
    output_dir: str = "output/hist_model"
    mask: str = "none"
    bin_min: float = -20.0
    bin_max: float = 80.0
    bin_width: float = 0.2
    edge_trim: int = 7
    clip_negative_to_zero: bool = False


@dataclass
class LinearClassificationConfig:
    linear_eccentricity_threshold: float
    linear_length_threshold_km: float
    mixed_eccentricity_threshold: float
    mixed_length_threshold_km: float


@dataclass
class Config:
    interpolation: InterpolationConfig | None = None
    observations: ObservationConfig | None = None
    model: ModelConfig | None = None
    matching: MatchingConfig | None = None
    linear_classification: LinearClassificationConfig | None = None
    fetch_mrms: FetchMrmsConfig | None = None
    histogram_observations: HistogramObservationConfig | None = None
    histogram_model: HistogramModelConfig | None = None


def require_section(section, section_name: str, config_path: str):
    """Raise a clear, named error if a driver's required section is absent.

    Every driver calls this immediately after load_config() for each section
    it actually needs -- e.g. `obs = require_section(cfg.observations,
    "observations", config_path)` -- rather than hand-rolling its own
    `if cfg.observations is None: raise ...`. A section being None just means
    it wasn't populated in this particular config file; that's only an error
    to a driver that actually needs it.
    """
    if section is None:
        raise ValueError(
            f"'{config_path}' has no '{section_name}:' section, but this driver requires one. "
            f"Add a '{section_name}:' section with its required fields "
            f"(see python_obj/configs/config.yaml for a populated example)."
        )
    return section


def _require_fields(section: dict, section_name: str, required: tuple[str, ...]) -> None:
    missing = [f for f in required if f not in section]
    if missing:
        raise ValueError(
            f"Config section '{section_name}' is missing required field(s): {', '.join(missing)}"
        )


def _check_allowed(section: dict, section_name: str, field: str, allowed: tuple[str, ...]) -> None:
    value = section.get(field)
    if value is not None and value not in allowed:
        raise ValueError(
            f"Config section '{section_name}' field '{field}'={value!r} is not one of {allowed}"
        )


def _build_source_config(
    cls, section: dict, section_name: str,
    extra_required: tuple[str, ...] = (),
    extra_optional_defaults: dict | None = None,
):
    extra_optional_defaults = extra_optional_defaults or {}
    _require_fields(section, section_name, _COMMON_REQUIRED + extra_required)
    _check_allowed(section, section_name, "mask", _VALID_MASKS)
    _check_allowed(section, section_name, "file_grouping", _VALID_FILE_GROUPINGS)
    if "lead_units" in section:
        _check_allowed(section, section_name, "lead_units", _VALID_LEAD_UNITS)

    kwargs = {k: section[k] for k in _COMMON_REQUIRED}
    for k in extra_required:
        kwargs[k] = section[k]
    for k, default in _COMMON_OPTIONAL_DEFAULTS.items():
        kwargs[k] = section.get(k, default)
    for k, default in extra_optional_defaults.items():
        kwargs[k] = section.get(k, default)
    if cls is ModelConfig:
        kwargs["lead_units"] = section.get("lead_units", "hours")

    return cls(**kwargs)


def _validate_time_mode(section: dict, section_name: str) -> None:
    """The 'at least one of two field-groups is present' check isn't
    expressible via the plain required-tuple mechanism _require_fields()
    already provides, so this gets its own small validator."""
    has_string_mode = "valid_time_attr" in section
    has_arith_mode = all(k in section for k in ("init_attr", "lead_attr", "init_format"))
    if not (has_string_mode or has_arith_mode):
        raise ValueError(
            f"Config section '{section_name}' needs either 'valid_time_attr'+'valid_time_format', "
            f"or 'init_attr'+'lead_attr'+'init_format'."
        )
    if ("valid_time_attr" in section) != ("valid_time_format" in section):
        raise ValueError(
            f"Config section '{section_name}': 'valid_time_attr' and 'valid_time_format' "
            f"must both be given, or neither."
        )


def _resolve_paths(obj, path_fields: tuple[str, ...], base_dir: str) -> None:
    for field_name in path_fields:
        value = getattr(obj, field_name)
        if value is not None and not os.path.isabs(value):
            setattr(obj, field_name, os.path.normpath(os.path.join(base_dir, value)))


def load_config(path: str) -> Config:
    """Parse a YAML config file into a Config. Every section is OPTIONAL --
    a section absent from the YAML yields None on the returned Config, not
    an error (drivers that need a section call require_section() themselves).
    A section that IS present is still validated for its own required fields,
    exactly as before -- "optional at the top level" and "validated once
    present" are independent behaviors. Raises a clear ValueError naming the
    missing section/field for a section that IS present but incomplete,
    rather than a generic KeyError or a silent guess.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"'{path}' did not parse into a mapping at the top level")

    base_dir = os.path.dirname(os.path.abspath(path))

    def _section(name: str) -> dict | None:
        if name not in raw:
            return None
        value = raw[name]
        if not isinstance(value, dict):
            raise ValueError(f"Config section '{name}' must be a mapping, got {type(value).__name__}")
        return value

    interpolation = None
    interp_section = _section("interpolation")
    if interp_section is not None:
        _require_fields(interp_section, "interpolation", ("raw_mrms_dir", "interp_mrms_dir", "target_grid_file"))
        date_range = interp_section.get("date_range")
        interpolation = InterpolationConfig(
            raw_mrms_dir=interp_section["raw_mrms_dir"],
            interp_mrms_dir=interp_section["interp_mrms_dir"],
            target_grid_file=interp_section["target_grid_file"],
            target_lat_name=interp_section.get("target_lat_name", "latitude"),
            target_lon_name=interp_section.get("target_lon_name", "longitude"),
            n_workers=interp_section.get("n_workers", 8),
            weight_cache_dir=interp_section.get("weight_cache_dir", "output/weight_cache"),
            date_range=tuple(date_range) if date_range is not None else None,
            max_files=interp_section.get("max_files"),
            file_pattern=interp_section.get("file_pattern", "**/*.grib2*"),
        )
        _resolve_paths(interpolation, _INTERPOLATION_PATH_FIELDS, base_dir)

    observations = None
    obs_section = _section("observations")
    if obs_section is not None:
        observations = _build_source_config(
            ObservationConfig, obs_section, "observations",
            extra_required=("interp_mrms_dir",),
            extra_optional_defaults={"object_output_dir": "output/obj_mrms"},
        )
        _resolve_paths(observations, _OBSERVATIONS_PATH_FIELDS, base_dir)

    model = None
    model_section = _section("model")
    if model_section is not None:
        _validate_time_mode(model_section, "model")
        model = _build_source_config(
            ModelConfig, model_section, "model",
            extra_required=("input_dir",),
            extra_optional_defaults={
                "object_output_dir": "output/obj_model",
                "member_subdirs": False,
                "member_subdir_pattern": "*",
                "stacked_members": False,
                "file_pattern": "*.nc",
                "init_attr": None,
                "lead_attr": None,
                "init_format": None,
                "valid_time_attr": None,
                "valid_time_format": None,
            },
        )
        _resolve_paths(model, _MODEL_PATH_FIELDS, base_dir)

    matching = None
    matching_section = _section("matching")
    if matching_section is not None:
        _require_fields(
            matching_section, "matching",
            ("max_boundary_disp_km", "max_centroid_disp_km", "ti_threshold",
             "truth_object_dir", "forecast_object_dir"),
        )
        matching = MatchingConfig(
            max_boundary_disp_km=matching_section["max_boundary_disp_km"],
            max_centroid_disp_km=matching_section["max_centroid_disp_km"],
            ti_threshold=matching_section["ti_threshold"],
            truth_object_dir=matching_section["truth_object_dir"],
            forecast_object_dir=matching_section["forecast_object_dir"],
            max_time_offset_minutes=matching_section.get("max_time_offset_minutes", 5.0),
            output_dir=matching_section.get("output_dir", "output/matches"),
            file_pattern=matching_section.get("file_pattern", "*.nc"),
        )
        _resolve_paths(matching, _MATCHING_PATH_FIELDS, base_dir)

    linear_classification = None
    linear_section = _section("linear_classification")
    if linear_section is not None:
        _require_fields(
            linear_section, "linear_classification",
            ("linear_eccentricity_threshold", "linear_length_threshold_km",
             "mixed_eccentricity_threshold", "mixed_length_threshold_km"),
        )
        linear_classification = LinearClassificationConfig(
            linear_eccentricity_threshold=linear_section["linear_eccentricity_threshold"],
            linear_length_threshold_km=linear_section["linear_length_threshold_km"],
            mixed_eccentricity_threshold=linear_section["mixed_eccentricity_threshold"],
            mixed_length_threshold_km=linear_section["mixed_length_threshold_km"],
        )

    fetch_mrms = None
    fetch_mrms_section = _section("fetch_mrms")
    if fetch_mrms_section is not None:
        _require_fields(fetch_mrms_section, "fetch_mrms", ("model_input_dir", "output_dir"))
        _validate_time_mode(fetch_mrms_section, "fetch_mrms")
        fetch_mrms = FetchMrmsConfig(
            model_input_dir=fetch_mrms_section["model_input_dir"],
            output_dir=fetch_mrms_section["output_dir"],
            file_pattern=fetch_mrms_section.get("file_pattern", "*.nc"),
            valid_time_attr=fetch_mrms_section.get("valid_time_attr"),
            valid_time_format=fetch_mrms_section.get("valid_time_format"),
            init_attr=fetch_mrms_section.get("init_attr"),
            lead_attr=fetch_mrms_section.get("lead_attr"),
            lead_units=fetch_mrms_section.get("lead_units", "hours"),
            init_format=fetch_mrms_section.get("init_format"),
            tolerance_minutes=fetch_mrms_section.get("tolerance_minutes", 5.0),
            s3_bucket=fetch_mrms_section.get("s3_bucket", "noaa-mrms-pds"),
            mrms_product=fetch_mrms_section.get("mrms_product", "MergedReflectivityQCComposite_00.50"),
            mirror_subdirs=fetch_mrms_section.get("mirror_subdirs", True),
            skip_existing=fetch_mrms_section.get("skip_existing", True),
            max_files=fetch_mrms_section.get("max_files"),
        )
        _resolve_paths(fetch_mrms, _FETCH_MRMS_PATH_FIELDS, base_dir)

    histogram_observations = None
    hist_obs_section = _section("histogram_observations")
    if hist_obs_section is not None:
        _require_fields(hist_obs_section, "histogram_observations", ("interp_mrms_dir",))
        _check_allowed(hist_obs_section, "histogram_observations", "mask", _VALID_MASKS)
        histogram_observations = HistogramObservationConfig(
            interp_mrms_dir=hist_obs_section["interp_mrms_dir"],
            var_name=hist_obs_section.get("var_name", "refl_consv"),
            lat_name=hist_obs_section.get("lat_name", "lat"),
            lon_name=hist_obs_section.get("lon_name", "lon"),
            output_dir=hist_obs_section.get("output_dir", "output/hist_mrms"),
            mask=hist_obs_section.get("mask", "none"),
            bin_min=hist_obs_section.get("bin_min", -20.0),
            bin_max=hist_obs_section.get("bin_max", 80.0),
            bin_width=hist_obs_section.get("bin_width", 0.2),
            edge_trim=hist_obs_section.get("edge_trim", 7),
            clip_negative_to_zero=hist_obs_section.get("clip_negative_to_zero", False),
        )
        _resolve_paths(histogram_observations, _HISTOGRAM_OBSERVATIONS_PATH_FIELDS, base_dir)

    histogram_model = None
    hist_model_section = _section("histogram_model")
    if hist_model_section is not None:
        _require_fields(hist_model_section, "histogram_model", ("input_dir",))
        _validate_time_mode(hist_model_section, "histogram_model")
        _check_allowed(hist_model_section, "histogram_model", "mask", _VALID_MASKS)
        histogram_model = HistogramModelConfig(
            input_dir=hist_model_section["input_dir"],
            var_name=hist_model_section.get("var_name", "refl10cm_max"),
            lat_name=hist_model_section.get("lat_name", "latitude"),
            lon_name=hist_model_section.get("lon_name", "longitude"),
            member_subdirs=hist_model_section.get("member_subdirs", False),
            member_subdir_pattern=hist_model_section.get("member_subdir_pattern", "*"),
            stacked_members=hist_model_section.get("stacked_members", False),
            file_pattern=hist_model_section.get("file_pattern", "*.nc"),
            init_attr=hist_model_section.get("init_attr"),
            lead_attr=hist_model_section.get("lead_attr"),
            lead_units=hist_model_section.get("lead_units", "hours"),
            init_format=hist_model_section.get("init_format"),
            valid_time_attr=hist_model_section.get("valid_time_attr"),
            valid_time_format=hist_model_section.get("valid_time_format"),
            init_time_attr=hist_model_section.get("init_time_attr", "init_time"),
            output_dir=hist_model_section.get("output_dir", "output/hist_model"),
            mask=hist_model_section.get("mask", "none"),
            bin_min=hist_model_section.get("bin_min", -20.0),
            bin_max=hist_model_section.get("bin_max", 80.0),
            bin_width=hist_model_section.get("bin_width", 0.2),
            edge_trim=hist_model_section.get("edge_trim", 7),
            clip_negative_to_zero=hist_model_section.get("clip_negative_to_zero", False),
        )
        _resolve_paths(histogram_model, _HISTOGRAM_MODEL_PATH_FIELDS, base_dir)

    return Config(
        interpolation=interpolation, observations=observations, model=model,
        matching=matching, linear_classification=linear_classification,
        fetch_mrms=fetch_mrms,
        histogram_observations=histogram_observations, histogram_model=histogram_model,
    )
