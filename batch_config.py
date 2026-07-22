"""Expand ONE template config file into many per-case config files, so a
batch of hundreds of date-named case directories never requires hand-writing
hundreds of near-identical YAML files.

A template is an ordinary config.py-schema YAML file (see configs/config.yaml)
with one addition: a top-level `cases:` section (dates/date_range +
date_format), and a literal "{date}" placeholder inside whichever path fields
vary per case (e.g. `input_dir: /glade/.../{date}00/post`). expand_batch_config()
substitutes "{date}" with each case's date string and writes one materialized
config file per case, ready to feed straight into
python_obj.batch_runner.run_cases_in_parallel -- run_one_case()/load_config()
themselves need no changes at all, since a materialized file is just a
completely ordinary config.

`cases:` schema:
    cases:
      dates: ["20230501", "20230503", "20230511"]   # explicit, non-contiguous
      # -- OR --
      date_range: ["20230501", "20230510"]           # inclusive, daily cadence
      date_format: "%Y%m%d"                          # default; must match both
                                                      # the {date} placeholders'
                                                      # format AND date_range's
                                                      # own string format

Two independent "this case can't be run" outcomes are distinguished, not
collapsed into one: a date whose case-defining directory doesn't exist at all
(e.g. the forecast was never initialized) vs. one whose directory exists but
is empty (e.g. the forecast crashed after creating its output directory but
before writing anything). Both are reported, never silently dropped, and
neither one stops the remaining dates from being processed.
"""

import os
from dataclasses import dataclass, fields, replace

import yaml

from python_obj.config import Config, load_config

# Per-section attribute name (on Config) -> the field on that section's own
# dataclass that names the one directory whose presence/non-emptiness defines
# whether this date's case can be run at all. Sections with no such single
# "read a directory of case files" concept (matching, linear_classification)
# are deliberately absent -- nothing to check for them.
_CASE_DIR_FIELDS_BY_SECTION = {
    "interpolation": "raw_mrms_dir",
    "observations": "interp_mrms_dir",
    "model": "input_dir",
    "fetch_mrms": "model_input_dir",
    "histogram_observations": "interp_mrms_dir",
    "histogram_model": "input_dir",
}

_DEFAULT_DATE_FORMAT = "%Y%m%d"


@dataclass
class ExpandedBatchConfig:
    case_paths: list  # list[str] -- materialized config paths, ready to run
    skipped_no_directory: list  # list[str] -- dates whose case directory doesn't exist at all
    skipped_no_files: list  # list[str] -- dates whose case directory exists but contains no files


def _parse_cases_section(template_path: str) -> tuple[list, str]:
    """Reads just the 'cases:' section by hand (load_config() has no concept
    of it -- it's meta-information about how to expand the template, not
    itself a pipeline-stage recipe)."""
    with open(template_path) as f:
        raw = yaml.safe_load(f) or {}

    cases = raw.get("cases")
    if cases is None:
        raise ValueError(
            f"'{template_path}' has no top-level 'cases:' section -- expand_batch_config() "
            f"needs one with either 'dates' or 'date_range' (see batch_config.py's module docstring)."
        )

    has_dates = "dates" in cases
    has_range = "date_range" in cases
    if has_dates == has_range:  # both or neither
        raise ValueError(
            f"'{template_path}' 'cases:' section needs exactly one of 'dates' or 'date_range', "
            f"not {'both' if has_dates else 'neither'}."
        )

    date_format = cases.get("date_format", _DEFAULT_DATE_FORMAT)

    if has_dates:
        return list(cases["dates"]), date_format

    from datetime import datetime, timedelta

    start_str, end_str = cases["date_range"]
    start = datetime.strptime(start_str, date_format)
    end = datetime.strptime(end_str, date_format)
    if end < start:
        raise ValueError(f"'{template_path}' 'cases.date_range': end ({end_str}) is before start ({start_str})")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime(date_format))
        current += timedelta(days=1)
    return dates, date_format


def _substitute_date(section_instance, date_str: str):
    """Returns a copy of this dataclass instance with '{date}' replaced by
    date_str in every string field -- not just the ones this module happens
    to know are path-shaped, since a case-varying value (e.g. output_dir)
    isn't limited to the small _CASE_DIR_FIELDS_BY_SECTION list above."""
    changes = {}
    for f in fields(section_instance):
        value = getattr(section_instance, f.name)
        if isinstance(value, str) and "{date}" in value:
            changes[f.name] = value.replace("{date}", date_str)
    return replace(section_instance, **changes) if changes else section_instance


def _dir_has_any_file(directory: str) -> bool:
    """True if `directory` contains at least one regular file anywhere below
    it. Walks with an early exit on the first file found -- never lists an
    entire large directory tree just to answer a yes/no question."""
    for _, _, filenames in os.walk(directory):
        if filenames:
            return True
    return False


def _to_yaml_safe(value):
    """Recursively converts tuples (e.g. InterpolationConfig.date_range) to
    lists -- PyYAML's safe dumper has no representer for a plain tuple."""
    if isinstance(value, tuple):
        return [_to_yaml_safe(v) for v in value]
    if isinstance(value, list):
        return [_to_yaml_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_yaml_safe(v) for k, v in value.items()}
    return value


def expand_batch_config(template_path: str, output_dir: str) -> ExpandedBatchConfig:
    """Expands one template config (see module docstring) into one
    materialized config file per case date, skipping (and reporting, never
    silently dropping) dates whose case directory is missing or empty.
    """
    dates, _ = _parse_cases_section(template_path)
    cfg: Config = load_config(template_path)  # fully parsed + path-resolved (relative to template_path's own dir)

    os.makedirs(output_dir, exist_ok=True)

    case_paths = []
    skipped_no_directory = []
    skipped_no_files = []

    for date_str in dates:
        substituted_sections = {}
        for f in fields(Config):
            section = getattr(cfg, f.name)
            substituted_sections[f.name] = None if section is None else _substitute_date(section, date_str)

        missing_dir = False
        empty_dir = False
        for section_name, dir_field in _CASE_DIR_FIELDS_BY_SECTION.items():
            section = substituted_sections.get(section_name)
            if section is None:
                continue
            directory = getattr(section, dir_field)
            if not os.path.isdir(directory):
                missing_dir = True
            elif not _dir_has_any_file(directory):
                empty_dir = True

        if missing_dir:
            skipped_no_directory.append(date_str)
            continue
        if empty_dir:
            print(f"WARNING: expand_batch_config: date '{date_str}' has an existing case directory with no files -- skipping")
            skipped_no_files.append(date_str)
            continue

        out_yaml = {
            name: _to_yaml_safe({f.name: getattr(section, f.name) for f in fields(section)})
            for name, section in substituted_sections.items() if section is not None
        }
        out_path = os.path.join(output_dir, f"config_{date_str}.yaml")
        with open(out_path, "w") as fh:
            yaml.safe_dump(out_yaml, fh, sort_keys=False)
        case_paths.append(out_path)

    if skipped_no_directory:
        print(f"expand_batch_config: skipped {len(skipped_no_directory)} date(s), case directory not found: {skipped_no_directory}")
    if skipped_no_files:
        print(f"expand_batch_config: skipped {len(skipped_no_files)} date(s), case directory exists but has no files: {skipped_no_files}")

    return ExpandedBatchConfig(
        case_paths=case_paths, skipped_no_directory=skipped_no_directory, skipped_no_files=skipped_no_files,
    )
