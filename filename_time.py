"""Derive a forecast file's init/valid time from its FILENAME rather than its
metadata -- an opt-in alternative for sources whose internal time metadata
is known to be unreliable (e.g. a WoFSCast file whose global valid_time
attribute was a stale copy of init_time), while the filename's own
timestamps are correct.

Template syntax (matched against the file's basename, whole-string):
  - `{init:<fmt>}` / `{valid:<fmt>}`: a timestamp slot, <fmt> built from
    ordinary strftime codes (%Y %m %d %H %M %S %j, plus literal separators).
  - `*`: wildcard for any run of characters (prefix, member/lead index, ...).
  - anything else: literal text that must match exactly.

WoFS example: "wofs_*_*_{init:%Y%m%d_%H%M}_{valid:%H%M}.nc" matches
"wofs_ALL_08_20260518_2300_0140.nc" -> init 2026-05-18 23:00, valid
2026-05-19 01:40.

Day rollover: when the valid slot carries no date codes (only a clock time,
as in WoFS), its date is taken from the RAW filename init time, plus one day
if the valid clock time is earlier than the init clock time. This assumes
forecasts shorter than 24 h (true for WoFS/WoFSCast/NowcastNet). A valid
slot with its own full date is used as-is.

init_time_offset_minutes: an integer number of minutes SUBTRACTED from the
filename init time, for a system initialized from a parent forecast some
minutes after that parent's own init (e.g. WoFSCast's 2220 file comes from
the 2200 WoFS run). Applied AFTER the day rollover above, and only to init
time -- valid time is never shifted.
"""

import os
import re
from datetime import datetime, timedelta
from functools import lru_cache

_CODE_PATTERNS = {"Y": r"\d{4}", "m": r"\d{2}", "d": r"\d{2}", "H": r"\d{2}", "M": r"\d{2}", "S": r"\d{2}", "j": r"\d{3}"}
_DATE_CODES = {"Y", "m", "d", "j"}
_SLOT_RE = re.compile(r"\{(init|valid):([^{}]+)\}")


def _format_to_regex(fmt: str, template: str) -> tuple[str, set]:
    """Converts one slot's strftime format to a regex fragment; returns the
    fragment plus the set of codes used."""
    parts, codes, i = [], set(), 0
    while i < len(fmt):
        if fmt[i] == "%":
            if i + 1 >= len(fmt) or fmt[i + 1] not in _CODE_PATTERNS:
                raise ValueError(
                    f"filename_time_template {template!r}: unsupported strftime code "
                    f"{fmt[i:i + 2]!r} (supported: {', '.join('%' + c for c in _CODE_PATTERNS)})"
                )
            parts.append(_CODE_PATTERNS[fmt[i + 1]])
            codes.add(fmt[i + 1])
            i += 2
        else:
            parts.append(re.escape(fmt[i]))
            i += 1
    return "".join(parts), codes


@lru_cache(maxsize=None)
def compile_filename_time_template(template: str) -> tuple[re.Pattern, dict]:
    """Compiles a template (see module docstring) into (anchored regex,
    {slot_name: (strftime_fmt, has_date_codes)}). Raises ValueError for a
    malformed template -- called at load_config() time too, so a bad
    template fails before any run starts."""
    slots: dict = {}
    parts, pos = [], 0
    for m in _SLOT_RE.finditer(template):
        literal = template[pos:m.start()]
        if "{" in literal or "}" in literal:
            raise ValueError(f"filename_time_template {template!r}: malformed slot near {literal!r} "
                             "(expected '{init:<fmt>}' or '{valid:<fmt>}')")
        parts.append(".*?".join(re.escape(chunk) for chunk in literal.split("*")))
        name, fmt = m.group(1), m.group(2)
        if name in slots:
            raise ValueError(f"filename_time_template {template!r}: slot '{name}' given more than once")
        frag, codes = _format_to_regex(fmt, template)
        slots[name] = (fmt, bool(codes & _DATE_CODES))
        parts.append(f"(?P<{name}>{frag})")
        pos = m.end()
    tail = template[pos:]
    if "{" in tail or "}" in tail:
        raise ValueError(f"filename_time_template {template!r}: malformed slot near {tail!r} "
                         "(expected '{init:<fmt>}' or '{valid:<fmt>}')")
    parts.append(".*?".join(re.escape(chunk) for chunk in tail.split("*")))

    if not slots:
        raise ValueError(f"filename_time_template {template!r} has no '{{init:...}}' or '{{valid:...}}' slot")
    if "valid" in slots and not slots["valid"][1] and "init" not in slots:
        raise ValueError(
            f"filename_time_template {template!r}: the valid slot has no date codes, so its date must "
            "come from an init slot, but the template has none"
        )
    if "init" in slots and not slots["init"][1]:
        raise ValueError(f"filename_time_template {template!r}: the init slot must include date codes (e.g. %Y%m%d)")
    return re.compile("".join(parts) + r"\Z"), slots


def parse_filename_times(
    filepath: str,
    template: str,
    init_time_offset_minutes: int | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Returns (init_time, valid_time) parsed from filepath's basename via
    `template`; either is None only if the template has no slot for it.
    Raises ValueError (naming the file and template) if the basename doesn't
    match -- never falls back to the file's metadata."""
    pattern, slots = compile_filename_time_template(template)
    basename = os.path.basename(filepath)
    m = pattern.match(basename)
    if m is None:
        raise ValueError(f"Filename '{basename}' does not match filename_time_template {template!r}")

    parsed = {}
    for name, (fmt, _) in slots.items():
        try:
            parsed[name] = datetime.strptime(m.group(name), fmt)
        except ValueError as exc:
            raise ValueError(
                f"Filename '{basename}': {name} text {m.group(name)!r} is not a valid time for format {fmt!r} "
                f"(filename_time_template {template!r})"
            ) from exc

    raw_init = parsed.get("init")
    valid = parsed.get("valid")
    if valid is not None and not slots["valid"][1]:
        valid = datetime.combine(raw_init.date(), valid.time())
        if valid.time() < raw_init.time():
            valid += timedelta(days=1)

    init = raw_init
    if init is not None and init_time_offset_minutes:
        init = raw_init - timedelta(minutes=init_time_offset_minutes)
    return init, valid


def validate_filename_time_options(template: str | None, init_time_offset_minutes, context: str) -> None:
    """Config-time checks shared by every section that accepts these fields."""
    if template is not None:
        if not isinstance(template, str):
            raise ValueError(f"{context}: filename_time_template must be a string, got {type(template).__name__}")
        compile_filename_time_template(template)
    if init_time_offset_minutes is not None:
        if template is None:
            raise ValueError(f"{context}: init_time_offset_minutes is only used together with filename_time_template")
        if isinstance(init_time_offset_minutes, bool) or not isinstance(init_time_offset_minutes, int):
            raise ValueError(
                f"{context}: init_time_offset_minutes must be an integer number of minutes, "
                f"got {init_time_offset_minutes!r}"
            )
