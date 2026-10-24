"""Shared check for skip_existing: does an output file already exist in usable form?

A file counts as existing only if it is present AND opens as NetCDF. A truncated
file -- e.g. from a job killed mid-write -- fails to open and so counts as missing;
the step then recomputes it and the writer overwrites it. (This catches truncation,
not a file that opens but is incomplete.)
"""

import os


def readable_output(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    import netCDF4  # local import: keeps importing this module (and config.py's callers) light

    try:
        netCDF4.Dataset(path, "r").close()
        return True
    except OSError:
        return False
