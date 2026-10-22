"""Builds py_obj/sample_data/grib2_case/ from real HRRR and RRFS GRIB2 forecasts
(this developer's local grib2_testing/{hrrr,rrfs}/, 2026-05-18 00Z run).

The source files are ~115 MB each (105-173 messages). Each output file keeps
only two real messages, byte-for-byte (pygrib tostring(), confirmed to
round-trip bit-identically):
  1. a DECOY: another reflectivity record from the same file
     (HRRR 0/16/195 at 1 km AGL; RRFS 0/16/4 at 1 km AGL), written FIRST so any
     code that just takes message 1 reads the wrong field and tests catch it;
  2. the composite reflectivity (REFC) record, in each model's own encoding:
     HRRR 0/16/196 (NCEP local table), RRFS 0/16/5 (WMO standard).
Lead times f000-f002; output filenames are identical to the sources'.
Total ~6.5 MB.

Idempotent -- safe to re-run; overwrites its own output. A provenance/
regeneration record, not part of the library's own runtime.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python py_obj/sample_data/build_grib2_sample.py
"""

import os

import pygrib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
SOURCE_DIR = os.path.join(REPO_ROOT, "grib2_testing")
OUT_DIR = os.path.join(_THIS_DIR, "grib2_case")
LEADS = ("000", "001", "002")

MODELS = {
    "hrrr": {
        "pattern": "hrrr_ncep_2026051800f{lead}.grib2",
        "decoy": {"discipline": 0, "parameterCategory": 16, "parameterNumber": 195, "typeOfLevel": "heightAboveGround", "level": 1000},
        "refc": {"discipline": 0, "parameterCategory": 16, "parameterNumber": 196},
    },
    "rrfs": {
        "pattern": "rrfs_2026051800f{lead}.grib2",
        "decoy": {"discipline": 0, "parameterCategory": 16, "parameterNumber": 4, "typeOfLevel": "heightAboveGround", "level": 1000},
        "refc": {"discipline": 0, "parameterCategory": 16, "parameterNumber": 5},
    },
}


def _one(grbs, keys, path):
    grbs.rewind()
    found = grbs.select(**keys)
    if len(found) != 1:
        raise ValueError(f"'{path}': expected exactly 1 message for {keys}, found {len(found)}")
    return found[0]


def main():
    for model, spec in MODELS.items():
        out_dir = os.path.join(OUT_DIR, model)
        os.makedirs(out_dir, exist_ok=True)
        for lead in LEADS:
            name = spec["pattern"].format(lead=lead)
            src = os.path.join(SOURCE_DIR, model, name)
            grbs = pygrib.open(src)
            try:
                payload = _one(grbs, spec["decoy"], src).tostring() + _one(grbs, spec["refc"], src).tostring()
            finally:
                grbs.close()
            dst = os.path.join(out_dir, name)
            with open(dst, "wb") as fh:
                fh.write(payload)
            print(f"wrote {dst} ({len(payload) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
