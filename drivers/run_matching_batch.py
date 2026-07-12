"""Standalone script: run run_matching.py's run_one_case() across many config
files in parallel, e.g. one config per case/day.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/run_matching_batch.py

Edit CASE_CONFIGS below to point at your own list of per-case config.yaml
paths -- this script never discovers cases on its own (see
python_obj.batch_runner's module docstring for why).
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.batch_runner import run_cases_in_parallel
from python_obj.drivers.run_matching import run_one_case

CASE_CONFIGS = [
    # os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml"),
    # os.path.join(os.path.dirname(_THIS_DIR), "configs", "config_case2.yaml"),
]

N_WORKERS = 4


if __name__ == "__main__":
    if not CASE_CONFIGS:
        raise ValueError("CASE_CONFIGS is empty -- edit this script to list your per-case config.yaml paths")
    run_cases_in_parallel(CASE_CONFIGS, run_one_case, n_workers=N_WORKERS)
