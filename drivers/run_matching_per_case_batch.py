"""Standalone script: run run_matching_per_case.py's run_one_case() across
many forecast case files in parallel via python_obj.batch_runner.

IMPORTANT (read before raising N_WORKERS): each case's peak memory is
dominated by decompressing one member's full label block at a time
(measured ~2 GB for a real 5-member x 133-lead-time x 1059x1799 case) plus
its own filtered truth-file subset (~1 GB for a ~133-hour window) -- so
N_WORKERS concurrent cases roughly multiply that peak. On a memory-
constrained machine, prefer N_WORKERS=1 (sequential -- equivalent to just
running run_matching_per_case.py directly) unless you have confirmed your
machine has enough RAM for N cases' peak memory at once.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/run_matching_per_case_batch.py

Edit CONFIG_PATH/N_WORKERS below.
"""

import functools
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.batch_runner import run_cases_in_parallel
from python_obj.config import load_config, require_section
from python_obj.drivers.run_matching_per_case import discover_forecast_cases, run_one_case

CONFIG_PATH = os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
N_WORKERS = 1  # see module docstring -- raise only after confirming available RAM


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    match = require_section(cfg.matching, "matching", CONFIG_PATH)
    case_files = discover_forecast_cases(match.forecast_object_dir, match.file_pattern)
    run_cases_in_parallel(case_files, functools.partial(run_one_case, CONFIG_PATH), n_workers=N_WORKERS)
