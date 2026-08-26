"""Standalone script: run run_matching_per_case.py's run_one_case() across
many forecast case files in parallel via python_obj.batch_runner.

IMPORTANT (read before raising n_workers): each case's peak memory is
dominated by decompressing one member's full label block at a time
(measured ~2 GB for a real 5-member x 133-lead-time x 1059x1799 case, and
864 MB for a real 18-member x 73-lead-time x 300x300 WoFS case) plus its own
filtered truth-file subset -- so n_workers concurrent cases roughly multiply
that peak. On a memory-constrained machine, prefer n_workers=1 (sequential --
equivalent to just running run_matching_per_case.py directly) unless you have
confirmed your machine has enough RAM for N cases' peak memory at once.

skip_existing (off by default, matching run_matching_per_case.py's own
run_all_cases_sequential): filters out any case whose expected
match_init_<init_time>.nc output already exists BEFORE dispatching to the
worker pool -- lets an interrupted batch run (e.g. killed by an external time
limit partway through hundreds of cases) resume without reprocessing already-
completed cases, combined with real parallelism (which
run_all_cases_sequential's own skip_existing cannot provide, since it is
single-process).

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/run_matching_per_case_batch.py \\
      [config_path] [skip_existing: true|false] [n_workers]

All three arguments are optional; defaults are config_path=python_obj/configs/config.yaml,
skip_existing=false, n_workers=1.
"""

import functools
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.batch_runner import run_cases_in_parallel
from python_obj.config import load_config, require_section
from python_obj.drivers.run_matching_per_case import discover_forecast_cases, expected_output_path, run_one_case

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml")
DEFAULT_N_WORKERS = 1  # see module docstring -- raise only after confirming available RAM


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    skip_existing = sys.argv[2].strip().lower() == "true" if len(sys.argv) > 2 else False
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_N_WORKERS

    cfg = load_config(config_path)
    match = require_section(cfg.matching, "matching", config_path)
    case_files = discover_forecast_cases(match.forecast_object_dir, match.file_pattern)
    print(f"Found {len(case_files)} forecast case files under '{match.forecast_object_dir}'")

    if skip_existing:
        before = len(case_files)
        case_files = [f for f in case_files if not os.path.exists(expected_output_path(f, match.output_dir))]
        print(f"skip_existing=True: {before - len(case_files)} case(s) already have output, "
              f"{len(case_files)} remaining to process")

    if case_files:
        run_cases_in_parallel(case_files, functools.partial(run_one_case, config_path), n_workers=n_workers)
    else:
        print("Nothing left to process.")
