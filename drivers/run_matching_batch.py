"""Standalone script: run run_matching.py's run_one_case() across many config
files in parallel, e.g. one config per case/day.

Two ways to supply cases, both optional via the command line (falls back to
the hardcoded CASE_CONFIGS list below when no template_path argument is
given):
  1. A template config (with a "cases:" section + "{date}"/"{init_time}"
     placeholders) -- expanded into one materialized config per case via
     python_obj.batch_config.expand_batch_config(), run, then the
     materialized per-case config files are deleted again (NOT the whole
     output directory -- only the specific files this run created) once
     every case has finished, so a later run's temp configs never collide
     with a previous run's leftovers.
  2. A literal list of already-existing per-case config paths -- edit
     CASE_CONFIGS below; this script never discovers cases on its own (see
     python_obj.batch_runner's module docstring for why).

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python python_obj/drivers/run_matching_batch.py \\
      [template_path] [n_workers]

Both arguments are optional. With no template_path given, falls back to the
hardcoded CASE_CONFIGS list below (empty by default -- edit it to use mode 2
without any command-line arguments).
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.batch_config import expand_batch_config
from python_obj.batch_runner import run_cases_in_parallel
from python_obj.drivers.run_matching import run_one_case

# Fallback list used only when no template_path is given on the command line.
CASE_CONFIGS = [
    # os.path.join(os.path.dirname(_THIS_DIR), "configs", "config.yaml"),
    # os.path.join(os.path.dirname(_THIS_DIR), "configs", "config_case2.yaml"),
]

DEFAULT_N_WORKERS = 4


if __name__ == "__main__":
    template_path = sys.argv[1] if len(sys.argv) > 1 else None
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_WORKERS

    expanded = None
    if template_path:
        expanded = expand_batch_config(
            template_path=template_path,
            output_dir=os.path.join(os.path.dirname(_THIS_DIR), "configs", "output", "_batch_configs"),
        )
        case_configs = expanded.case_paths
        n_skipped = len(expanded.skipped_no_directory) + len(expanded.skipped_no_files)
        print(f"Expanded template '{template_path}' into {len(case_configs)} per-case config(s) "
              f"({n_skipped} case(s) skipped -- see expanded.skipped_no_directory/skipped_no_files for reasons)")
    else:
        case_configs = CASE_CONFIGS
        if not case_configs:
            raise ValueError(
                "No template_path given on the command line and CASE_CONFIGS is empty -- "
                "either pass a template config path as the first argument, or edit this "
                "script to list your own per-case config.yaml paths in CASE_CONFIGS."
            )

    try:
        run_cases_in_parallel(case_configs, run_one_case, n_workers=n_workers)
    finally:
        if expanded is not None:
            for p in expanded.case_paths:
                if os.path.exists(p):
                    os.remove(p)
            print(f"Cleaned up {len(expanded.case_paths)} materialized temp config file(s) under "
                  f"'{os.path.dirname(expanded.case_paths[0]) if expanded.case_paths else '(none written)'}'")
