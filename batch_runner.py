"""Generic, domain-agnostic parallel case runner.

Every driver in python_obj/drivers/ needs to run its own single-case entry
point across a caller-supplied list of cases in parallel, collecting a
success/failure report per case -- this is exactly the same shape of problem
as run_batch_interpolation's file-level Pool, just one level up (case-level
instead of file-level), and just as generic across drivers, so it lives here
rather than being reimplemented per driver.

Generalizes the legacy python_base/wofs_obj_match_wrapper_*.py pattern
(multiprocessing.Pool.apply_async over a hardcoded case_ids list, one
subprocess per case) into an in-process Pool.map over a plain callable, with
no hardcoded paths/case lists baked in here -- case_specs is always supplied
by the caller (e.g. a list of per-case config file paths), never discovered
via a directory-naming convention, consistent with this library's existing
manifest-building principle.

IMPORTANT: case_fn must be a module-level, picklable callable (not a lambda
or closure) -- multiprocessing.Pool on macOS/most modern platforms uses the
`spawn` start method, which re-imports the calling script in every worker.
Any script driving run_cases_in_parallel must guard its top-level call with
`if __name__ == "__main__":`, exactly like run_batch_interpolation
(python_obj/regrid/batch_interpolate.py) already requires -- calling it from
a REPL or `python -c "..."` will fail with a pickling/import error, not a
silent hang.
"""

from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Callable


@dataclass
class CaseResult:
    case: object  # the caller's own case spec, e.g. a config file path
    success: bool
    result: object | None = None  # case_fn's return value on success, else None
    error: str | None = None  # str(exception) on failure, else None


@dataclass
class BatchCaseSummary:
    n_total: int
    n_success: int
    n_failed: int
    results: list = field(default_factory=list)  # list[CaseResult]

    def __str__(self) -> str:
        lines = [f"BatchCaseSummary: {self.n_success}/{self.n_total} succeeded, {self.n_failed} failed"]
        for r in self.results:
            if not r.success:
                lines.append(f"  FAILED: {r.case}: {r.error}")
        return "\n".join(lines)


def _run_one_case(args: tuple) -> CaseResult:
    case_fn, case = args
    try:
        return CaseResult(case=case, success=True, result=case_fn(case))
    except Exception as exc:  # noqa: BLE001 -- deliberate: collect per-case failures, don't crash the pool
        return CaseResult(case=case, success=False, error=f"{type(exc).__name__}: {exc}")


def run_cases_in_parallel(
    case_specs: list,
    case_fn: Callable[[object], object],
    n_workers: int = 4,
) -> BatchCaseSummary:
    """Run case_fn(spec) for every spec in case_specs via an in-process
    multiprocessing.Pool.map, catching each case's own exception rather than
    letting it kill the pool -- one bad case is reported in the returned
    summary, never silently drops or blocks the rest.

    case_specs: a plain list the caller builds themselves (e.g. per-case
    config file paths) -- this function never discovers cases on its own.
    case_fn: a module-level, picklable callable (see module docstring for the
    `spawn`/`if __name__ == "__main__":` requirement).
    """
    with Pool(processes=n_workers) as pool:
        results = pool.map(_run_one_case, [(case_fn, c) for c in case_specs])

    n_success = sum(1 for r in results if r.success)
    summary = BatchCaseSummary(
        n_total=len(results),
        n_success=n_success,
        n_failed=len(results) - n_success,
        results=results,
    )
    print(summary)
    return summary
