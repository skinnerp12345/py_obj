"""Step 6c validation: generic parallel case runner.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_batch_runner.py -v -s

Uses only synthetic, module-level case functions (multiprocessing's spawn
start method requires case_fn to be picklable, i.e. importable by name --
a local closure/lambda would fail to pickle).
"""

import time

from python_obj.batch_runner import run_cases_in_parallel


def _double(x: int) -> int:
    return x * 2


def _fail_on_even(x: int) -> int:
    if x % 2 == 0:
        raise ValueError(f"deliberate failure for even case {x}")
    return x * 10


def _sleep_one_second(x: int) -> int:
    time.sleep(1.0)
    return x


def test_all_success():
    summary = run_cases_in_parallel([1, 2, 3, 4], _double, n_workers=2)
    assert summary.n_total == 4
    assert summary.n_success == 4
    assert summary.n_failed == 0
    results_by_case = {r.case: r.result for r in summary.results}
    assert results_by_case == {1: 2, 2: 4, 3: 6, 4: 8}
    print(f"\n[batch-runner-check1] {summary}")


def test_mixed_success_and_failure_does_not_stop_other_cases():
    case_specs = [1, 2, 3, 4, 5]
    summary = run_cases_in_parallel(case_specs, _fail_on_even, n_workers=2)
    assert summary.n_total == 5
    assert summary.n_success == 3  # 1, 3, 5
    assert summary.n_failed == 2  # 2, 4

    by_case = {r.case: r for r in summary.results}
    assert by_case[1].success and by_case[1].result == 10
    assert by_case[3].success and by_case[3].result == 30
    assert by_case[5].success and by_case[5].result == 50
    assert not by_case[2].success
    assert "deliberate failure for even case 2" in by_case[2].error
    assert not by_case[4].success
    assert "deliberate failure for even case 4" in by_case[4].error
    print(f"\n[batch-runner-check2] {summary}")


def test_real_parallelism_speedup():
    n_cases = 8
    n_workers = 4
    start = time.time()
    summary = run_cases_in_parallel(list(range(n_cases)), _sleep_one_second, n_workers=n_workers)
    elapsed = time.time() - start

    assert summary.n_success == n_cases
    # serial would take ~8s; with 4 workers should take ~2s -- generous bound to avoid flakiness
    assert elapsed < 6.0, f"expected meaningful speedup from parallelism, took {elapsed:.2f}s"
    print(f"\n[batch-runner-check3] {n_cases} cases, {n_workers} workers, {elapsed:.2f}s wall time")
