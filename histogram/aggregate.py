"""Aggregation/subsetting over many histogram files, plus the matched-
percentile-threshold method (ported from python_base's
wofs_dz_histogram_plotter.py, generalized into a reusable function instead of
copy-pasted per comparison).

The whole point of histogram_io.py's per-slice (not pre-collapsed) schema is
so this module can recombine an arbitrary subset of slices across many
files -- e.g. every MRMS slice valid at a given hour of day (a climatology),
or every model slice within a given forecast-lead-hour range (a "day N of
the forecast" bucket) -- which would be impossible to reconstruct from
python_base's original one-histogram-per-directory output.
"""

from typing import Callable

import numpy as np

from .histogram_io import HistogramSlice, read_histogram_file

Predicate = Callable[[HistogramSlice], bool]


def by_hour_of_day(hours: int | set[int]) -> Predicate:
    """Keep only slices whose valid_time falls in the given UTC hour(s) --
    e.g. by_hour_of_day(18) for an 18Z MRMS climatology, or
    by_hour_of_day({0, 1, 2}) for a multi-hour window."""
    wanted = {hours} if isinstance(hours, int) else set(hours)
    return lambda s: s.valid_time.hour in wanted


def by_lead_hours_range(min_hours: float, max_hours: float) -> Predicate:
    """Keep only slices with lead_hours in [min_hours, max_hours) -- e.g.
    by_lead_hours_range(0, 24) for "day 1" of a multi-day forecast,
    by_lead_hours_range(24, 48) for "day 2". Slices with no lead_hours
    concept (pure obs) are never kept by this predicate."""
    return lambda s: s.lead_hours is not None and min_hours <= s.lead_hours < max_hours


def sum_histograms(paths: list[str], predicate: Predicate | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Reads many histogram files and sums every slice's bin counts where
    predicate(slice) is True (or every slice, if predicate is omitted).
    Returns (bins, summed_hist). Raises if the files don't share the same
    bin edges -- summing histograms built with different bins would be
    meaningless, not silently mergeable."""
    if not paths:
        raise ValueError("sum_histograms: paths must be non-empty")

    bins = None
    total = None
    n_matched = 0
    for path in paths:
        contents = read_histogram_file(path)
        if bins is None:
            bins = contents.bins
            total = np.zeros(len(bins) - 1, dtype=np.int64)
        elif not np.array_equal(bins, contents.bins):
            raise ValueError(
                f"sum_histograms: '{path}' has different bin edges than earlier files -- "
                "histograms built with different bins cannot be summed"
            )
        for s in contents.slices:
            if predicate is None or predicate(s):
                total += s.hist
                n_matched += 1

    if n_matched == 0:
        raise ValueError("sum_histograms: predicate matched zero slices across all given files")

    return bins, total


def histogram_to_pdf(bins: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """Normalized probability density over the full histogram, exactly as
    given -- no bin is zeroed or excluded. Every valid gridpoint (clear air
    included) counts toward the total, so two distributions built from the
    same grid/edge_trim are directly comparable as gridpoint fractions, and
    percentiles reflect the whole population rather than a source-dependent
    "clear air" carve-out (there is no single dBZ value that reliably means
    "clear air" across sources -- e.g. confirmed MRMS's is ~0.0 dBZ, MPAS's
    is -35.0 dBZ -- so this module no longer tries to guess one)."""
    hist = hist.astype(np.float64).copy()
    total = hist.sum()
    if total == 0:
        raise ValueError("histogram_to_pdf: all counts are zero, cannot normalize")
    return hist / total


def histogram_to_cdf(bins: np.ndarray, hist: np.ndarray) -> np.ndarray:
    return np.cumsum(histogram_to_pdf(bins, hist))


def value_at_percentile(bins: np.ndarray, cdf: np.ndarray, percentile: float) -> float:
    """The bin value (left edge, per python_base's own `plot_bins` convention
    -- within one bin_width of the true value) whose CDF is closest to
    `percentile` (0.0-1.0 scale)."""
    if not 0.0 <= percentile <= 1.0:
        raise ValueError(f"value_at_percentile: percentile must be in [0, 1], got {percentile}")
    idx = int(np.argmin(np.abs(cdf - percentile)))
    return float(bins[:-1][idx])


def match_percentile_threshold(
    source_bins: np.ndarray, source_hist: np.ndarray, source_value: float,
    target_bins: np.ndarray, target_hist: np.ndarray,
) -> tuple[float, float]:
    """The generalized port of wofs_dz_histogram_plotter.py's threshold-
    matching method: find `source_value`'s percentile in the source
    distribution, then find the target distribution's value at that same
    percentile. Returns (source_percentile, target_value).

    Example: "MRMS's 40 dBZ is the Nth percentile of the real MRMS
    distribution -- what MPAS reflectivity value is that same Nth percentile
    of the MPAS distribution?" -- the actual method used to pick matched,
    source/target-comparable object-identification thresholds.
    """
    source_cdf = histogram_to_cdf(source_bins, source_hist)
    source_idx = int(np.argmin(np.abs(source_bins[:-1] - source_value)))
    source_percentile = float(source_cdf[source_idx])

    target_cdf = histogram_to_cdf(target_bins, target_hist)
    target_value = value_at_percentile(target_bins, target_cdf, source_percentile)

    return source_percentile, target_value
