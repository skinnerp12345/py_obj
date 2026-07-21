"""Core histogram computation: generalizes python_base's mrms_dz_histogram_*.py
and wofs_dz_histogram_*.py into one configurable-bins/configurable-variable
function. Takes a plain array (already loaded via python_obj.regrid's own
loaders) -- no file I/O here, mirroring this codebase's existing
identify_objects()-vs-run_object_id_series() split between core algorithm and
pipeline/batch driver.
"""

import numpy as np

DEFAULT_BIN_MIN = -20.0
DEFAULT_BIN_MAX = 80.0
DEFAULT_BIN_WIDTH = 0.2
DEFAULT_EDGE_TRIM = 7  # matches python_base's hardcoded edge=7 (avoids regridding-edge artifacts)


def default_bin_edges(
    bin_min: float = DEFAULT_BIN_MIN,
    bin_max: float = DEFAULT_BIN_MAX,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> np.ndarray:
    """Bin edges from bin_min to bin_max (inclusive) in bin_width steps.

    Uses linspace with an explicitly computed point count (not np.arange with
    a float step) -- np.arange's float-accumulation error can silently add or
    drop the final edge for step sizes like 0.2 that aren't exactly
    representable in binary floating point. Rounded to 6 decimals to strip
    the residual representation noise linspace itself still leaves behind.
    """
    n_edges = round((bin_max - bin_min) / bin_width) + 1
    return np.round(np.linspace(bin_min, bin_max, n_edges), 6)


def compute_histogram(
    data: np.ndarray,
    bins: np.ndarray,
    edge_trim: int = DEFAULT_EDGE_TRIM,
    clip_negative_to_zero: bool = False,
    missing_value: float | None = None,
) -> np.ndarray:
    """Histogram of `data` against `bins` (edges), trimming `edge_trim`
    pixels off each side of the last two axes first (domain-boundary
    artifacts, e.g. regridding edge effects -- ported from python_base's
    hardcoded edge=7 crop). Works for a plain 2D (y, x) obs field or a 3D
    (member, y, x) model field alike -- trimming only ever touches the last
    two axes, and every remaining pixel (across all members, if present) is
    ravelled into one histogram together, matching python_base's WoFS
    behavior of pooling all ensemble members into a single distribution.

    Every valid pixel is guaranteed to land in some bin: real values outside
    `[bins[0], bins[-1]]` are clamped to the nearest edge bin (e.g. a real
    -23.2 dBZ reading with bins starting at -20.0 is counted in the
    `[-20, -19.8)` bin), rather than silently dropped by np.histogram's own
    out-of-range exclusion. This is deliberate: it makes `hist.sum()` equal
    to the count of valid input pixels regardless of where a given source's
    "clear air"/floor value happens to sit (confirmed on real data to vary
    by source -- MRMS's is ~0.0 dBZ, MPAS's is exactly -35.0 dBZ), so two
    histograms built from the same grid/edge_trim always carry the same
    total gridpoint count and are directly comparable.

    missing_value: a sentinel to exclude entirely (e.g. MRMS's -999.0 "no
    observational coverage" value) -- NOT clamped into the first bin like a
    real out-of-range reading. This must be passed explicitly for any source
    that has one: a real fill value (e.g. MRMS's -999.0) is finite and, once
    out-of-range clamping is in play, would otherwise be silently counted as
    a legitimate -20 dBZ reading instead of being excluded. (Confirmed: the
    interpolated-MRMS loader returns -999.0 as a literal value, not NaN --
    it was previously excluded only by coincidence, via np.histogram's own
    range truncation, which this function no longer relies on.)

    clip_negative_to_zero (default False, NOT the python_base default):
    python_base's WoFS scripts silently floored every negative dBZ value to
    0 before histogramming. That's a real, data-altering choice, so here it
    is an explicit opt-in rather than a silent default -- callers that want
    the old behavior must ask for it.

    Returns raw bin counts (`np.histogram(...)[0]`), not a probability
    density -- summing counts across many files/times is exact; summing
    already-normalized densities is not.
    """
    if edge_trim > 0:
        if data.ndim < 2:
            raise ValueError(f"compute_histogram: expected at least 2D data, got shape {data.shape}")
        trimmed = data[..., edge_trim:-edge_trim, edge_trim:-edge_trim]
    else:
        trimmed = data

    flat = np.asarray(trimmed).ravel()
    flat = flat[np.isfinite(flat)]  # NaNs/inf never counted
    if missing_value is not None:
        flat = flat[flat != missing_value]  # sentinel excluded entirely, never clamped in as data
    if clip_negative_to_zero:
        flat = np.where(flat < 0.0, 0.0, flat)

    flat = np.clip(flat, bins[0], bins[-1])  # out-of-range real values -> nearest edge bin, never dropped
    counts, _ = np.histogram(flat, bins)
    return counts.astype(np.int64)
