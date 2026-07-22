"""Histogram-capability validation: generalizes python_base's
mrms_dz_histogram_*.py/wofs_dz_histogram_*.py into python_obj.histogram +
the build_histogram_{mrms,model}.py/aggregate_histograms.py drivers.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest python_obj/tests/test_histogram.py -v -s
"""

import glob
import os
from datetime import datetime

import netCDF4
import numpy as np
import pytest

from python_obj.config import load_config
from python_obj.histogram import (
    HistogramSlice,
    by_hour_of_day,
    by_lead_hours_range,
    compute_histogram,
    default_bin_edges,
    histogram_to_cdf,
    histogram_to_pdf,
    match_percentile_threshold,
    read_histogram_file,
    sum_histograms,
    value_at_percentile,
    write_histogram_file,
)
from python_obj.regrid import load_mrms_netcdf, load_model_netcdf

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


# --- Check 1: default_bin_edges correctness -------------------------------

def test_default_bin_edges_matches_stated_default():
    bins = default_bin_edges()
    print(f"\n[hist-check1] n_edges={len(bins)} first={bins[0]} last={bins[-1]}")
    assert bins[0] == -20.0
    assert bins[-1] == 80.0
    assert len(bins) == 501  # (80 - -20) / 0.2 + 1
    diffs = np.diff(bins)
    assert np.allclose(diffs, 0.2), "bin widths should be uniform 0.2 dBZ, no float-accumulation drift"


def test_default_bin_edges_custom_range():
    bins = default_bin_edges(bin_min=0.0, bin_max=10.0, bin_width=1.0)
    np.testing.assert_array_equal(bins, np.arange(0.0, 11.0, 1.0))


# --- Check 2: compute_histogram (synthetic) -------------------------------

def test_compute_histogram_edge_trim_and_counts():
    bins = default_bin_edges(bin_min=0.0, bin_max=10.0, bin_width=1.0)
    data = np.full((20, 20), 5.0)
    data[:2, :] = 999.0  # would-be-trimmed border noise, way outside bin range
    data[-2:, :] = 999.0
    data[:, :2] = 999.0
    data[:, -2:] = 999.0

    counts = compute_histogram(data, bins, edge_trim=2)
    print(f"\n[hist-check2] trimmed total count={counts.sum()} (expect 16x16=256)")
    assert counts.sum() == 256  # (20-2*2)^2
    assert counts[5] == 256  # all remaining pixels are exactly 5.0 -> bin [5,6)


def test_compute_histogram_negative_clip_is_opt_in():
    bins = default_bin_edges(bin_min=-5.0, bin_max=5.0, bin_width=1.0)
    data = np.array([[-3.0, -3.0], [2.0, 2.0]])

    counts_default = compute_histogram(data, bins, edge_trim=0)
    assert counts_default[2] == 2  # bin [-3,-2) gets the two -3.0 values, untouched

    counts_clipped = compute_histogram(data, bins, edge_trim=0, clip_negative_to_zero=True)
    zero_bin_idx = int(np.searchsorted(bins, 0.0, side="right") - 1)
    print(f"\n[hist-check2b] clipped: zero_bin count={counts_clipped[zero_bin_idx]} (expect 2, the former -3.0s)")
    assert counts_clipped[zero_bin_idx] == 2
    assert counts_clipped[2] == 0  # the -3 bin is now empty


def test_compute_histogram_excludes_nan():
    bins = default_bin_edges(bin_min=0.0, bin_max=10.0, bin_width=1.0)
    data = np.array([[1.0, np.nan], [np.nan, 2.0]])
    counts = compute_histogram(data, bins, edge_trim=0)
    assert counts.sum() == 2, "NaNs must never be counted"


def test_compute_histogram_clamps_out_of_range_values_to_nearest_edge_bin():
    # every valid pixel must land in some bin -- out-of-range real values
    # (e.g. a real dBZ reading below bin_min, like MPAS's -35.0 dBZ "clear
    # air" floor) get clamped to the nearest edge bin rather than dropped,
    # so hist.sum() always equals the count of valid input pixels.
    bins = default_bin_edges(bin_min=-20.0, bin_max=80.0, bin_width=0.2)
    data = np.array([[-23.2, 85.0], [5.0, 5.0]])
    counts = compute_histogram(data, bins, edge_trim=0)
    print(f"\n[hist-check2c] total={counts.sum()} (expect 4, none dropped), "
          f"first bin={counts[0]} (expect 1, the clamped -23.2), last bin={counts[-1]} (expect 1, the clamped 85.0)")
    assert counts.sum() == 4, "no pixel should be silently dropped due to being outside the bin range"
    assert counts[0] == 1  # -23.2 clamped into [-20, -19.8)
    assert counts[-1] == 1  # 85.0 clamped into the last (inclusive) bin


def test_compute_histogram_missing_value_excluded_not_clamped():
    # regression check for the real risk found while designing this: MRMS's
    # -999.0 "no coverage" sentinel is a real finite value (confirmed: the
    # loader does not convert it to NaN), so once out-of-range values are
    # clamped instead of dropped, it must be filtered explicitly via
    # missing_value -- otherwise it would be silently counted as a fake -20
    # dBZ reading.
    bins = default_bin_edges(bin_min=-20.0, bin_max=80.0, bin_width=0.2)
    data = np.array([-999.0, -999.0, 5.0, 5.0])

    counts_unsafe = compute_histogram(data, bins, edge_trim=0)
    print(f"\n[hist-check2d] without missing_value: first bin={counts_unsafe[0]} (demonstrates the risk: -999s wrongly clamped in)")
    assert counts_unsafe[0] == 2, "demonstrates the risk this guards against: -999 silently clamped into the first bin"

    counts_safe = compute_histogram(data, bins, edge_trim=0, missing_value=-999.0)
    print(f"[hist-check2d] with missing_value=-999.0: first bin={counts_safe[0]} (expect 0), total={counts_safe.sum()} (expect 2)")
    assert counts_safe[0] == 0
    assert counts_safe.sum() == 2


# --- Check 3: histogram file round-trip -----------------------------------

def test_histogram_file_roundtrip_obs_and_model_slices(tmp_path):
    bins = default_bin_edges(bin_min=0.0, bin_max=10.0, bin_width=1.0)
    h1 = np.zeros(len(bins) - 1, dtype=np.int64); h1[3] = 100
    h2 = np.zeros(len(bins) - 1, dtype=np.int64); h2[5] = 50

    # obs-style: no lead_hours/member_id at all
    p_obs = str(tmp_path / "obs.nc")
    obs_slices = [
        HistogramSlice(valid_time=datetime(2023, 5, 1, 0, 0, 0), hist=h1),
        HistogramSlice(valid_time=datetime(2023, 5, 1, 1, 0, 0), hist=h2),
    ]
    write_histogram_file(p_obs, bins, obs_slices, "refl_consv", 7, False, 2)
    c_obs = read_histogram_file(p_obs)
    assert np.array_equal(c_obs.bins, bins)
    assert len(c_obs.slices) == 2
    assert c_obs.slices[0].lead_hours is None and c_obs.slices[0].member_id is None
    np.testing.assert_array_equal(c_obs.slices[0].hist, h1)
    np.testing.assert_array_equal(c_obs.slices[1].hist, h2)
    assert c_obs.varname == "refl_consv" and c_obs.edge_trim == 7 and c_obs.clip_negative_to_zero is False
    assert c_obs.n_source_files == 2
    print(f"\n[hist-check3] obs round-trip OK: {len(c_obs.slices)} slices, n_source_files={c_obs.n_source_files}")

    # model-style: every slice has lead_hours + member_id
    p_model = str(tmp_path / "model.nc")
    model_slices = [
        HistogramSlice(valid_time=datetime(2023, 5, 1, 1, 0, 0), hist=h1, lead_hours=1.0, member_id="mem00"),
        HistogramSlice(valid_time=datetime(2023, 5, 1, 1, 0, 0), hist=h2, lead_hours=1.0, member_id="mem01"),
    ]
    write_histogram_file(p_model, bins, model_slices, "comp_dz", 0, True, 1)
    c_model = read_histogram_file(p_model)
    assert c_model.slices[0].lead_hours == 1.0 and c_model.slices[0].member_id == "mem00"
    assert c_model.slices[1].member_id == "mem01"
    assert c_model.clip_negative_to_zero is True
    print(f"[hist-check3] model round-trip OK: lead_hours={[s.lead_hours for s in c_model.slices]}, "
          f"member_ids={[s.member_id for s in c_model.slices]}")


def test_write_histogram_file_rejects_empty_or_mismatched_shape(tmp_path):
    bins = default_bin_edges(bin_min=0.0, bin_max=10.0, bin_width=1.0)
    with pytest.raises(ValueError, match="non-empty"):
        write_histogram_file(str(tmp_path / "x.nc"), bins, [], "v", 0, False, 0)

    bad_slice = HistogramSlice(valid_time=datetime(2023, 5, 1), hist=np.zeros(3))
    with pytest.raises(ValueError, match="does not match bins"):
        write_histogram_file(str(tmp_path / "y.nc"), bins, [bad_slice], "v", 0, False, 1)


# --- Check 4: aggregate.sum_histograms subsetting (synthetic, known truth) -

def test_sum_histograms_hour_of_day_subset_recovers_expected_counts(tmp_path):
    bins = default_bin_edges(bin_min=0.0, bin_max=10.0, bin_width=1.0)
    # 3 distinct hours, each with a distinct, known count at a distinct bin
    slices = [
        HistogramSlice(valid_time=datetime(2023, 5, 1, 0, 0, 0), hist=_hist_with(bins, {1: 10})),
        HistogramSlice(valid_time=datetime(2023, 5, 2, 0, 0, 0), hist=_hist_with(bins, {1: 20})),  # same hour, different day
        HistogramSlice(valid_time=datetime(2023, 5, 1, 12, 0, 0), hist=_hist_with(bins, {1: 999})),  # different hour
    ]
    p = str(tmp_path / "day.nc")
    write_histogram_file(p, bins, slices, "refl_consv", 7, False, 1)

    _, hour0_total = sum_histograms([p], predicate=by_hour_of_day(0))
    print(f"\n[hist-check4] hour=0 subset total at bin 1: {hour0_total[1]} (expect 30, excluding the 999 at hour 12)")
    assert hour0_total[1] == 30

    _, all_total = sum_histograms([p])
    assert all_total[1] == 1029


def test_sum_histograms_lead_hours_bucket_recovers_expected_counts(tmp_path):
    bins = default_bin_edges(bin_min=0.0, bin_max=10.0, bin_width=1.0)
    slices = [
        HistogramSlice(valid_time=datetime(2023, 5, 1, 1, 0, 0), hist=_hist_with(bins, {2: 5}), lead_hours=1.0),
        HistogramSlice(valid_time=datetime(2023, 5, 1, 20, 0, 0), hist=_hist_with(bins, {2: 7}), lead_hours=20.0),
        HistogramSlice(valid_time=datetime(2023, 5, 2, 1, 0, 0), hist=_hist_with(bins, {2: 500}), lead_hours=25.0),  # day 2
    ]
    p = str(tmp_path / "forecast.nc")
    write_histogram_file(p, bins, slices, "refl10cm_max", 7, False, 1)

    _, day1_total = sum_histograms([p], predicate=by_lead_hours_range(0.0, 24.0))
    print(f"\n[hist-check4b] day-1 (lead 0-24h) subset total at bin 2: {day1_total[2]} (expect 12, excluding the 500 at lead 25h)")
    assert day1_total[2] == 12

    _, day2_total = sum_histograms([p], predicate=by_lead_hours_range(24.0, 48.0))
    assert day2_total[2] == 500


def test_sum_histograms_rejects_mismatched_bins(tmp_path):
    bins_a = default_bin_edges(bin_min=0.0, bin_max=10.0, bin_width=1.0)
    bins_b = default_bin_edges(bin_min=0.0, bin_max=20.0, bin_width=1.0)
    p_a = str(tmp_path / "a.nc")
    p_b = str(tmp_path / "b.nc")
    write_histogram_file(p_a, bins_a, [HistogramSlice(valid_time=datetime(2023, 5, 1), hist=_hist_with(bins_a, {1: 1}))], "v", 0, False, 0)
    write_histogram_file(p_b, bins_b, [HistogramSlice(valid_time=datetime(2023, 5, 1), hist=_hist_with(bins_b, {1: 1}))], "v", 0, False, 0)

    with pytest.raises(ValueError, match="different bin edges"):
        sum_histograms([p_a, p_b])


def _hist_with(bins: np.ndarray, counts: dict) -> np.ndarray:
    h = np.zeros(len(bins) - 1, dtype=np.int64)
    for idx, val in counts.items():
        h[idx] = val
    return h


# --- Check 5: no bin is ever excluded from the PDF/CDF ---------------------
#
# Design revision: histogram_to_pdf/cdf no longer zero out a "clear air" bin
# at all -- there's no single dBZ value that reliably means "clear air"
# across sources (confirmed: MRMS's floor is ~0.0 dBZ, MPAS's is exactly
# -35.0 dBZ), so guessing one and zeroing it was a source-specific
# assumption baked into supposedly-generic code. Every bin (including
# whatever holds each source's own clear-air spike) now counts toward the
# total, so two histograms from the same grid/edge_trim carry the same
# total gridpoint count and are directly comparable.

def test_histogram_to_pdf_does_not_exclude_any_bin():
    bins = default_bin_edges(bin_min=-1.0, bin_max=10.0, bin_width=1.0)
    hist = _hist_with(bins, {0: 12345, 1: 100})  # bin 1 is [0,1) here

    pdf = histogram_to_pdf(bins, hist)
    print(f"\n[hist-check5] pdf at [0,1) bin={pdf[1]:.4f} (expect > 0, no bin is zeroed anymore)")
    assert pdf[1] > 0.0, "no bin, including whichever one holds a source's own clear-air spike, should be zeroed"
    assert np.isclose(pdf.sum(), 1.0)


# --- Check 6: real end-to-end against bundled sample_data -----------------

def test_real_build_histogram_drivers_and_matched_percentile(tmp_path):
    from python_obj.drivers import aggregate_histograms, build_histogram_mrms, build_histogram_model

    config_path = os.path.join(CONFIGS_DIR, "config_sample_histogram.yaml")

    mrms_out_paths = build_histogram_mrms.run_one_case(config_path)
    model_out_path = build_histogram_model.run_one_case(config_path)

    assert len(mrms_out_paths) == 1  # one bundled day
    mrms_contents = read_histogram_file(mrms_out_paths[0])
    assert len(mrms_contents.slices) == 3  # 3 bundled interpolated-MRMS files
    assert all(s.hist.sum() > 0 for s in mrms_contents.slices), "expected real, non-zero counts"

    model_contents = read_histogram_file(model_out_path)
    assert len(model_contents.slices) == 3  # f001-f003
    assert {s.lead_hours for s in model_contents.slices} == {1.0, 2.0, 3.0}
    print(f"\n[hist-check6] MRMS slices={len(mrms_contents.slices)}, "
          f"model slices={len(model_contents.slices)} lead_hours={[s.lead_hours for s in model_contents.slices]}")

    # the actual goal of the fixed-range/clamped-edge design: MRMS and MPAS
    # here share the same 236x236 (post edge_trim=7) grid, so their total
    # histogram counts must now be exactly equal -- before this design
    # revision this was false (115,517 for MPAS vs. 167,088 for MRMS)
    # because MPAS's real -35.0 dBZ clear-air floor fell entirely outside
    # the default [-20, 80] bin range and was silently dropped.
    mrms_total = sum(int(s.hist.sum()) for s in mrms_contents.slices)
    model_total = sum(int(s.hist.sum()) for s in model_contents.slices)
    print(f"[hist-check6] MRMS total counts={mrms_total}, model total counts={model_total} (expect equal, same grid)")
    assert mrms_total == model_total, "same grid + edge_trim must yield equal total gridpoint counts across sources"

    result = aggregate_histograms.run_one_case(config_path, source_threshold_dbz=40.0)
    assert 0.0 <= result["source_percentile"] <= 1.0
    print(f"[hist-check6] matched-percentile: {result['source_percentile']*100:.1f}th pct -> "
          f"target value {result['target_value']:.1f} dBZ")

    # independent cross-check: does the histogram-based percentile roughly
    # match numpy.percentile computed directly on the raw ravelled MRMS data
    # (same edge_trim, same clamping-to-range applied by hand -- no clear-air
    # exclusion, matching the library's new full-inclusion behavior)?
    cfg = load_config(config_path)
    raw_values = []
    for f in sorted(glob.glob(os.path.join(cfg.histogram_observations.interp_mrms_dir, "**", "*.nc"), recursive=True)):
        field = load_mrms_netcdf(f, varname=cfg.histogram_observations.var_name,
                                  lat_name=cfg.histogram_observations.lat_name, lon_name=cfg.histogram_observations.lon_name)
        trimmed = field.data[7:-7, 7:-7].ravel()
        valid = trimmed[np.isfinite(trimmed)]
        if field.missing_value is not None:
            valid = valid[valid != field.missing_value]
        raw_values.append(valid)
    all_raw = np.concatenate(raw_values)
    all_raw_clamped = np.clip(all_raw, cfg.histogram_observations.bin_min, cfg.histogram_observations.bin_max)
    independent_pct = float((all_raw_clamped <= 40.0).mean())
    print(f"[hist-check6] independent numpy percentile at 40 dBZ: {independent_pct*100:.1f}th "
          f"(histogram-based: {result['source_percentile']*100:.1f}th)")
    assert abs(independent_pct - result["source_percentile"]) < 0.02, "histogram-based and direct percentile estimates should be close"


# --- Check 7: domain masking (real end-to-end, conus_east) ------------------

def test_real_mask_excludes_cells_entirely_not_as_fake_clear_air(tmp_path):
    """Masked cells must be excluded from the histogram entirely (as NaN),
    not zeroed like the object-ID pipeline does (which would inject fake
    clear-air counts) -- confirmed by comparing masked vs. unmasked total
    counts on the same real bundled grid, where conus_east is known (checked
    directly against the grid) to exclude ~51% of cells, a substantial,
    real, non-trivial fraction."""
    from python_obj.drivers import build_histogram_mrms

    unmasked_cfg_path = os.path.join(CONFIGS_DIR, "config_sample_histogram.yaml")
    cfg = load_config(unmasked_cfg_path)  # resolves interp_mrms_dir to an absolute path

    masked_cfg_path = str(tmp_path / "config_masked.yaml")
    with open(masked_cfg_path, "w") as f:
        f.write(
            "histogram_observations:\n"
            f"  interp_mrms_dir: {cfg.histogram_observations.interp_mrms_dir}\n"
            f"  var_name: {cfg.histogram_observations.var_name}\n"
            f"  lat_name: {cfg.histogram_observations.lat_name}\n"
            f"  lon_name: {cfg.histogram_observations.lon_name}\n"
            f"  output_dir: {tmp_path / 'hist_mrms_masked'}\n"
            "  mask: conus_east\n"
        )

    unmasked_paths = build_histogram_mrms.run_one_case(unmasked_cfg_path)
    masked_paths = build_histogram_mrms.run_one_case(masked_cfg_path)

    unmasked_total = sum(int(s.hist.sum()) for p in unmasked_paths for s in read_histogram_file(p).slices)
    masked_total = sum(int(s.hist.sum()) for p in masked_paths for s in read_histogram_file(p).slices)
    excluded_frac = 1.0 - masked_total / unmasked_total
    print(f"\n[hist-check7] unmasked total={unmasked_total}, masked (conus_east) total={masked_total}, "
          f"excluded={excluded_frac*100:.1f}% (expect ~51%, matching the grid's own known conus_east fraction)")
    assert masked_total < unmasked_total, "masking must strictly reduce the total count -- cells must be excluded, not zeroed in"
    assert 0.45 < excluded_frac < 0.57
