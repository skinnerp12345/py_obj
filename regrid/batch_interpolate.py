"""Batch-interpolate every MRMS file in a directory to a fixed target grid,
in parallel, writing one NetCDF file per input file.

Standalone capability, usable with no downstream object ID/matching applied --
e.g. to pre-compute and store interpolated MRMS fields for reuse, or as a
deliverable in its own right.

Multiprocessing design rationale: the source (native MRMS) grid and target
(model) grid are identical across
every file in a batch, so there is exactly ONE ESMF regridder to build for an
entire run, not one per file, and not one per worker call:
  - the main process pre-warms the on-disk weight cache once, serially, before
    spawning any workers (avoids workers racing to write the same weight file)
  - each worker then builds its own regridder ONCE at process startup (a
    Pool(initializer=...) hit against the now-warm cache), reusing it for every
    file that worker processes, rather than rebuilding it per file
"""

import glob
import os
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool

import netCDF4
import numpy as np

from .grid_spec import GridSpec, crop_to_bbox
from .io_mrms import MRMS_MISSING_VALUE, clip_near_zero_sentinel, load_mrms_grib2
from .io_grid import load_target_grid
from .regridder import build_conservative_regridder, regrid_field


@dataclass
class BatchFileResult:
    input_path: str
    output_path: str | None
    success: bool
    error: str | None = None


@dataclass
class BatchSummary:
    n_total: int
    n_success: int
    n_failed: int
    failures: list = field(default_factory=list)  # list[BatchFileResult], failures only

    def __str__(self) -> str:
        lines = [f"BatchSummary: {self.n_success}/{self.n_total} succeeded, {self.n_failed} failed"]
        for f in self.failures:
            lines.append(f"  FAILED: {f.input_path}: {f.error}")
        return "\n".join(lines)


def discover_mrms_files(input_dir: str, pattern: str = "**/*.grib2*") -> list[str]:
    """Recursively find MRMS files under input_dir (handles a YYYYMMDD/ per-day
    subdirectory layout, as well as a flat directory)."""
    matches = glob.glob(os.path.join(input_dir, pattern), recursive=True)
    files = sorted(f for f in matches if os.path.isfile(f))
    return files


def _filter_by_date_range(files: list[str], date_range: tuple[str, str] | None) -> list[str]:
    """Filter files by their immediate parent directory name, if it looks like a
    YYYYMMDD date and date_range is given. Files whose parent directory doesn't
    look like a date are kept as-is (can't be filtered, but not dropped either)."""
    if date_range is None:
        return files
    start, end = date_range
    kept = []
    for f in files:
        dirname = os.path.basename(os.path.dirname(f))
        if len(dirname) == 8 and dirname.isdigit():
            if start <= dirname <= end:
                kept.append(f)
        else:
            kept.append(f)
    return kept


def make_output_path(mrms_filepath: str, valid_time: datetime, output_dir: str, mirror_subdirs: bool = True) -> str:
    """output_dir/YYYYMMDD/interp_mrms_YYYYMMDD_HHMMSS.nc, derived from valid_time
    (not the input filename), so naming is robust to whatever the source looks like."""
    yyyymmdd = valid_time.strftime("%Y%m%d")
    hhmmss = valid_time.strftime("%H%M%S")
    out_subdir = os.path.join(output_dir, yyyymmdd) if mirror_subdirs else output_dir
    return os.path.join(out_subdir, f"interp_mrms_{yyyymmdd}_{hhmmss}.nc")


def write_interpolated_mrms_netcdf(
    out_path: str,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    data2d: np.ndarray,
    valid_time: datetime,
    varname: str = "refl_consv",
    fill_value: float = MRMS_MISSING_VALUE,
    source_file: str | None = None,
) -> None:
    """Write lat/lon/<varname> to a NetCDF file, using the same variable
    naming convention this library's own load_mrms_netcdf() (regrid/io_mrms.py)
    expects ('lat', 'lon', and a reflectivity variable)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ny, nx = data2d.shape
    with netCDF4.Dataset(out_path, "w") as ds:
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)

        lat_var = ds.createVariable("lat", "f8", ("south_north", "west_east"))
        lon_var = ds.createVariable("lon", "f8", ("south_north", "west_east"))
        data_var = ds.createVariable(varname, "f8", ("south_north", "west_east"), fill_value=fill_value)

        lat_var[:, :] = lat2d
        lon_var[:, :] = lon2d
        data_var[:, :] = data2d

        ds.valid_time = valid_time.isoformat()
        if source_file is not None:
            ds.source_file = os.path.basename(source_file)


# Worker-process globals, populated once by _init_worker (either directly by the
# main process to pre-warm the weight cache, or as a multiprocessing.Pool
# initializer so each worker builds its regridder exactly once).
_TARGET_GRID: GridSpec | None = None
_SRC_BBOX: tuple[float, float, float, float] | None = None
_REGRIDDER = None
_BBOX_BUFFER_DEG = 0.3


def _init_worker(
    target_grid_file: str | None,
    target_lat_name: str,
    target_lon_name: str,
    weight_cache_dir: str,
    sample_mrms_file: str,
    bbox_buffer_deg: float = 0.3,
    target_grid: GridSpec | None = None,
) -> None:
    global _TARGET_GRID, _SRC_BBOX, _REGRIDDER, _BBOX_BUFFER_DEG

    _BBOX_BUFFER_DEG = bbox_buffer_deg
    # target_grid (an already-built GridSpec, e.g. from build_corner_spacing_grid)
    # takes priority over target_grid_file when both are somehow given -- callers
    # are expected to pass exactly one (run_batch_interpolation enforces this),
    # this is just a defensive tie-breaker for _init_worker's own direct callers.
    _TARGET_GRID = target_grid if target_grid is not None else load_target_grid(target_grid_file, target_lat_name, target_lon_name)
    _SRC_BBOX = (
        float(_TARGET_GRID.lat2d.min()),
        float(_TARGET_GRID.lat2d.max()),
        float(_TARGET_GRID.lon2d.min()),
        float(_TARGET_GRID.lon2d.max()),
    )

    # MRMS's native grid is identical across every file, so any one file's lat/lon
    # can be used to (re)construct the same cropped source grid the regridder was
    # (or will be) built from -- this is what lets build_conservative_regridder's
    # weight-cache key match between the main process's pre-warm call and every
    # worker's own call below.
    sample = load_mrms_grib2(sample_mrms_file)
    src_grid_full = GridSpec(lat2d=sample.lat2d, lon2d=sample.lon2d)
    cropped_src_grid, _ = crop_to_bbox(src_grid_full, sample.data, *_SRC_BBOX, buffer_deg=_BBOX_BUFFER_DEG)

    _REGRIDDER = build_conservative_regridder(cropped_src_grid, _TARGET_GRID, weight_cache_dir)


def _process_one_file(args: tuple[str, str, str, float]) -> BatchFileResult:
    mrms_filepath, output_dir, varname, fill_value = args
    try:
        mrms = load_mrms_grib2(mrms_filepath)
        data = clip_near_zero_sentinel(mrms.data)

        src_grid_full = GridSpec(lat2d=mrms.lat2d, lon2d=mrms.lon2d)
        _, cropped_data = crop_to_bbox(src_grid_full, data, *_SRC_BBOX, buffer_deg=_BBOX_BUFFER_DEG)

        out, _report = regrid_field(_REGRIDDER, cropped_data, fill_value=fill_value, missing_value=MRMS_MISSING_VALUE)

        out_path = make_output_path(mrms_filepath, mrms.valid_time, output_dir)
        write_interpolated_mrms_netcdf(
            out_path, _TARGET_GRID.lat2d, _TARGET_GRID.lon2d, out, mrms.valid_time,
            varname=varname, fill_value=fill_value, source_file=mrms_filepath,
        )
        return BatchFileResult(input_path=mrms_filepath, output_path=out_path, success=True)
    except Exception as exc:  # noqa: BLE001 -- deliberate: collect per-file failures, don't crash the pool
        return BatchFileResult(input_path=mrms_filepath, output_path=None, success=False, error=f"{type(exc).__name__}: {exc}")


def run_batch_interpolation(
    input_dir: str,
    output_dir: str,
    target_grid_file: str | None = None,
    target_lat_name: str = "latitude",
    target_lon_name: str = "longitude",
    weight_cache_dir: str = "weight_cache",
    n_workers: int = 8,
    bbox_buffer_deg: float = 0.3,
    varname: str = "refl_consv",
    fill_value: float = MRMS_MISSING_VALUE,
    date_range: tuple[str, str] | None = None,
    max_files: int | None = None,
    file_pattern: str = "**/*.grib2*",
    target_grid: GridSpec | None = None,
) -> BatchSummary:
    """Interpolate every MRMS file under input_dir onto a target grid, writing
    one NetCDF file per input under output_dir.

    Exactly one of target_grid_file (read via load_target_grid()) or
    target_grid (an already-built GridSpec -- e.g. from
    build_corner_spacing_grid(), for a user-specified corner+spacing+dims
    grid with no backing model file) must be given. This is a generic "bring
    your own GridSpec" escape hatch rather than teaching this function about
    corner/spacing/dims specifically, so any future grid-construction method
    plugs in the same way with no signature changes needed here.

    date_range, if given, is an inclusive (start_yyyymmdd, end_yyyymmdd) filter
    applied to each discovered file's per-day subdirectory name.

    max_files, if given, caps the number of files processed after discovery/date
    filtering -- a quick dry-run/smoke-test knob, not something a production run
    would normally set.

    file_pattern (default "**/*.grib2*", i.e. every grib2 file, preserving prior
    behavior exactly) restricts discovery to one MRMS product when a date
    directory holds more than one (e.g. MergedReflectivityQCComposite alongside
    MESH or RotationTrack) -- load_mrms_grib2() has no product-type awareness of
    its own (it just reads whichever GRIB2 message is first in the file), so an
    unfiltered glob sweeping up the wrong product would silently interpolate and
    write out non-reflectivity values as if they were dBZ. e.g.
    "**/*MergedReflectivityQCComposite*" selects only that product.
    """
    if (target_grid_file is None) == (target_grid is None):
        raise ValueError(
            "run_batch_interpolation: exactly one of target_grid_file or target_grid must be given "
            f"(got target_grid_file={target_grid_file!r}, target_grid={'<GridSpec>' if target_grid is not None else None})"
        )

    files = discover_mrms_files(input_dir, pattern=file_pattern)
    files = _filter_by_date_range(files, date_range)
    if not files:
        raise ValueError(f"No MRMS files found under '{input_dir}' (pattern='{file_pattern}', date_range={date_range})")
    if max_files is not None:
        files = files[:max_files]

    # Pre-warm the weight cache ONCE, serially, in the main process -- this calls
    # the same construction _init_worker will call, so it populates this
    # process's own globals too (harmless; the main process doesn't process
    # files) but more importantly writes the weight-cache file before any worker
    # starts, so every worker's own call below is a cache hit, not a race.
    _init_worker(
        target_grid_file, target_lat_name, target_lon_name, weight_cache_dir,
        sample_mrms_file=files[0], bbox_buffer_deg=bbox_buffer_deg, target_grid=target_grid,
    )

    task_args = [(f, output_dir, varname, fill_value) for f in files]

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(target_grid_file, target_lat_name, target_lon_name, weight_cache_dir, files[0], bbox_buffer_deg, target_grid),
    ) as pool:
        results = pool.map(_process_one_file, task_args)

    failures = [r for r in results if not r.success]
    summary = BatchSummary(
        n_total=len(results),
        n_success=len(results) - len(failures),
        n_failed=len(failures),
        failures=failures,
    )
    print(summary)
    return summary
