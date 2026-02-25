#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native-resolution GHRSST Hobday analysis in spatial chunks (tiles), with
full-output mode for typhoon interaction studies.

This runner keeps native grid resolution and processes one tile at a time,
so memory usage stays bounded. It is designed for long runs on tborder and
supports resume-by-tile.

Outputs per tile:
- climatology + threshold DOY NetCDF (main baseline)
- daily heatwave state NetCDF (anomaly/delta/mask/start/exceedance)
- optional event catalog NetCDF
- annual metrics NetCDF
- trend maps NetCDF
- baseline sensitivity summary NetCDF (worker-level)
- done marker

Also writes domain-level QC/inventory reports once (worker 0).
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ghrsst_hobday_pipeline import (
    build_event_catalog,
    compute_annual_metrics,
    parse_baseline_window,
    parse_yyyymmdd_from_filename,
    progress_iter,
    run_baseline_detection,
    save_annual_metrics_nc,
    save_trend_nc,
    summarize_baseline,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Native-resolution GHRSST Hobday analysis in chunks."
    )
    p.add_argument("--input-dir", required=True)
    p.add_argument(
        "--file-glob",
        default="*JPL-L4_GHRSST-SSTfnd-MUR-GLOB*.nc4",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start-date", default="2003-01-01")
    p.add_argument("--end-date", default="2025-12-31")
    p.add_argument("--var-name", default="analysed_sst")
    p.add_argument("--lat-name", default="lat")
    p.add_argument("--lon-name", default="lon")
    p.add_argument(
        "--bbox",
        default="100,180,0,70",
        help=(
            "lon_min,lon_max,lat_min,lat_max in dataset convention. "
            "Default matches typhoon coverage: 100E-180E, 0N-70N."
        ),
    )
    p.add_argument("--tile-lat", type=int, default=512)
    p.add_argument("--tile-lon", type=int, default=512)
    p.add_argument("--pctile", type=float, default=90.0)
    p.add_argument("--window-half-width", type=int, default=5)
    p.add_argument("--smooth-percentile-width", type=int, default=31)
    p.add_argument("--min-duration", type=int, default=5)
    p.add_argument("--max-gap", type=int, default=2)
    p.add_argument("--baseline-main", default="2003-2022")
    p.add_argument(
        "--baseline-sensitivity",
        default="",
        help="Semicolon-separated extra baselines for sensitivity summary.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of parallel workers over tile list.",
    )
    p.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="0-based worker index in [0, num_workers).",
    )
    p.add_argument(
        "--save-clim-threshold",
        action="store_true",
        help="Save per-tile climatology/threshold DOY NetCDF for main baseline.",
    )
    p.add_argument(
        "--save-daily-state",
        action="store_true",
        help="Save per-tile daily heatwave state NetCDF.",
    )
    p.add_argument(
        "--save-events",
        action="store_true",
        help="Save per-tile event catalog NetCDF.",
    )
    p.add_argument(
        "--save-annual-trend",
        action="store_true",
        help="Save annual metrics and trend maps (disabled by default for basic run).",
    )
    p.add_argument(
        "--qc-csv-name",
        default="qc_ghrsst_2003_2025.csv",
        help="Final merged QC CSV filename under output directory.",
    )
    p.add_argument(
        "--bad-low-c",
        type=float,
        default=-2.5,
        help="Lower physical-limit threshold in degC for bad-value counting.",
    )
    p.add_argument(
        "--bad-high-c",
        type=float,
        default=40.0,
        help="Upper physical-limit threshold in degC for bad-value counting.",
    )
    p.add_argument(
        "--qc-wait-timeout-hours",
        type=float,
        default=72.0,
        help="Worker0 wait timeout for collecting all worker QC counters.",
    )
    p.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="NetCDF zlib compression level (0-9) for large outputs.",
    )
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def parse_bbox(text: str) -> Tuple[float, float, float, float]:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be lon_min,lon_max,lat_min,lat_max")
    return tuple(map(float, parts))


def parse_sensitivity_windows(text: str, main_label: str) -> List[str]:
    out: List[str] = []
    if not text.strip():
        return out
    for part in text.split(";"):
        p = part.strip()
        if not p:
            continue
        b = parse_baseline_window(p)
        if b.label == main_label:
            continue
        if b.label not in out:
            out.append(b.label)
    return out


def discover_files(
    input_dir: str, file_glob: str, start_date: str, end_date: str
) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, file_glob)))
    if not files:
        raise FileNotFoundError(f"No files in {input_dir} matching {file_glob}")
    rows = []
    for p in files:
        dt = parse_yyyymmdd_from_filename(p)
        if dt is None:
            continue
        rows.append((p, dt.normalize()))
    df = pd.DataFrame(rows, columns=["path", "date"])
    df = df.sort_values(["date", "path"]).reset_index(drop=True)
    d0 = pd.to_datetime(start_date).normalize()
    d1 = pd.to_datetime(end_date).normalize()
    df = df[(df["date"] >= d0) & (df["date"] <= d1)].copy()
    if df.empty:
        raise RuntimeError("No files after date filtering.")
    # keep one file per day for processing, but QC writer handles duplicates
    df = df.drop_duplicates(subset=["date"], keep="first")
    return df.reset_index(drop=True)


def write_qc_reports(
    input_dir: str,
    file_glob: str,
    start_date: str,
    end_date: str,
    out_dir: str,
    compression_level: int,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, file_glob)))
    if not files:
        raise FileNotFoundError(f"No files in {input_dir} matching {file_glob}")

    recs = []
    for p in files:
        dt = parse_yyyymmdd_from_filename(p)
        recs.append({"path": p, "date": dt})
    df = pd.DataFrame(recs)
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        raise RuntimeError("No parseable GHRSST dates from filenames.")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "path"]).reset_index(drop=True)
    df["is_duplicate_date"] = df.duplicated(subset=["date"], keep=False)

    d0 = pd.to_datetime(start_date).normalize()
    d1 = pd.to_datetime(end_date).normalize()
    df_range = df[(df["date"] >= d0) & (df["date"] <= d1)].copy()

    expected = pd.date_range(d0, d1, freq="D")
    found_unique = pd.DatetimeIndex(df_range["date"].drop_duplicates().sort_values())
    missing = expected.difference(found_unique)
    dup_dates = (
        df_range[df_range["is_duplicate_date"]]["date"].drop_duplicates().sort_values()
    )

    ds_index = xr.Dataset(
        data_vars={
            "path": ("record", df["path"].astype(str).to_numpy(dtype=str)),
            "is_duplicate_date": (
                "record",
                df["is_duplicate_date"].astype(np.int8).to_numpy(),
            ),
        },
        coords={
            "record": np.arange(len(df), dtype=np.int32),
            "date": ("record", df["date"].to_numpy(dtype="datetime64[ns]")),
        },
    )
    ds_index.to_netcdf(
        os.path.join(out_dir, "qc_file_index.nc"),
        encoding={
            "is_duplicate_date": {
                "dtype": "int8",
                "_FillValue": -1,
                "zlib": True,
                "complevel": compression_level,
                "shuffle": True,
            }
        },
    )

    ds_missing = xr.Dataset(
        coords={"missing_date": ("missing_date", missing.to_numpy(dtype="datetime64[ns]"))}
    )
    ds_missing.to_netcdf(os.path.join(out_dir, "qc_missing_dates.nc"))

    ds_duplicates = xr.Dataset(
        coords={
            "duplicate_date": (
                "duplicate_date",
                dup_dates.to_numpy(dtype="datetime64[ns]"),
            )
        }
    )
    ds_duplicates.to_netcdf(os.path.join(out_dir, "qc_duplicate_dates.nc"))

    ds_summary = xr.Dataset(
        data_vars={
            "expected_days": ((), np.int32(len(expected))),
            "files_in_range": ((), np.int32(len(df_range))),
            "unique_days_found": ((), np.int32(len(found_unique))),
            "missing_days": ((), np.int32(len(missing))),
            "duplicate_days": ((), np.int32(len(dup_dates))),
        }
    )
    ds_summary.attrs["start_date"] = d0.date().isoformat()
    ds_summary.attrs["end_date"] = d1.date().isoformat()
    ds_summary.to_netcdf(os.path.join(out_dir, "qc_summary.nc"))

    return {
        "start_date": d0.date().isoformat(),
        "end_date": d1.date().isoformat(),
        "expected_days": int(len(expected)),
        "files_in_range": int(len(df_range)),
        "unique_days_found": int(len(found_unique)),
        "missing_days": int(len(missing)),
        "duplicate_days": int(len(dup_dates)),
        "missing_days_list": ";".join(
            pd.DatetimeIndex(missing).strftime("%Y-%m-%d").tolist()
        ),
        "duplicate_days_list": ";".join(
            pd.DatetimeIndex(dup_dates).strftime("%Y-%m-%d").tolist()
        ),
    }


def write_final_qc_csv(
    output_dir: str,
    qc_dir: str,
    qc_csv_name: str,
    qc_summary: dict,
    num_workers: int,
    wait_timeout_hours: float,
) -> None:
    deadline = time.time() + max(0.0, wait_timeout_hours) * 3600.0
    done_flags = [os.path.join(qc_dir, f"worker_{w}.done") for w in range(num_workers)]
    while True:
        if all(os.path.exists(p) for p in done_flags):
            break
        if time.time() >= deadline:
            break
        time.sleep(30)

    rows = []
    for w in range(num_workers):
        p = os.path.join(qc_dir, f"bad_counts_worker{w}.csv")
        if os.path.exists(p):
            rows.append(pd.read_csv(p))
    if rows:
        bad_df = pd.concat(rows, ignore_index=True)
        nan_count = int(bad_df["nan_count"].sum())
        low_count = int(bad_df["below_low_count"].sum())
        high_count = int(bad_df["above_high_count"].sum())
        total_checked = int(bad_df["total_values_checked"].sum())
        bad_value_count = int(low_count + high_count)
        workers_reported = int(len(bad_df))
    else:
        nan_count = 0
        low_count = 0
        high_count = 0
        total_checked = 0
        bad_value_count = 0
        workers_reported = 0

    out = dict(qc_summary)
    out["workers_expected"] = int(num_workers)
    out["workers_reported"] = workers_reported
    out["qc_bad_counts_complete"] = int(workers_reported == num_workers)
    out["total_values_checked"] = total_checked
    out["nan_count"] = nan_count
    out["below_low_count"] = low_count
    out["above_high_count"] = high_count
    out["bad_value_count"] = bad_value_count

    out_path = os.path.join(output_dir, qc_csv_name)
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f"QC CSV written: {out_path}")


def build_tile_ranges(n: int, step: int) -> List[Tuple[int, int]]:
    out = []
    i = 0
    while i < n:
        j = min(i + step, n)
        out.append((i, j))
        i = j
    return out


def find_bbox_indices(
    lat: np.ndarray,
    lon: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    lon_min, lon_max, lat_min, lat_max = bbox
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise RuntimeError("BBox does not overlap data coordinates.")
    return lat_idx, lon_idx


def to_deg_c(arr: np.ndarray, units: str) -> np.ndarray:
    u = (units or "").lower()
    if ("kelvin" in u) or (u == "k") or (u == ""):
        return arr.astype(np.float32) - 273.15
    return arr.astype(np.float32)


def load_tile_timeseries(
    files: Sequence[str],
    var_name: str,
    lat_name: str,
    lon_name: str,
    il0: int,
    il1: int,
    io0: int,
    io1: int,
    use_progress: bool,
) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    sst_list = []
    times = []
    lat_sub = None
    lon_sub = None
    units = ""

    it = progress_iter(files, enable=use_progress, desc=f"Load tile lat[{il0}:{il1}) lon[{io0}:{io1})")
    for path in it:
        with xr.open_dataset(path, engine="netcdf4", decode_times=True, mask_and_scale=True) as ds:
            if var_name not in ds.variables:
                continue
            t = pd.to_datetime(ds["time"].values[0]).normalize()
            da = ds[var_name].isel(time=0, **{lat_name: slice(il0, il1), lon_name: slice(io0, io1)})
            if lat_sub is None:
                lat_sub = ds[lat_name].values[il0:il1].astype(np.float32)
                lon_sub = ds[lon_name].values[io0:io1].astype(np.float32)
                units = str(ds[var_name].attrs.get("units", ""))
            arr = da.values
            sst_list.append(arr)
            times.append(t)

    if not sst_list:
        raise RuntimeError("Tile has no loaded samples.")

    sst = np.stack(sst_list, axis=0)
    sst = to_deg_c(sst, units=units)
    tix = pd.DatetimeIndex(times)
    order = np.argsort(tix.values)
    sst = sst[order, :, :]
    tix = tix[order]
    return sst, tix, lat_sub, lon_sub


def drop_feb29_np(sst: np.ndarray, times: pd.DatetimeIndex) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    keep = ~((times.month == 2) & (times.day == 29))
    idx = np.where(keep)[0]
    return sst[idx, :, :], times[idx]


def _enc_field_3d(compression_level: int, ch_t: int, ch_lat: int, ch_lon: int) -> dict:
    return {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 0.0,
        "_FillValue": -32768,
        "zlib": True,
        "complevel": compression_level,
        "shuffle": True,
        "chunksizes": (ch_t, ch_lat, ch_lon),
    }


def save_clim_seas_nc_compressed(
    out_path: str,
    clim_doy: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    compression_level: int,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "climatology_c": (("doy", "lat", "lon"), clim_doy.astype(np.float32)),
        },
        coords={"doy": np.arange(1, 366, dtype=np.int16), "lat": lat, "lon": lon},
    )

    ch_lat = min(256, len(lat))
    ch_lon = min(256, len(lon))
    ds.to_netcdf(
        out_path,
        encoding={"climatology_c": _enc_field_3d(compression_level, 31, ch_lat, ch_lon)},
    )


def save_clim_thresh_nc_compressed(
    out_path: str,
    thresh_doy: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    compression_level: int,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "threshold_c": (("doy", "lat", "lon"), thresh_doy.astype(np.float32)),
        },
        coords={"doy": np.arange(1, 366, dtype=np.int16), "lat": lat, "lon": lon},
    )

    ch_lat = min(256, len(lat))
    ch_lon = min(256, len(lon))
    ds.to_netcdf(
        out_path,
        encoding={"threshold_c": _enc_field_3d(compression_level, 31, ch_lat, ch_lon)},
    )


def save_daily_state_nc_compressed(
    out_path: str,
    times: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    anomaly: np.ndarray,
    exceed: np.ndarray,
    mask: np.ndarray,
    compression_level: int,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "anomaly_c": (("time", "lat", "lon"), anomaly.astype(np.float32)),
            "exceedance": (("time", "lat", "lon"), exceed.astype(np.int8)),
            "mhw_mask": (("time", "lat", "lon"), mask.astype(np.int8)),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds["anomaly_c"].attrs["units"] = "degC"

    ch_t = min(31, len(times))
    ch_lat = min(256, len(lat))
    ch_lon = min(256, len(lon))
    enc_f = _enc_field_3d(compression_level, ch_t, ch_lat, ch_lon)
    enc_i = {
        "dtype": "int8",
        "_FillValue": -1,
        "zlib": True,
        "complevel": compression_level,
        "shuffle": True,
        "chunksizes": (ch_t, ch_lat, ch_lon),
    }
    enc = {
        "anomaly_c": dict(enc_f),
        "exceedance": dict(enc_i),
        "mhw_mask": dict(enc_i),
    }
    ds.to_netcdf(out_path, encoding=enc)


def save_event_catalog_nc(
    events: pd.DataFrame,
    out_path: str,
    compression_level: int,
) -> None:
    category_order = ["unknown", "moderate", "strong", "severe", "extreme"]
    cat_to_code = {name: i for i, name in enumerate(category_order)}

    n = len(events)
    event_dim = np.arange(n, dtype=np.int32)
    epoch = np.datetime64("1970-01-01", "D")

    if n == 0:
        ds = xr.Dataset(
            data_vars={
                "event_id": ("event", np.array([], dtype=np.int32)),
                "lat": ("event", np.array([], dtype=np.float32)),
                "lon": ("event", np.array([], dtype=np.float32)),
                "start_day": ("event", np.array([], dtype=np.int32)),
                "end_day": ("event", np.array([], dtype=np.int32)),
                "peak_day": ("event", np.array([], dtype=np.int32)),
                "duration_days": ("event", np.array([], dtype=np.int16)),
                "intensity_max_c": ("event", np.array([], dtype=np.float32)),
                "intensity_mean_c": ("event", np.array([], dtype=np.float32)),
                "intensity_cumulative_cdays": ("event", np.array([], dtype=np.float32)),
                "delta_peak_c": ("event", np.array([], dtype=np.float32)),
                "severity_ratio": ("event", np.array([], dtype=np.float32)),
                "category_code": ("event", np.array([], dtype=np.int8)),
            },
            coords={"event": event_dim},
        )
    else:
        start_day = (
            pd.to_datetime(events["start_date"]).to_numpy(dtype="datetime64[D]") - epoch
        ) / np.timedelta64(1, "D")
        end_day = (
            pd.to_datetime(events["end_date"]).to_numpy(dtype="datetime64[D]") - epoch
        ) / np.timedelta64(1, "D")
        peak_day = (
            pd.to_datetime(events["peak_date"]).to_numpy(dtype="datetime64[D]") - epoch
        ) / np.timedelta64(1, "D")

        category_code = (
            events["category"]
            .fillna("unknown")
            .map(cat_to_code)
            .fillna(cat_to_code["unknown"])
            .astype(np.int8)
            .to_numpy()
        )

        ds = xr.Dataset(
            data_vars={
                "event_id": ("event", events["event_id"].astype(np.int32).to_numpy()),
                "lat": ("event", events["lat"].astype(np.float32).to_numpy()),
                "lon": ("event", events["lon"].astype(np.float32).to_numpy()),
                "start_day": ("event", start_day.astype(np.int32)),
                "end_day": ("event", end_day.astype(np.int32)),
                "peak_day": ("event", peak_day.astype(np.int32)),
                "duration_days": (
                    "event",
                    events["duration_days"].astype(np.int16).to_numpy(),
                ),
                "intensity_max_c": (
                    "event",
                    events["intensity_max_c"].astype(np.float32).to_numpy(),
                ),
                "intensity_mean_c": (
                    "event",
                    events["intensity_mean_c"].astype(np.float32).to_numpy(),
                ),
                "intensity_cumulative_cdays": (
                    "event",
                    events["intensity_cumulative_cdays"].astype(np.float32).to_numpy(),
                ),
                "delta_peak_c": (
                    "event",
                    events["delta_peak_c"].astype(np.float32).to_numpy(),
                ),
                "severity_ratio": (
                    "event",
                    events["severity_ratio"].astype(np.float32).to_numpy(),
                ),
                "category_code": ("event", category_code),
            },
            coords={"event": event_dim},
        )

    ds["start_day"].attrs["units"] = "days since 1970-01-01"
    ds["end_day"].attrs["units"] = "days since 1970-01-01"
    ds["peak_day"].attrs["units"] = "days since 1970-01-01"
    ds["category_code"].attrs["category_table"] = (
        "0:unknown,1:moderate,2:strong,3:severe,4:extreme"
    )

    ch_ev = max(1, min(65536, max(1, n)))
    enc = {
        "event_id": {
            "dtype": "int32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "lat": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "lon": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "start_day": {
            "dtype": "int32",
            "_FillValue": -2147483648,
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "end_day": {
            "dtype": "int32",
            "_FillValue": -2147483648,
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "peak_day": {
            "dtype": "int32",
            "_FillValue": -2147483648,
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "duration_days": {
            "dtype": "int16",
            "_FillValue": -32768,
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "intensity_max_c": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "intensity_mean_c": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "intensity_cumulative_cdays": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "delta_peak_c": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "severity_ratio": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
        "category_code": {
            "dtype": "int8",
            "_FillValue": -1,
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_ev,),
        },
    }
    ds.to_netcdf(out_path, encoding=enc)


def save_sensitivity_rows_nc(
    rows: List[dict],
    out_path: str,
    compression_level: int,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows:
        xr.Dataset(coords={"row": np.array([], dtype=np.int32)}).to_netcdf(out_path)
        return

    df = pd.DataFrame(rows)
    baseline_labels = sorted(df["baseline"].astype(str).drop_duplicates().tolist())
    baseline_to_code = {b: i for i, b in enumerate(baseline_labels)}
    tile_labels = sorted(df["tile_name"].astype(str).drop_duplicates().tolist())
    tile_to_code = {t: i for i, t in enumerate(tile_labels)}

    df["baseline_code"] = df["baseline"].astype(str).map(baseline_to_code).astype(np.int16)
    df["tile_code"] = df["tile_name"].astype(str).map(tile_to_code).astype(np.int32)

    n = len(df)
    row = np.arange(n, dtype=np.int32)
    ds = xr.Dataset(
        data_vars={
            "worker_id": ("row", df["worker_id"].astype(np.int16).to_numpy()),
            "tile_code": ("row", df["tile_code"].astype(np.int32).to_numpy()),
            "baseline_code": ("row", df["baseline_code"].astype(np.int16).to_numpy()),
            "total_event_days": ("row", df["total_event_days"].astype(np.int32).to_numpy()),
            "total_event_starts": (
                "row",
                df["total_event_starts"].astype(np.int32).to_numpy(),
            ),
            "event_day_fraction": (
                "row",
                df["event_day_fraction"].astype(np.float32).to_numpy(),
            ),
            "mean_event_intensity_c": (
                "row",
                df["mean_event_intensity_c"].astype(np.float32).to_numpy(),
            ),
            "mean_event_duration_days": (
                "row",
                df["mean_event_duration_days"].astype(np.float32).to_numpy(),
            ),
        },
        coords={
            "row": row,
            "tile_label": ("tile_code", np.asarray(tile_labels, dtype=str)),
            "baseline_label": ("baseline_code", np.asarray(baseline_labels, dtype=str)),
        },
    )

    ch_row = max(1, min(2048, n))
    enc = {
        "worker_id": {
            "dtype": "int16",
            "_FillValue": -32768,
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_row,),
        },
        "tile_code": {
            "dtype": "int32",
            "_FillValue": -2147483648,
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_row,),
        },
        "baseline_code": {
            "dtype": "int16",
            "_FillValue": -32768,
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_row,),
        },
        "total_event_days": {
            "dtype": "int32",
            "_FillValue": -2147483648,
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_row,),
        },
        "total_event_starts": {
            "dtype": "int32",
            "_FillValue": -2147483648,
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_row,),
        },
        "event_day_fraction": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_row,),
        },
        "mean_event_intensity_c": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_row,),
        },
        "mean_event_duration_days": {
            "dtype": "float32",
            "zlib": True,
            "complevel": compression_level,
            "shuffle": True,
            "chunksizes": (ch_row,),
        },
    }
    ds.to_netcdf(out_path, encoding=enc)


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if (args.worker_id < 0) or (args.worker_id >= args.num_workers):
        raise ValueError("--worker-id must be in [0, num_workers)")
    if (args.compression_level < 0) or (args.compression_level > 9):
        raise ValueError("--compression-level must be in [0, 9]")

    use_progress = not args.no_progress
    baseline = parse_baseline_window(args.baseline_main)
    sensitivity_labels = parse_sensitivity_windows(
        args.baseline_sensitivity, main_label=baseline.label
    )

    os.makedirs(args.output_dir, exist_ok=True)
    tiles_dir = os.path.join(args.output_dir, "tiles")
    qc_dir = os.path.join(args.output_dir, "qc")
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)
    sens_dir = os.path.join(args.output_dir, "sensitivity")
    if sensitivity_labels:
        os.makedirs(sens_dir, exist_ok=True)

    qc_summary: dict = {}
    if args.worker_id == 0:
        qc_summary = write_qc_reports(
            input_dir=args.input_dir,
            file_glob=args.file_glob,
            start_date=args.start_date,
            end_date=args.end_date,
            out_dir=qc_dir,
            compression_level=args.compression_level,
        )

    df = discover_files(
        input_dir=args.input_dir,
        file_glob=args.file_glob,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    files = df["path"].tolist()

    with xr.open_dataset(files[0], engine="netcdf4") as ds0:
        lat_all = ds0[args.lat_name].values
        lon_all = ds0[args.lon_name].values

    bbox = parse_bbox(args.bbox)
    lon_min_req, lon_max_req, lat_min_req, lat_max_req = bbox
    lon_min_avail = float(np.nanmin(lon_all))
    lon_max_avail = float(np.nanmax(lon_all))
    lat_min_avail = float(np.nanmin(lat_all))
    lat_max_avail = float(np.nanmax(lat_all))
    if (
        (lon_min_req < lon_min_avail)
        or (lon_max_req > lon_max_avail)
        or (lat_min_req < lat_min_avail)
        or (lat_max_req > lat_max_avail)
    ):
        print(
            "Requested bbox exceeds available GHRSST grid; clipping to available range "
            f"lon[{lon_min_avail},{lon_max_avail}] lat[{lat_min_avail},{lat_max_avail}]"
        )
    lat_idx, lon_idx = find_bbox_indices(lat_all, lon_all, bbox)
    lat_ranges = build_tile_ranges(len(lat_idx), args.tile_lat)
    lon_ranges = build_tile_ranges(len(lon_idx), args.tile_lon)

    tile_jobs = []
    for ia0, ia1 in lat_ranges:
        for io0, io1 in lon_ranges:
            il0 = int(lat_idx[ia0])
            il1 = int(lat_idx[ia1 - 1]) + 1
            jl0 = int(lon_idx[io0])
            jl1 = int(lon_idx[io1 - 1]) + 1
            tile_jobs.append((ia0, ia1, io0, io1, il0, il1, jl0, jl1))

    total_tiles = len(tile_jobs)
    if args.num_workers > 1:
        tile_jobs = [
            job
            for idx, job in enumerate(tile_jobs)
            if (idx % args.num_workers) == args.worker_id
        ]

    print(
        f"Worker {args.worker_id + 1}/{args.num_workers} assigned "
        f"{len(tile_jobs)} of {total_tiles} total tiles"
    )
    print(f"Files in date range: {len(files)}")
    print(f"Main baseline: {baseline.label}")
    if sensitivity_labels:
        print("Sensitivity baselines: " + ", ".join(sensitivity_labels))
    else:
        print("Sensitivity baselines: none")
    print(
        "Enabled outputs: "
        f"clim/threshold={int(args.save_clim_threshold)}, "
        f"daily_state={int(args.save_daily_state)}, "
        f"annual_trend={int(args.save_annual_trend)}, "
        f"events={int(args.save_events)}"
    )

    job_it = progress_iter(
        tile_jobs,
        enable=use_progress,
        desc=f"Tiles w{args.worker_id + 1}/{args.num_workers}",
    )
    sens_nc = ""
    sens_rows: List[dict] = []
    if sensitivity_labels:
        sens_nc = os.path.join(
            sens_dir, f"baseline_sensitivity_worker{args.worker_id}.nc"
        )

    worker_nan = 0
    worker_low = 0
    worker_high = 0
    worker_total = 0

    for (ia0, ia1, io0, io1, il0, il1, jl0, jl1) in job_it:
        tile_name = f"tile_lat{il0}-{il1}_lon{jl0}-{jl1}"
        done_flag = os.path.join(tiles_dir, f"{tile_name}.done")
        if os.path.exists(done_flag):
            continue

        sst, times, lat_sub, lon_sub = load_tile_timeseries(
            files=files,
            var_name=args.var_name,
            lat_name=args.lat_name,
            lon_name=args.lon_name,
            il0=il0,
            il1=il1,
            io0=jl0,
            io1=jl1,
            use_progress=use_progress,
        )

        worker_total += int(sst.size)
        worker_nan += int(np.isnan(sst).sum(dtype=np.int64))
        worker_low += int(np.sum(sst < args.bad_low_c, dtype=np.int64))
        worker_high += int(np.sum(sst > args.bad_high_c, dtype=np.int64))
        sst, times = drop_feb29_np(sst, times)

        res = run_baseline_detection(
            sst=sst,
            times=times,
            baseline=baseline,
            pctile=args.pctile,
            half_window=args.window_half_width,
            smooth_width=args.smooth_percentile_width,
            min_duration=args.min_duration,
            max_gap=args.max_gap,
            use_progress=use_progress,
        )

        if args.save_clim_threshold:
            clim_nc = os.path.join(
                tiles_dir,
                f"{tile_name}_clim_seas_{baseline.label}.nc",
            )
            thresh_nc = os.path.join(
                tiles_dir,
                f"{tile_name}_clim_thresh_p{int(args.pctile)}_{baseline.label}.nc",
            )
            save_clim_seas_nc_compressed(
                out_path=clim_nc,
                clim_doy=res["clim_doy"],
                lat=lat_sub,
                lon=lon_sub,
                compression_level=args.compression_level,
            )
            save_clim_thresh_nc_compressed(
                out_path=thresh_nc,
                thresh_doy=res["thresh_doy"],
                lat=lat_sub,
                lon=lon_sub,
                compression_level=args.compression_level,
            )

        if args.save_daily_state:
            daily_nc = os.path.join(
                tiles_dir,
                f"{tile_name}_mhw_daily_state_{args.start_date[:4]}_{args.end_date[:4]}.nc",
            )
            save_daily_state_nc_compressed(
                out_path=daily_nc,
                times=times,
                lat=lat_sub,
                lon=lon_sub,
                anomaly=res["anomaly"],
                exceed=res["exceed"],
                mask=res["mask"],
                compression_level=args.compression_level,
            )

        if args.save_annual_trend:
            years, metrics = compute_annual_metrics(
                mask=res["mask"],
                starts=res["starts"],
                anomaly=res["anomaly"],
                times=times,
            )

            annual_nc = os.path.join(tiles_dir, f"{tile_name}_annual_metrics.nc")
            trend_nc = os.path.join(tiles_dir, f"{tile_name}_trend_maps.nc")

            save_annual_metrics_nc(
                out_path=annual_nc,
                years=years,
                lat=lat_sub,
                lon=lon_sub,
                metrics=metrics,
            )
            save_trend_nc(
                out_path=trend_nc,
                lat=lat_sub,
                lon=lon_sub,
                metrics=metrics,
                years=years,
                use_progress=use_progress,
            )

        if args.save_events:
            events = build_event_catalog(
                mask=res["mask"],
                starts=res["starts"],
                anomaly=res["anomaly"],
                delta_daily=res["delta_daily"],
                times=times,
                lat=lat_sub,
                lon=lon_sub,
                use_progress=use_progress,
            )
            save_event_catalog_nc(
                events=events,
                out_path=os.path.join(tiles_dir, f"{tile_name}_events.nc"),
                compression_level=args.compression_level,
            )

        if sensitivity_labels:
            main_summary = summarize_baseline(
                baseline=baseline,
                mask=res["mask"],
                starts=res["starts"],
                anomaly=res["anomaly"],
            )
            main_summary["worker_id"] = args.worker_id
            main_summary["tile_name"] = tile_name
            sens_rows.append(main_summary)

            for sens_label in sensitivity_labels:
                sens_b = parse_baseline_window(sens_label)
                sens_res = run_baseline_detection(
                    sst=sst,
                    times=times,
                    baseline=sens_b,
                    pctile=args.pctile,
                    half_window=args.window_half_width,
                    smooth_width=args.smooth_percentile_width,
                    min_duration=args.min_duration,
                    max_gap=args.max_gap,
                    use_progress=use_progress,
                )
                sens_summary = summarize_baseline(
                    baseline=sens_b,
                    mask=sens_res["mask"],
                    starts=sens_res["starts"],
                    anomaly=sens_res["anomaly"],
                )
                sens_summary["worker_id"] = args.worker_id
                sens_summary["tile_name"] = tile_name
                sens_rows.append(sens_summary)

            save_sensitivity_rows_nc(
                rows=sens_rows,
                out_path=sens_nc,
                compression_level=args.compression_level,
            )

        with open(done_flag, "w", encoding="utf-8") as f:
            f.write("ok\n")

    worker_bad_path = os.path.join(qc_dir, f"bad_counts_worker{args.worker_id}.csv")
    pd.DataFrame(
        [
            {
                "worker_id": int(args.worker_id),
                "total_values_checked": int(worker_total),
                "nan_count": int(worker_nan),
                "below_low_count": int(worker_low),
                "above_high_count": int(worker_high),
                "bad_value_count": int(worker_low + worker_high),
            }
        ]
    ).to_csv(worker_bad_path, index=False)
    with open(os.path.join(qc_dir, f"worker_{args.worker_id}.done"), "w", encoding="utf-8") as f:
        f.write("ok\n")

    if args.worker_id == 0:
        qc_summary["bad_low_c"] = float(args.bad_low_c)
        qc_summary["bad_high_c"] = float(args.bad_high_c)
        write_final_qc_csv(
            output_dir=args.output_dir,
            qc_dir=qc_dir,
            qc_csv_name=args.qc_csv_name,
            qc_summary=qc_summary,
            num_workers=args.num_workers,
            wait_timeout_hours=args.qc_wait_timeout_hours,
        )

    print("Native chunked run finished.")


if __name__ == "__main__":
    main()
