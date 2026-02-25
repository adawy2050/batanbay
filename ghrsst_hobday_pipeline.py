#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GHRSST Hobday-style marine heatwave pipeline (2003-2025 ready).

This script implements a reproducible heatwave workflow designed to be coupled
later with typhoon tracks:

1) QC/inventory from filenames
2) Hobday-style climatology + threshold fields
3) Daily MHW state fields
4) Grid-cell event catalog
5) Annual MHW metrics
6) Trend maps
7) Baseline sensitivity summary

Method choices:
- Core event definition follows Hobday et al. (2016):
  percentile threshold + min 5-day duration + max 2-day gap merge.
- Optional category labels follow Hobday et al. (2018)-style exceedance ratios.
- Default processing target is Japan-region with coarsening; global native MUR
  resolution is usually too large for one-shot in-memory analysis.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


def progress_iter(iterable, enable: bool, **kwargs):
    if enable and (_tqdm is not None):
        return _tqdm(iterable, **kwargs)
    return iterable


@dataclass(frozen=True)
class BaselineWindow:
    start: int
    end: int

    @property
    def label(self) -> str:
        return f"{self.start}-{self.end}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GHRSST Hobday marine heatwave pipeline with full outputs."
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing daily GHRSST NetCDF files.",
    )
    p.add_argument(
        "--file-glob",
        default="*JPL-L4_GHRSST-SSTfnd-MUR-GLOB*.nc4",
        help="Filename glob pattern under --input-dir.",
    )
    p.add_argument(
        "--start-date",
        default="2003-01-01",
        help="Analysis start date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--end-date",
        default="2025-12-31",
        help="Analysis end date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--var-name",
        default="analysed_sst",
        help="SST variable name in GHRSST files.",
    )
    p.add_argument(
        "--lat-name",
        default="lat",
        help="Latitude coordinate name.",
    )
    p.add_argument(
        "--lon-name",
        default="lon",
        help="Longitude coordinate name.",
    )
    p.add_argument(
        "--bbox",
        default="100,180,0,70",
        help=(
            "lon_min,lon_max,lat_min,lat_max (degrees East if dataset uses 0..360). "
            "Default matches typhoon coverage: 100E-180E, 0N-70N."
        ),
    )
    p.add_argument(
        "--coarsen-lat",
        type=int,
        default=25,
        help="Latitude coarsening factor (native MUR is very high resolution).",
    )
    p.add_argument(
        "--coarsen-lon",
        type=int,
        default=25,
        help="Longitude coarsening factor (native MUR is very high resolution).",
    )
    p.add_argument(
        "--pctile",
        type=float,
        default=90.0,
        help="Percentile threshold for MHW definition.",
    )
    p.add_argument(
        "--window-half-width",
        type=int,
        default=5,
        help="Half-width in days around DOY for threshold/climatology pooling.",
    )
    p.add_argument(
        "--smooth-percentile-width",
        type=int,
        default=31,
        help="Circular smoothing width for percentile threshold (set 1 to disable).",
    )
    p.add_argument(
        "--min-duration",
        type=int,
        default=5,
        help="Minimum duration (days) for an event.",
    )
    p.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="Maximum gap (days) for merging adjacent warm segments.",
    )
    p.add_argument(
        "--baseline-main",
        default="2003-2022",
        help="Primary baseline window for full output production.",
    )
    p.add_argument(
        "--baseline-sensitivity",
        default="2004-2023;2005-2024",
        help="Semicolon-separated extra baseline windows for sensitivity summary.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for all products.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return p.parse_args()


def parse_baseline_window(text: str) -> BaselineWindow:
    m = re.fullmatch(r"\s*(\d{4})\s*-\s*(\d{4})\s*", text)
    if not m:
        raise ValueError(f"Invalid baseline window: {text!r}")
    y0 = int(m.group(1))
    y1 = int(m.group(2))
    if y1 < y0:
        raise ValueError(f"Invalid baseline range (end<start): {text!r}")
    return BaselineWindow(y0, y1)


def parse_bbox(text: str) -> Tuple[float, float, float, float]:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be lon_min,lon_max,lat_min,lat_max")
    lon_min, lon_max, lat_min, lat_max = map(float, parts)
    if lat_max <= lat_min:
        raise ValueError("bbox requires lat_max > lat_min")
    return lon_min, lon_max, lat_min, lat_max


def parse_yyyymmdd_from_filename(path: str) -> pd.Timestamp | None:
    name = os.path.basename(path)
    m = re.match(r"^(\d{8})\d{6}-", name)
    if not m:
        return None
    try:
        return pd.to_datetime(m.group(1), format="%Y%m%d")
    except Exception:
        return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_key_value_csv(path: str, rows: Sequence[Tuple[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in rows:
            w.writerow([k, v])


def discover_files(input_dir: str, file_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(input_dir, file_glob)))
    if not paths:
        raise FileNotFoundError(
            f"No files found for pattern {file_glob!r} under {input_dir!r}"
        )
    records = []
    for p in paths:
        dt = parse_yyyymmdd_from_filename(p)
        records.append({"path": p, "date": dt})
    df = pd.DataFrame(records)
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        raise RuntimeError("No parseable GHRSST dates from filenames.")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "path"]).reset_index(drop=True)
    return df


def build_qc_reports(
    file_df: pd.DataFrame, start_date: str, end_date: str, out_dir: str
) -> pd.DataFrame:
    ensure_dir(out_dir)
    df = file_df.copy()
    df["is_duplicate_date"] = df.duplicated(subset=["date"], keep=False)
    df.to_csv(os.path.join(out_dir, "qc_file_index.csv"), index=False)

    d0 = pd.to_datetime(start_date).normalize()
    d1 = pd.to_datetime(end_date).normalize()
    df_range = df[(df["date"] >= d0) & (df["date"] <= d1)].copy()

    expected = pd.date_range(d0, d1, freq="D")
    found_unique = pd.DatetimeIndex(df_range["date"].drop_duplicates().sort_values())
    missing = expected.difference(found_unique)

    dup_dates = (
        df_range[df_range["is_duplicate_date"]]["date"].drop_duplicates().sort_values()
    )
    pd.DataFrame({"missing_date": missing}).to_csv(
        os.path.join(out_dir, "qc_missing_dates.csv"), index=False
    )
    pd.DataFrame({"duplicate_date": dup_dates}).to_csv(
        os.path.join(out_dir, "qc_duplicate_dates.csv"), index=False
    )

    rows = [
        ("start_date", d0.date().isoformat()),
        ("end_date", d1.date().isoformat()),
        ("expected_days", len(expected)),
        ("files_in_range", len(df_range)),
        ("unique_days_found", len(found_unique)),
        ("missing_days", len(missing)),
        ("duplicate_days", len(dup_dates)),
    ]
    write_key_value_csv(os.path.join(out_dir, "qc_summary.csv"), rows)
    return df_range.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)


def convert_sst_to_c(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower()
    if units in ("k", "kelvin") or ("kelvin" in units):
        out = da - 273.15
    elif "c" in units:
        out = da
    else:
        # GHRSST analysed_sst is typically Kelvin if units are missing/ambiguous.
        out = da - 273.15
    out = out.astype("float32")
    out.attrs["units"] = "degC"
    return out


def normalize_lon_for_dataset(lon_val: float, lon_coord: xr.DataArray) -> float:
    lon_min = float(lon_coord.min().values)
    lon_max = float(lon_coord.max().values)
    if lon_min >= 0 and lon_max > 180:
        return lon_val % 360.0
    if lon_val > 180:
        return ((lon_val + 180.0) % 360.0) - 180.0
    return lon_val


def subset_bbox(
    da: xr.DataArray,
    lon_name: str,
    lat_name: str,
    bbox: Tuple[float, float, float, float],
) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_min = normalize_lon_for_dataset(lon_min, da[lon_name])
    lon_max = normalize_lon_for_dataset(lon_max, da[lon_name])

    da = da.sortby(lat_name)
    if lon_min <= lon_max:
        da = da.sel({lon_name: slice(lon_min, lon_max)})
    else:
        left = da.sel({lon_name: slice(lon_min, None)})
        right = da.sel({lon_name: slice(None, lon_max)})
        da = xr.concat([left, right], dim=lon_name)
    da = da.sel({lat_name: slice(lat_min, lat_max)})
    return da


def load_sst_data(
    files: Sequence[str],
    var_name: str,
    lon_name: str,
    lat_name: str,
    start_date: str,
    end_date: str,
    bbox: Tuple[float, float, float, float],
    coarsen_lat: int,
    coarsen_lon: int,
    use_progress: bool,
) -> xr.DataArray:
    if len(files) == 0:
        raise RuntimeError("No files provided to loader.")

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    data_list: List[np.ndarray] = []
    time_list: List[pd.Timestamp] = []
    lat_vals = None
    lon_vals = None

    it = progress_iter(files, enable=use_progress, desc="Read/subset/coarsen files")
    for path in it:
        with xr.open_dataset(
            path, engine="netcdf4", decode_times=True, mask_and_scale=True
        ) as ds:
            if var_name not in ds.variables:
                continue
            if "time" not in ds.coords:
                continue

            t = pd.to_datetime(ds["time"].values[0])
            if (t < start_ts) or (t > end_ts):
                continue

            da = ds[var_name].isel(time=0)
            da = convert_sst_to_c(da)
            da = subset_bbox(da, lon_name=lon_name, lat_name=lat_name, bbox=bbox)

            if coarsen_lat > 1 or coarsen_lon > 1:
                da = da.coarsen(
                    {lat_name: coarsen_lat, lon_name: coarsen_lon}, boundary="trim"
                ).mean()

            if lat_vals is None:
                lat_vals = da[lat_name].values.astype(np.float32)
                lon_vals = da[lon_name].values.astype(np.float32)

            arr = da.values.astype(np.float32, copy=False)
            data_list.append(arr)
            time_list.append(t.normalize())

    if len(data_list) == 0:
        raise RuntimeError("No SST samples loaded after filtering.")

    sst = np.stack(data_list, axis=0)
    time_arr = pd.DatetimeIndex(time_list)

    order = np.argsort(time_arr.values)
    sst = sst[order, :, :]
    time_arr = time_arr[order]

    da_out = xr.DataArray(
        sst,
        dims=("time", "lat", "lon"),
        coords={"time": time_arr, "lat": lat_vals, "lon": lon_vals},
        name="sst_c",
    )
    return da_out


def drop_feb29(da: xr.DataArray) -> xr.DataArray:
    t = pd.DatetimeIndex(da["time"].values)
    keep = ~((t.month == 2) & (t.day == 29))
    return da.isel(time=np.where(keep)[0])


def doy_noleap(times: pd.DatetimeIndex) -> np.ndarray:
    doy = times.dayofyear.to_numpy().astype(np.int16)
    adjust = ((times.is_leap_year) & (times.month > 2)).astype(np.int16)
    return doy - adjust


def compute_clim_thresh_doy(
    sst: np.ndarray,
    times: pd.DatetimeIndex,
    baseline: BaselineWindow,
    pctile: float,
    half_window: int,
    smooth_width: int,
    use_progress: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = times.year.to_numpy()
    bmask = (years >= baseline.start) & (years <= baseline.end)
    if bmask.sum() < 365:
        raise RuntimeError(
            f"Baseline {baseline.label} has too few samples: {int(bmask.sum())}"
        )

    sst_b = sst[bmask, :, :]
    doy_b = doy_noleap(times[bmask])

    _, ny, nx = sst_b.shape
    clim = np.full((365, ny, nx), np.nan, dtype=np.float32)
    thresh = np.full((365, ny, nx), np.nan, dtype=np.float32)

    for d in progress_iter(
        range(1, 366),
        enable=use_progress,
        desc=f"DOY climatology ({baseline.label})",
    ):
        dist = np.abs(doy_b - d)
        circ = np.minimum(dist, 365 - dist)
        wmask = circ <= half_window
        vals = sst_b[wmask, :, :]
        if vals.shape[0] == 0:
            continue
        clim[d - 1, :, :] = np.nanmean(vals, axis=0).astype(np.float32)
        thresh[d - 1, :, :] = np.nanpercentile(vals, pctile, axis=0).astype(np.float32)

    if smooth_width > 1:
        thresh = uniform_filter1d(
            thresh, size=smooth_width, axis=0, mode="wrap"
        ).astype(np.float32)

    delta = (thresh - clim).astype(np.float32)
    return clim, thresh, delta


def map_doy_fields_to_daily(
    clim_doy: np.ndarray,
    thresh_doy: np.ndarray,
    delta_doy: np.ndarray,
    times: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    doy = doy_noleap(times)
    idx = doy - 1
    clim_daily = clim_doy[idx, :, :]
    thresh_daily = thresh_doy[idx, :, :]
    delta_daily = delta_doy[idx, :, :]
    return clim_daily, thresh_daily, delta_daily


def detect_mask_and_starts(
    exceed: np.ndarray,
    min_duration: int,
    max_gap: int,
    use_progress: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    tlen, ny, nx = exceed.shape
    npts = ny * nx
    exc2 = exceed.reshape(tlen, npts)
    mask2 = np.zeros_like(exc2, dtype=bool)
    starts2 = np.zeros_like(exc2, dtype=bool)

    for p in progress_iter(
        range(npts), enable=use_progress, desc="Detect events per grid"
    ):
        s = exc2[:, p]
        true_idx = np.flatnonzero(s)
        if true_idx.size == 0:
            continue

        run_starts: List[int] = [int(true_idx[0])]
        run_ends: List[int] = []
        prev = int(true_idx[0])
        for idx in true_idx[1:]:
            ii = int(idx)
            if ii != prev + 1:
                run_ends.append(prev)
                run_starts.append(ii)
            prev = ii
        run_ends.append(prev)

        merged_starts: List[int] = [run_starts[0]]
        merged_ends: List[int] = [run_ends[0]]
        for s0, e0 in zip(run_starts[1:], run_ends[1:]):
            gap = s0 - merged_ends[-1] - 1
            if gap <= max_gap:
                merged_ends[-1] = e0
            else:
                merged_starts.append(s0)
                merged_ends.append(e0)

        for s0, e0 in zip(merged_starts, merged_ends):
            dur = e0 - s0 + 1
            if dur >= min_duration:
                mask2[s0 : e0 + 1, p] = True
                starts2[s0, p] = True

    return mask2.reshape(tlen, ny, nx), starts2.reshape(tlen, ny, nx)


def category_from_peak_ratio(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "unknown"
    if ratio < 2.0:
        return "moderate"
    if ratio < 3.0:
        return "strong"
    if ratio < 4.0:
        return "severe"
    return "extreme"


def build_event_catalog(
    mask: np.ndarray,
    starts: np.ndarray,
    anomaly: np.ndarray,
    delta_daily: np.ndarray,
    times: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    use_progress: bool,
) -> pd.DataFrame:
    tlen, ny, nx = mask.shape
    npts = ny * nx
    mask2 = mask.reshape(tlen, npts)
    starts2 = starts.reshape(tlen, npts)
    anom2 = anomaly.reshape(tlen, npts)
    delta2 = delta_daily.reshape(tlen, npts)

    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)

    rows = []
    ev_id = 1
    for p in progress_iter(
        range(npts), enable=use_progress, desc="Build event catalog"
    ):
        sidx = np.flatnonzero(starts2[:, p])
        if sidx.size == 0:
            continue
        for s0 in sidx:
            e0 = int(s0)
            while (e0 + 1) < tlen and mask2[e0 + 1, p]:
                e0 += 1
            seg = anom2[s0 : e0 + 1, p]
            if seg.size == 0:
                continue
            if np.all(~np.isfinite(seg)):
                continue
            peak_rel = int(np.nanargmax(seg))
            peak_idx = int(s0 + peak_rel)
            peak_val = float(seg[peak_rel])
            mean_val = float(np.nanmean(seg))
            cum_val = float(np.nansum(seg))
            dpk = float(delta2[peak_idx, p])
            ratio = peak_val / dpk if np.isfinite(dpk) and dpk > 0 else np.nan

            rows.append(
                {
                    "event_id": ev_id,
                    "lat": float(lat_flat[p]),
                    "lon": float(lon_flat[p]),
                    "start_date": times[int(s0)].date().isoformat(),
                    "end_date": times[e0].date().isoformat(),
                    "peak_date": times[peak_idx].date().isoformat(),
                    "duration_days": int(e0 - int(s0) + 1),
                    "intensity_max_c": peak_val,
                    "intensity_mean_c": mean_val,
                    "intensity_cumulative_cdays": cum_val,
                    "delta_peak_c": dpk,
                    "severity_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
                    "category": category_from_peak_ratio(ratio),
                }
            )
            ev_id += 1

    return pd.DataFrame(rows)


def compute_annual_metrics(
    mask: np.ndarray,
    starts: np.ndarray,
    anomaly: np.ndarray,
    times: pd.DatetimeIndex,
) -> Tuple[np.ndarray, dict]:
    years = times.year.to_numpy()
    uniq_years = np.array(sorted(np.unique(years)), dtype=np.int16)
    ny, nx = mask.shape[1], mask.shape[2]
    nyr = len(uniq_years)

    out = {
        "mhw_days": np.full((nyr, ny, nx), np.nan, dtype=np.float32),
        "mhw_frequency": np.full((nyr, ny, nx), np.nan, dtype=np.float32),
        "mhw_mean_duration": np.full((nyr, ny, nx), np.nan, dtype=np.float32),
        "mhw_mean_intensity": np.full((nyr, ny, nx), np.nan, dtype=np.float32),
        "mhw_max_intensity": np.full((nyr, ny, nx), np.nan, dtype=np.float32),
        "mhw_cumulative_intensity": np.full((nyr, ny, nx), np.nan, dtype=np.float32),
    }

    for i, yy in enumerate(uniq_years):
        idx = np.where(years == yy)[0]
        m = mask[idx, :, :]
        s = starts[idx, :, :]
        a = anomaly[idx, :, :]

        days = m.sum(axis=0).astype(np.float32)
        freq = s.sum(axis=0).astype(np.float32)
        mean_dur = np.where(freq > 0, days / freq, np.nan).astype(np.float32)

        event_vals = np.where(m, a, np.nan)
        mean_int = np.nanmean(event_vals, axis=0).astype(np.float32)
        with np.errstate(all="ignore"):
            max_int = np.nanmax(event_vals, axis=0).astype(np.float32)
        max_int[~np.isfinite(max_int)] = np.nan
        cum_int = np.nansum(event_vals, axis=0).astype(np.float32)

        out["mhw_days"][i, :, :] = days
        out["mhw_frequency"][i, :, :] = freq
        out["mhw_mean_duration"][i, :, :] = mean_dur
        out["mhw_mean_intensity"][i, :, :] = mean_int
        out["mhw_max_intensity"][i, :, :] = max_int
        out["mhw_cumulative_intensity"][i, :, :] = cum_int

    return uniq_years, out


def trend_slope_pvalue(
    data: np.ndarray, years: np.ndarray, use_progress: bool
) -> Tuple[np.ndarray, np.ndarray]:
    nyr, ny, nx = data.shape
    d2 = data.reshape(nyr, ny * nx)
    slope = np.full((ny * nx,), np.nan, dtype=np.float32)
    pval = np.full((ny * nx,), np.nan, dtype=np.float32)
    x = years.astype(float)
    for p in progress_iter(
        range(d2.shape[1]), enable=use_progress, desc="Trend per grid"
    ):
        y = d2[:, p]
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        lr = linregress(x[ok], y[ok])
        slope[p] = np.float32(lr.slope)
        pval[p] = np.float32(lr.pvalue)
    return slope.reshape(ny, nx), pval.reshape(ny, nx)


def save_clim_threshold_nc(
    out_path: str,
    clim_doy: np.ndarray,
    thresh_doy: np.ndarray,
    delta_doy: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> None:
    doy = np.arange(1, 366, dtype=np.int16)
    ds = xr.Dataset(
        data_vars={
            "climatology_c": (("doy", "lat", "lon"), clim_doy),
            "threshold_c": (("doy", "lat", "lon"), thresh_doy),
            "delta_c": (("doy", "lat", "lon"), delta_doy),
        },
        coords={"doy": doy, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(out_path)


def save_daily_state_nc(
    out_path: str,
    times: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    sst: np.ndarray,
    clim_daily: np.ndarray,
    thresh_daily: np.ndarray,
    anomaly: np.ndarray,
    exceed: np.ndarray,
    mask: np.ndarray,
    starts: np.ndarray,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "sst_c": (("time", "lat", "lon"), sst.astype(np.float32)),
            "climatology_c": (("time", "lat", "lon"), clim_daily.astype(np.float32)),
            "threshold_c": (("time", "lat", "lon"), thresh_daily.astype(np.float32)),
            "anomaly_c": (("time", "lat", "lon"), anomaly.astype(np.float32)),
            "exceedance": (("time", "lat", "lon"), exceed.astype(np.int8)),
            "mhw_mask": (("time", "lat", "lon"), mask.astype(np.int8)),
            "event_start": (("time", "lat", "lon"), starts.astype(np.int8)),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(out_path)


def save_annual_metrics_nc(
    out_path: str, years: np.ndarray, lat: np.ndarray, lon: np.ndarray, metrics: dict
) -> None:
    ds = xr.Dataset(
        data_vars={
            k: (("year", "lat", "lon"), v.astype(np.float32)) for k, v in metrics.items()
        },
        coords={"year": years.astype(np.int16), "lat": lat, "lon": lon},
    )
    ds.to_netcdf(out_path)


def save_trend_nc(
    out_path: str,
    lat: np.ndarray,
    lon: np.ndarray,
    metrics: dict,
    years: np.ndarray,
    use_progress: bool,
) -> None:
    data_vars = {}
    for name, arr in metrics.items():
        slope, pval = trend_slope_pvalue(arr, years, use_progress=use_progress)
        data_vars[f"{name}_slope_per_year"] = (("lat", "lon"), slope.astype(np.float32))
        data_vars[f"{name}_pvalue"] = (("lat", "lon"), pval.astype(np.float32))
    ds = xr.Dataset(data_vars=data_vars, coords={"lat": lat, "lon": lon})
    ds.to_netcdf(out_path)


def summarize_baseline(
    baseline: BaselineWindow,
    mask: np.ndarray,
    starts: np.ndarray,
    anomaly: np.ndarray,
) -> dict:
    event_vals = np.where(mask, anomaly, np.nan)
    total_days = int(mask.sum())
    total_starts = int(starts.sum())
    mean_int = float(np.nanmean(event_vals)) if np.any(np.isfinite(event_vals)) else np.nan
    frac = float(mask.mean())
    mean_dur = (float(total_days) / float(total_starts)) if total_starts > 0 else np.nan
    return {
        "baseline": baseline.label,
        "total_event_days": total_days,
        "total_event_starts": total_starts,
        "event_day_fraction": frac,
        "mean_event_intensity_c": mean_int,
        "mean_event_duration_days": mean_dur,
    }


def run_baseline_detection(
    sst: np.ndarray,
    times: pd.DatetimeIndex,
    baseline: BaselineWindow,
    pctile: float,
    half_window: int,
    smooth_width: int,
    min_duration: int,
    max_gap: int,
    use_progress: bool,
) -> dict:
    clim_doy, thresh_doy, delta_doy = compute_clim_thresh_doy(
        sst=sst,
        times=times,
        baseline=baseline,
        pctile=pctile,
        half_window=half_window,
        smooth_width=smooth_width,
        use_progress=use_progress,
    )
    clim_daily, thresh_daily, delta_daily = map_doy_fields_to_daily(
        clim_doy, thresh_doy, delta_doy, times
    )
    anomaly = (sst - clim_daily).astype(np.float32)
    exceed = sst > thresh_daily
    mask, starts = detect_mask_and_starts(
        exceed=exceed,
        min_duration=min_duration,
        max_gap=max_gap,
        use_progress=use_progress,
    )
    return {
        "clim_doy": clim_doy,
        "thresh_doy": thresh_doy,
        "delta_doy": delta_doy,
        "clim_daily": clim_daily,
        "thresh_daily": thresh_daily,
        "delta_daily": delta_daily,
        "anomaly": anomaly,
        "exceed": exceed,
        "mask": mask,
        "starts": starts,
    }


def main() -> None:
    args = parse_args()
    use_progress = not args.no_progress
    ensure_dir(args.output_dir)
    qc_dir = os.path.join(args.output_dir, "qc")
    out_main = os.path.join(args.output_dir, "main")
    out_sens = os.path.join(args.output_dir, "sensitivity")
    ensure_dir(qc_dir)
    ensure_dir(out_main)
    ensure_dir(out_sens)

    main_baseline = parse_baseline_window(args.baseline_main)
    sens_windows = []
    if args.baseline_sensitivity.strip():
        for part in args.baseline_sensitivity.split(";"):
            txt = part.strip()
            if txt:
                sens_windows.append(parse_baseline_window(txt))
    all_baselines = [main_baseline] + [b for b in sens_windows if b != main_baseline]

    bbox = parse_bbox(args.bbox)

    print("[1/8] Discovering files...")
    file_df = discover_files(args.input_dir, args.file_glob)
    use_df = build_qc_reports(file_df, args.start_date, args.end_date, qc_dir)

    print("[2/8] Loading GHRSST subset...")
    sst_da = load_sst_data(
        files=use_df["path"].tolist(),
        var_name=args.var_name,
        lon_name=args.lon_name,
        lat_name=args.lat_name,
        start_date=args.start_date,
        end_date=args.end_date,
        bbox=bbox,
        coarsen_lat=args.coarsen_lat,
        coarsen_lon=args.coarsen_lon,
        use_progress=use_progress,
    )
    sst_da = drop_feb29(sst_da)
    sst_da = sst_da.sortby("time")

    times = pd.DatetimeIndex(sst_da["time"].values)
    lat = sst_da[args.lat_name].values.astype(np.float32)
    lon = sst_da[args.lon_name].values.astype(np.float32)
    sst = sst_da.values.astype(np.float32)

    write_key_value_csv(
        os.path.join(qc_dir, "loaded_domain_summary.csv"),
        [
            ("n_time", int(sst.shape[0])),
            ("n_lat", int(sst.shape[1])),
            ("n_lon", int(sst.shape[2])),
            ("time_min", str(times.min().date())),
            ("time_max", str(times.max().date())),
            ("lat_min", float(np.nanmin(lat))),
            ("lat_max", float(np.nanmax(lat))),
            ("lon_min", float(np.nanmin(lon))),
            ("lon_max", float(np.nanmax(lon))),
        ],
    )

    print(f"[3/8] Running main baseline detection ({main_baseline.label})...")
    main_res = run_baseline_detection(
        sst=sst,
        times=times,
        baseline=main_baseline,
        pctile=args.pctile,
        half_window=args.window_half_width,
        smooth_width=args.smooth_percentile_width,
        min_duration=args.min_duration,
        max_gap=args.max_gap,
        use_progress=use_progress,
    )

    print("[4/8] Writing main baseline core outputs...")
    save_clim_threshold_nc(
        out_path=os.path.join(
            out_main, f"clim_threshold_{main_baseline.label}_doy.nc"
        ),
        clim_doy=main_res["clim_doy"],
        thresh_doy=main_res["thresh_doy"],
        delta_doy=main_res["delta_doy"],
        lat=lat,
        lon=lon,
    )
    save_daily_state_nc(
        out_path=os.path.join(
            out_main, f"mhw_daily_state_{args.start_date}_{args.end_date}.nc"
        ),
        times=times,
        lat=lat,
        lon=lon,
        sst=sst,
        clim_daily=main_res["clim_daily"],
        thresh_daily=main_res["thresh_daily"],
        anomaly=main_res["anomaly"],
        exceed=main_res["exceed"],
        mask=main_res["mask"],
        starts=main_res["starts"],
    )

    print("[5/8] Building and writing event catalog...")
    events = build_event_catalog(
        mask=main_res["mask"],
        starts=main_res["starts"],
        anomaly=main_res["anomaly"],
        delta_daily=main_res["delta_daily"],
        times=times,
        lat=lat,
        lon=lon,
        use_progress=use_progress,
    )
    events_csv = os.path.join(
        out_main, f"mhw_events_catalog_{args.start_date}_{args.end_date}.csv"
    )
    events.to_csv(events_csv, index=False)
    try:
        events.to_parquet(
            os.path.join(
                out_main, f"mhw_events_catalog_{args.start_date}_{args.end_date}.parquet"
            ),
            index=False,
        )
    except Exception:
        pass

    print("[6/8] Computing annual metrics and trends...")
    years, metrics = compute_annual_metrics(
        mask=main_res["mask"],
        starts=main_res["starts"],
        anomaly=main_res["anomaly"],
        times=times,
    )
    save_annual_metrics_nc(
        out_path=os.path.join(out_main, "mhw_annual_metrics.nc"),
        years=years,
        lat=lat,
        lon=lon,
        metrics=metrics,
    )
    save_trend_nc(
        out_path=os.path.join(out_main, "mhw_trend_maps.nc"),
        lat=lat,
        lon=lon,
        metrics=metrics,
        years=years,
        use_progress=use_progress,
    )

    region_summary = []
    for i, yy in enumerate(years):
        row = {"year": int(yy)}
        for k, arr in metrics.items():
            row[f"{k}_region_mean"] = float(np.nanmean(arr[i, :, :]))
        region_summary.append(row)
    pd.DataFrame(region_summary).to_csv(
        os.path.join(out_main, "mhw_annual_region_summary.csv"), index=False
    )

    print("[7/8] Running baseline sensitivity summary...")
    sens_rows = []
    sens_rows.append(
        summarize_baseline(
            baseline=main_baseline,
            mask=main_res["mask"],
            starts=main_res["starts"],
            anomaly=main_res["anomaly"],
        )
    )
    for b in all_baselines:
        if b == main_baseline:
            continue
        print(f"      - sensitivity baseline {b.label}")
        res = run_baseline_detection(
            sst=sst,
            times=times,
            baseline=b,
            pctile=args.pctile,
            half_window=args.window_half_width,
            smooth_width=args.smooth_percentile_width,
            min_duration=args.min_duration,
            max_gap=args.max_gap,
            use_progress=use_progress,
        )
        sens_rows.append(
            summarize_baseline(
                baseline=b,
                mask=res["mask"],
                starts=res["starts"],
                anomaly=res["anomaly"],
            )
        )
    pd.DataFrame(sens_rows).to_csv(
        os.path.join(out_sens, "baseline_sensitivity_summary.csv"), index=False
    )

    print("[8/8] Writing run metadata...")
    write_key_value_csv(
        os.path.join(args.output_dir, "run_config.csv"),
        [
            ("input_dir", args.input_dir),
            ("file_glob", args.file_glob),
            ("start_date", args.start_date),
            ("end_date", args.end_date),
            ("var_name", args.var_name),
            ("lat_name", args.lat_name),
            ("lon_name", args.lon_name),
            ("bbox", args.bbox),
            ("coarsen_lat", args.coarsen_lat),
            ("coarsen_lon", args.coarsen_lon),
            ("pctile", args.pctile),
            ("window_half_width", args.window_half_width),
            ("smooth_percentile_width", args.smooth_percentile_width),
            ("min_duration", args.min_duration),
            ("max_gap", args.max_gap),
            ("baseline_main", main_baseline.label),
            (
                "baseline_sensitivity",
                ";".join(b.label for b in all_baselines if b != main_baseline),
            ),
        ],
    )

    print("Done.")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
