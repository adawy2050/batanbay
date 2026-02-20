# -*- coding: utf-8 -*-
# Copyright Ahmed Eladawy

"""
CHIRPS seasonal redistribution and wetting-mechanism analysis (point-based, 2003-2023).

This script builds a multi-panel figure that highlights how rainfall seasonality shifted over time.
Main components:
- Baseline vs late monthly climatology with IQR across years.
- Late-minus-baseline monthly deltas with bootstrap 95% confidence intervals.
- Frequency-intensity decomposition using WetDays and SDII.
- Timing index based on rainfall-centroid day of year.
- Time series for the most amplified wetting and drying months.
- Tile basemap support (ESRI/OSM) with a safe fallback if tile rendering fails.

Definitions (ETCCDI-aligned):
- Wet day: P >= WET_THRESH (mm/day)
- Monthly PRCPTOT: monthly sum over wet days
- Monthly WetDays: number of wet days per month
- Monthly SDII: PRCPTOT / WetDays
"""

import os
import re
import glob
import warnings
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import xarray as xr
import rasterio

from scipy.stats import theilslopes, kendalltau
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

from dask.diagnostics import ProgressBar

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleWTS, OSM

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "chirps_geotiff")  # <-- set your CHIRPS GeoTIFF folder
OUT_DIR  = os.path.join(BASE_DIR, "outputs", "chirps_seasonal_shift")  # <-- set output folder

START_DATE = "2003-01-01"
END_DATE   = "2023-12-31"

BASELINE_START = "2003-01-01"
BASELINE_END   = "2012-12-31"
LATE_START     = "2014-01-01"
LATE_END       = "2023-12-31"

WET_THRESH = 1.0  # mm/day

WRITE_ZARR_CACHE = True
ZARR_PATH = os.path.join(OUT_DIR, f"CHIRPS_BatanBay_daily_{START_DATE[:4]}_{END_DATE[:4]}.zarr")

# Point selection
USE_MANUAL_POINTS = False
MANUAL_POINTS = [
    ("P1", 122.355551, 11.349985),
    ("P2", 122.2615719, 11.574564),
    ("P3", 122.3595383, 11.529648),
    ("P4", 122.2515719, 11.439816),
    ("P5", 122.3595383, 11.394901),
    ("P6", 122.3505551, 11.484732),
    ("P7", 122.2515719, 11.349985),
]

N_POINTS = 10
POINT_LABELS = [f"P{i}" for i in range(1, N_POINTS + 1)]
POINT_BBOX = (122.32, 122.6, 11.33, 11.55)  # (lon_min, lon_max, lat_min, lat_max)
MIN_SEP_DEG = 0.02
RANDOM_SEED = 42

BATAN_BAY_LON = 122.4323508
BATAN_BAY_LAT = 11.5911672

# Basemap
USE_TILE_BASEMAP_DEFAULT = True
TILE_STYLE = "ESRI"   # "ESRI" or "OSM"
TILE_ZOOM  = 13
TILE_ALPHA = 0.95

# Output
FIG_PATH = os.path.join(OUT_DIR, "Fig2_CHIRPS_SeasonalShift_Mechanism_Points_Natures.png")
FIG_DPI  = 450

BOOT_N = 2000
BOOT_SEED = 123


# =============================================================================
# GeoTIFF loading
# =============================================================================
def _parse_date_from_name(fname: str) -> Optional[pd.Timestamp]:
    base = os.path.basename(fname)
    m = re.search(r"(?P<y>\d{4})[-_]?((?P<m>\d{2})[-_]?)(?P<d>\d{2})", base)
    if m:
        try:
            return pd.Timestamp(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        except Exception:
            return None
    m2 = re.search(r"(?P<ymd>\d{8})", base)
    if m2:
        try:
            return pd.to_datetime(m2.group("ymd"), format="%Y%m%d")
        except Exception:
            return None
    return None


def _read_geotiff_singleband(fpath: str, time: pd.Timestamp, var_name="precip") -> xr.DataArray:
    with rasterio.open(fpath) as src:
        arr = src.read(1).astype("float32")
        trans = src.transform
        ny, nx = arr.shape
        xs = trans.c + (np.arange(nx) + 0.5) * trans.a
        ys = trans.f + (np.arange(ny) + 0.5) * trans.e

    da = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"x": ("x", xs), "y": ("y", ys)},
        name=var_name,
    ).expand_dims(time=[np.datetime64(time)])
    return da


def _read_geotiff_multiband_stack(fpath: str, start_year: int, var_name="precip") -> xr.DataArray:
    with rasterio.open(fpath) as src:
        data = src.read().astype("float32")  # (band, y, x)
        trans = src.transform
        nb, ny, nx = data.shape
        xs = trans.c + (np.arange(nx) + 0.5) * trans.a
        ys = trans.f + (np.arange(ny) + 0.5) * trans.e

    times = pd.date_range(f"{start_year}-01-01", periods=nb, freq="D")
    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": times, "x": ("x", xs), "y": ("y", ys)},
        name=var_name,
    )
    return da


def load_chirps_from_dir(data_dir: str, var_name="precip") -> xr.DataArray:
    tifs = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    if not tifs:
        raise FileNotFoundError(f"No .tif files found in: {data_dir}")

    parsed = [(_parse_date_from_name(f), f) for f in tifs]
    n_dates = sum(p is not None for p, _ in parsed)

    # Case 1: daily files with dates in names
    if n_dates >= max(5, int(0.6 * len(tifs))):
        items = []
        for t, f in parsed:
            if t is None:
                continue
            items.append(_read_geotiff_singleband(f, t, var_name=var_name))
        da = xr.concat(items, dim="time").sortby("time")
        return da

    # Case 2: multi-band stacks per year
    stacks = []
    for f in tifs:
        base = os.path.basename(f)
        m = re.search(r"_(\d{4})_\d{4}\.tif$", base)
        if m:
            y0 = int(m.group(1))
            stacks.append(_read_geotiff_multiband_stack(f, y0, var_name=var_name))

    if not stacks:
        raise RuntimeError("Could not detect dated daily files or multi-band stacks.")

    da = xr.concat(stacks, dim="time").sortby("time")
    return da


# =============================================================================
# Point selection helpers
# =============================================================================
def _bbox_mask(da2: xr.DataArray, bbox: Tuple[float, float, float, float]) -> xr.DataArray:
    lon0, lon1, lat0, lat1 = bbox
    return (da2["x"] >= lon0) & (da2["x"] <= lon1) & (da2["y"] >= lat0) & (da2["y"] <= lat1)


def select_points_maximin(mask_xy: xr.DataArray, n: int, min_sep_deg: float, seed: int,
                          prefer_far_from: Tuple[float, float] = (BATAN_BAY_LON, BATAN_BAY_LAT)) -> List[Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    ys = mask_xy["y"].values
    xs = mask_xy["x"].values
    M = mask_xy.values.astype(bool)

    cand = np.argwhere(M)
    if cand.size == 0:
        return []

    cand = cand.copy()
    rng.shuffle(cand)
    lonlat = np.array([(xs[j], ys[i]) for i, j in cand], dtype="float64")

    def d2(a, b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2

    target = np.array(prefer_far_from, dtype="float64")
    dist_target = np.array([d2(p, target) for p in lonlat])
    idx0 = int(np.nanargmax(dist_target))
    selected = [lonlat[idx0]]

    while len(selected) < n:
        best_idx = None
        best_score = -np.inf
        for k, p in enumerate(lonlat):
            if any(np.sqrt(d2(p, s)) < min_sep_deg for s in selected):
                continue
            mind = min(d2(p, s) for s in selected)
            if mind > best_score:
                best_score = mind
                best_idx = k
        if best_idx is None:
            break
        selected.append(lonlat[best_idx])

    uniq = []
    for p in selected:
        if not any((abs(p[0]-q[0]) < 1e-9 and abs(p[1]-q[1]) < 1e-9) for q in uniq):
            uniq.append(p)
    return [(float(p[0]), float(p[1])) for p in uniq[:n]]


# =============================================================================
# Statistical helpers
# =============================================================================
def theilsen_mk_slope_per_decade(y: np.ndarray, years: np.ndarray) -> Tuple[float, float]:
    msk = np.isfinite(y) & np.isfinite(years)
    if msk.sum() < 8:
        return np.nan, np.nan
    yy = y[msk]
    tt = years[msk].astype("float64")
    try:
        slope, _, _, _ = theilslopes(yy, tt)  # per year
    except Exception:
        slope = np.nan
    try:
        _, p = kendalltau(tt, yy)
    except Exception:
        p = np.nan
    return float(slope * 10.0), float(p)


def rolling_mean_centered(y: np.ndarray, win: int = 5) -> np.ndarray:
    s = pd.Series(y)
    return s.rolling(win, center=True, min_periods=max(2, win//2)).mean().to_numpy()


# =============================================================================
# Basemap tiles
# =============================================================================
class EsriImagery(GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile[:3]
        return (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            f"World_Imagery/MapServer/tile/{z}/{y}/{x}"
        )


def add_basemap(ax, extent, use_tiles: bool, style="ESRI", zoom=12, alpha=0.95):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    if not use_tiles:
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.93", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.88", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, zorder=3)
        return

    try:
        tiler = EsriImagery() if style.upper() == "ESRI" else OSM()
        ax.add_image(tiler, zoom, alpha=alpha, zorder=0)
    except Exception:
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.93", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.88", zorder=0)

    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, zorder=3)


def add_gridlabels(ax):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.3, color="k", alpha=0.25, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}


def tidy_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# Monthly metrics at selected points
# =============================================================================
def compute_monthly_point_metrics(daily_pts: xr.DataArray, wet_thresh: float):
    """
    daily_pts: (time, point)
    Returns:
      mon_tot:    (time_ms, point) monthly total over wet days (P>=wet_thresh)
      mon_wd:     (time_ms, point) number of wet days in month
      mon_sdii:   (time_ms, point) mon_tot/mon_wd
    """
    wet = daily_pts >= wet_thresh
    mon_tot = daily_pts.where(wet).resample(time="1MS").sum("time").astype("float32")
    mon_wd = wet.resample(time="1MS").sum("time").astype("float32")
    mon_sdii = (mon_tot / mon_wd).where(mon_wd >= 1).astype("float32")
    return mon_tot, mon_wd, mon_sdii


def to_year_month_df(mon: xr.DataArray) -> pd.DataFrame:
    """
    mon: (time_ms, point)
    Returns df with columns: year, month, value (points-median)
    """
    mon_med = mon.median("point")
    dt = pd.to_datetime(mon_med["time"].values)
    df = pd.DataFrame({"time": dt, "val": mon_med.values})
    df["year"] = df["time"].dt.year.astype(int)
    df["month"] = df["time"].dt.month.astype(int)
    return df[["year", "month", "val"]].copy()


def period_climatology(df: pd.DataFrame, y0: int, y1: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each month, compute mean and IQR across YEARS (using points-median values).
    Returns mean[12], q25[12], q75[12]
    """
    out_mean, out_q25, out_q75 = [], [], []
    for m in range(1, 13):
        s = df[(df["year"] >= y0) & (df["year"] <= y1) & (df["month"] == m)]["val"].to_numpy(dtype="float64")
        out_mean.append(np.nanmean(s))
        out_q25.append(np.nanquantile(s, 0.25))
        out_q75.append(np.nanquantile(s, 0.75))
    return np.array(out_mean), np.array(out_q25), np.array(out_q75)


def bootstrap_delta_by_month(df: pd.DataFrame,
                             base_years: np.ndarray,
                             late_years: np.ndarray,
                             nboot: int,
                             seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap difference (late - baseline) in monthly means.
    Returns delta_mean[12], ci_low[12], ci_high[12]
    """
    rng = np.random.default_rng(seed)

    def month_mean_for_years(years_sample, m):
        s = df[(df["year"].isin(years_sample)) & (df["month"] == m)]["val"].to_numpy(dtype="float64")
        return np.nanmean(s)

    boot = np.full((nboot, 12), np.nan, dtype="float64")
    for b in range(nboot):
        ys_b = rng.choice(base_years, size=len(base_years), replace=True)
        ys_l = rng.choice(late_years, size=len(late_years), replace=True)
        for mi, m in enumerate(range(1, 13)):
            mb = month_mean_for_years(ys_b, m)
            ml = month_mean_for_years(ys_l, m)
            boot[b, mi] = ml - mb

    delta = np.nanmean(boot, axis=0)
    lo = np.nanquantile(boot, 0.025, axis=0)
    hi = np.nanquantile(boot, 0.975, axis=0)
    return delta, lo, hi


def rainfall_centroid_doy(df_tot: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Yearly rainfall centroid timing index based on monthly totals (points-median).
    Uses mid-month DOY; returns arrays (years, centroid_doy).
    """
    years = np.sort(df_tot["year"].unique())
    # Mid-month DOY for a non-leap reference year
    ref = 2001
    mid_doy = np.array([pd.Timestamp(ref, m, 15).dayofyear for m in range(1, 13)], dtype="float64")
    theta = 2.0 * np.pi * mid_doy / 365.0

    cent = []
    for y in years:
        w = np.array([df_tot[(df_tot["year"] == y) & (df_tot["month"] == m)]["val"].mean()
                      for m in range(1, 13)], dtype="float64")
        w = np.where(np.isfinite(w), w, 0.0)
        if w.sum() <= 0:
            cent.append(np.nan)
            continue
        vx = np.sum(w * np.cos(theta))
        vy = np.sum(w * np.sin(theta))
        ang = np.arctan2(vy, vx)
        if ang < 0:
            ang += 2*np.pi
        doy = 365.0 * ang / (2*np.pi)
        cent.append(doy)

    cent = np.array(cent, dtype="float64")
    # unwrap to avoid artificial jumps (usually not needed here but safe)
    ang_series = 2*np.pi*cent/365.0
    ang_unwrap = np.unwrap(ang_series)
    cent_unwrap = 365.0 * ang_unwrap / (2*np.pi)
    return years.astype(int), cent_unwrap


def month_series(df: pd.DataFrame, month: int) -> Tuple[np.ndarray, np.ndarray]:
    s = df[df["month"] == month].sort_values("year")
    return s["year"].to_numpy(dtype=int), s["val"].to_numpy(dtype="float64")


# =============================================================================
# Figure assembly
# =============================================================================
def build_figure(extent, labels, pts,
                 df_tot, df_wd, df_sdii,
                 base_year0, base_year1,
                 late_year0, late_year1,
                 nboot, seed,
                 use_tiles: bool):

    mpl.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
    })

    months = np.arange(1, 13)
    monlab = [pd.Timestamp(2000, m, 1).strftime("%b") for m in months]

    # Monthly climatologies
    base_mean_T, base_q25_T, base_q75_T = period_climatology(df_tot, base_year0, base_year1)
    late_mean_T, late_q25_T, late_q75_T = period_climatology(df_tot, late_year0, late_year1)

    base_mean_W, base_q25_W, base_q75_W = period_climatology(df_wd, base_year0, base_year1)
    late_mean_W, late_q25_W, late_q75_W = period_climatology(df_wd, late_year0, late_year1)

    base_mean_S, base_q25_S, base_q75_S = period_climatology(df_sdii, base_year0, base_year1)
    late_mean_S, late_q25_S, late_q75_S = period_climatology(df_sdii, late_year0, late_year1)

    # Monthly deltas and confidence intervals
    base_years = np.arange(base_year0, base_year1 + 1)
    late_years = np.arange(late_year0, late_year1 + 1)

    dT, loT, hiT = bootstrap_delta_by_month(df_tot, base_years, late_years, nboot, seed)
    dW, loW, hiW = bootstrap_delta_by_month(df_wd,  base_years, late_years, nboot, seed+1)
    dS, loS, hiS = bootstrap_delta_by_month(df_sdii, base_years, late_years, nboot, seed+2)

    # Months with strongest wetting and drying shifts
    wet_month = int(np.nanargmax(dT) + 1)
    dry_month = int(np.nanargmin(dT) + 1)

    # Rainfall timing index
    yrsC, centDOY = rainfall_centroid_doy(df_tot)
    slopeC, pC = theilsen_mk_slope_per_decade(centDOY, yrsC.astype("float64"))

    # Time series for amplified months (monthly totals, median across points)
    yrs_wet, ts_wet = month_series(df_tot, wet_month)
    yrs_dry, ts_dry = month_series(df_tot, dry_month)
    sm_wet = rolling_mean_centered(ts_wet, 5)
    sm_dry = rolling_mean_centered(ts_dry, 5)
    slope_wet, p_wet = theilsen_mk_slope_per_decade(ts_wet, yrs_wet.astype("float64"))
    slope_dry, p_dry = theilsen_mk_slope_per_decade(ts_dry, yrs_dry.astype("float64"))

    # Layout: 3 rows x 4 columns (10 panels)
    fig = plt.figure(figsize=(21.0, 12.5), dpi=FIG_DPI)
    gs = GridSpec(nrows=3, ncols=4, figure=fig, hspace=0.45, wspace=0.32)

    # a: map
    axA = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    add_basemap(axA, extent, use_tiles=use_tiles, style=TILE_STYLE, zoom=TILE_ZOOM, alpha=TILE_ALPHA)
    add_gridlabels(axA)
    axA.set_title("a  Upstream context & selected points", loc="left", fontweight="bold")
    axA.plot(BATAN_BAY_LON, BATAN_BAY_LAT, marker="*", ms=10, color="r", transform=ccrs.PlateCarree(), zorder=4)
    axA.text(BATAN_BAY_LON + 0.01, BATAN_BAY_LAT, "Batan Bay",
             transform=ccrs.PlateCarree(), fontsize=10,color="r", fontweight="bold", zorder=4)
    for lab, (lon, lat) in zip(labels, pts):
        axA.plot(lon, lat, marker="s", ms=6, mfc="white", mec="k", transform=ccrs.PlateCarree(), zorder=4)
        axA.text(lon + 0.01, lat + 0.005, lab, transform=ccrs.PlateCarree(), fontsize=9, zorder=4,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # Helper: climatology lines with IQR ribbons
    def plot_clim(ax, base_mean, base_q25, base_q75, late_mean, late_q25, late_q75, title, ylab):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.plot(months, base_mean, lw=2.0, label=f"Baseline ({base_year0}-{base_year1})")
        ax.fill_between(months, base_q25, base_q75, alpha=0.20)
        ax.plot(months, late_mean, lw=2.0, label=f"Late ({late_year0}-{late_year1})")
        ax.fill_between(months, late_q25, late_q75, alpha=0.20)
        ax.set_xticks(months)
        ax.set_xticklabels(monlab)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.25)
        tidy_axes(ax)

    # Helper: delta bars with confidence intervals
    def plot_delta(ax, delta, lo, hi, title, ylab):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.bar(months, delta)
        ax.errorbar(months, delta, yerr=[delta - lo, hi - delta], fmt="none", lw=1.0, capsize=2)
        ax.axhline(0, lw=0.8, color="k", alpha=0.6)
        ax.set_xticks(months)
        ax.set_xticklabels(monlab)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.25)
        tidy_axes(ax)
        # Add a star when the CI does not cross zero
        ymax = np.nanmax(np.abs(delta)) if np.isfinite(delta).any() else 1.0
        for i, m in enumerate(months):
            if np.isfinite(lo[i]) and np.isfinite(hi[i]) and (lo[i] > 0 or hi[i] < 0):
                ax.text(m, delta[i] + 0.06*ymax*np.sign(delta[i] if delta[i] != 0 else 1),
                        "*", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # b: climatology totals
    axB = fig.add_subplot(gs[0, 1:3])
    plot_clim(axB, base_mean_T, base_q25_T, base_q75_T, late_mean_T, late_q25_T, late_q75_T,
              "b  Seasonal cycle of monthly PRCPTOT (points-median; IQR across years)",
              "mm month$^{-1}$")

    axB.legend(frameon=False, ncol=2, loc="upper left")

    # c: delta totals
    axC = fig.add_subplot(gs[0, 3])
    plot_delta(axC, dT, loT, hiT, "c  Δ PRCPTOT (late − baseline)", "mm month$^{-1}$")

    # d: climatology wet days
    axD = fig.add_subplot(gs[1, 0:2])
    plot_clim(axD, base_mean_W, base_q25_W, base_q75_W, late_mean_W, late_q25_W, late_q75_W,
              "d  Seasonal cycle of wet-day frequency (WetDays)", "days month$^{-1}$")

    # e: delta wet days
    axE = fig.add_subplot(gs[1, 2])
    plot_delta(axE, dW, loW, hiW, "e  Δ WetDays (late − baseline)", "days month$^{-1}$")

    # f: climatology SDII
    axF = fig.add_subplot(gs[1, 3])
    plot_clim(axF, base_mean_S, base_q25_S, base_q75_S, late_mean_S, late_q25_S, late_q75_S,
              "f  Seasonal cycle of intensity (SDII = PRCPTOT/WetDays)", "mm wetday$^{-1}$")

    # g: delta SDII
    axG = fig.add_subplot(gs[2, 0])
    plot_delta(axG, dS, loS, hiS, "g  Δ SDII (late − baseline)", "mm wetday$^{-1}$")

    # h: centroid timing
    axH = fig.add_subplot(gs[2, 1])
    axH.set_title("h  Rainfall timing index", loc="left", fontweight="bold")
    axH.plot(yrsC, centDOY, marker="o", ms=3, lw=1.0)
    axH.plot(yrsC, rolling_mean_centered(centDOY, 5), lw=2.0)
    axH.set_xlabel("Year")
    axH.set_ylabel("Centroid DOY")
    axH.grid(True, alpha=0.25)
    tidy_axes(axH)
    axH.text(0.02, 0.05, f"Theil–Sen: {slopeC:.2f} DOY/dec, MK p={pC:.3g}",
             transform=axH.transAxes, fontsize=9,
             bbox=dict(fc="white", ec="none", alpha=0.75))

    # i: amplified wetting month time series
    axI = fig.add_subplot(gs[2, 2])
    wet_name = pd.Timestamp(2000, wet_month, 1).strftime("%b")
    axI.set_title(f"i  Amplified wetting month: {wet_name} PRCPTOT", loc="left", fontweight="bold")
    axI.plot(yrs_wet, ts_wet, marker="o", ms=3, lw=1.0)
    axI.plot(yrs_wet, sm_wet, lw=2.0)
    axI.axvspan(base_year0, base_year1, alpha=0.08)
    axI.axvspan(int(LATE_START[:4]), int(LATE_END[:4]), alpha=0.08)
    axI.set_xlabel("Year")
    axI.set_ylabel("mm month$^{-1}$")
    axI.grid(True, alpha=0.25)
    tidy_axes(axI)
    axI.text(0.02, 0.05, f"Theil–Sen: {slope_wet:.2f} mm/dec, MK p={p_wet:.3g}",
             transform=axI.transAxes, fontsize=9,
             bbox=dict(fc="white", ec="none", alpha=0.75))

    # j: amplified drying month time series
    axJ = fig.add_subplot(gs[2, 3])
    dry_name = pd.Timestamp(2000, dry_month, 1).strftime("%b")
    axJ.set_title(f"j  Amplified drying month: {dry_name} PRCPTOT", loc="left", fontweight="bold")
    axJ.plot(yrs_dry, ts_dry, marker="o", ms=3, lw=1.0)
    axJ.plot(yrs_dry, sm_dry, lw=2.0)
    axJ.axvspan(base_year0, base_year1, alpha=0.08)
    axJ.axvspan(int(LATE_START[:4]), int(LATE_END[:4]), alpha=0.08)
    axJ.set_xlabel("Year")
    axJ.set_ylabel("mm month$^{-1}$")
    axJ.grid(True, alpha=0.25)
    tidy_axes(axJ)
    axJ.text(0.02, 0.05, f"Theil–Sen: {slope_dry:.2f} mm/dec, MK p={p_dry:.3g}",
             transform=axJ.transAxes, fontsize=9,
             bbox=dict(fc="white", ec="none", alpha=0.75))

    point_span = f"{labels[0]}-{labels[-1]}" if len(labels) >= 2 else labels[0]
    fig.suptitle(
        f"CHIRPS seasonal redistribution over Batan Bay upstream ({point_span}): "
        "explicit seasonal shift + frequency-intensity mechanism (2003-2023)",
        y=0.995, fontweight="bold"
    )

    foot = (
        f"Baseline: {BASELINE_START}–{BASELINE_END}; Late: {LATE_START}–{LATE_END}. "
        f"Wet day threshold: P ≥ {WET_THRESH:.1f} mm d$^{{-1}}$. "
        f"PRCPTOT computed as monthly sum over wet days; WetDays is monthly wet-day count; SDII = PRCPTOT/WetDays. "
        f"Panels c,e,g show late−baseline differences with bootstrap 95% CI (n={nboot}); '*' indicates CI excludes 0. "
        f"Panel h timing uses rainfall centroid (monthly totals weighted by mid-month DOY). "
        f"Amplified months (i–j) selected from Δ PRCPTOT maxima/minima."
    )
    fig.text(0.01, 0.01, foot, ha="left", va="bottom", fontsize=8)

    return fig


# =============================================================================
# Main execution
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading CHIRPS GeoTIFFs...")
    if os.path.exists(ZARR_PATH):
        print("Loading from Zarr cache...")
        da = xr.open_zarr(ZARR_PATH)["precip"]
    else:
        da = load_chirps_from_dir(DATA_DIR, var_name="precip")
        da = da.sortby("y").sortby("x")
        if WRITE_ZARR_CACHE:
            print("Writing Zarr cache...")
            with ProgressBar():
                da.chunk({"time": 365, "y": 256, "x": 256}).to_dataset(name="precip").to_zarr(ZARR_PATH)
            da = xr.open_zarr(ZARR_PATH)["precip"]

    da = da.sel(time=slice(np.datetime64(START_DATE), np.datetime64(END_DATE))).sortby("y").sortby("x")
    da = da.chunk({"time": 365, "y": 256, "x": 256})
    extent = (float(da["x"].min()), float(da["x"].max()), float(da["y"].min()), float(da["y"].max()))

    # Build mask for point selection
    base = da.sel(time=slice(np.datetime64(BASELINE_START), np.datetime64(BASELINE_END)))
    with ProgressBar():
        p95 = base.quantile(0.95, dim="time").compute()
    mask = xr.where(np.isfinite(p95), True, False)

    # Choose points
    if USE_MANUAL_POINTS:
        labels = [lab for lab, _, _ in MANUAL_POINTS]
        pts = [(lon, lat) for _, lon, lat in MANUAL_POINTS]
    else:
        bbox_m = _bbox_mask(mask, POINT_BBOX)
        cand_mask = mask & bbox_m
        n_cand = int(cand_mask.values.sum())
        print(f"Candidate pixels in POINT_BBOX: {n_cand}")
        if n_cand < N_POINTS:
            raise RuntimeError("Not enough candidate pixels in bbox. Expand POINT_BBOX or reduce N_POINTS.")
        pts = select_points_maximin(cand_mask, n=N_POINTS, min_sep_deg=MIN_SEP_DEG, seed=RANDOM_SEED)
        if len(pts) != N_POINTS:
            raise RuntimeError("Point selection failed. Expand POINT_BBOX or reduce MIN_SEP_DEG.")
        labels = POINT_LABELS

    print("Selected points (lon,lat):")
    for lab, (lon, lat) in zip(labels, pts):
        print(f"  {lab}: ({lon:.6f}, {lat:.6f})")

    # Extract point time series
    daily_pts = []
    for lab, (lon, lat) in zip(labels, pts):
        s = da.sel(x=lon, y=lat, method="nearest").rename(lab)
        daily_pts.append(s)
    daily_pts = xr.concat(daily_pts, dim="point")
    daily_pts = daily_pts.assign_coords(point=("point", labels)).transpose("time", "point")

    # Compute monthly metrics
    print("Computing monthly point metrics...")
    mon_tot, mon_wd, mon_sdii = compute_monthly_point_metrics(daily_pts, WET_THRESH)

    df_tot = to_year_month_df(mon_tot)
    df_wd  = to_year_month_df(mon_wd)
    df_sdii = to_year_month_df(mon_sdii)

    base_year0 = int(BASELINE_START[:4])
    base_year1 = int(BASELINE_END[:4])
    late_year0 = int(LATE_START[:4])
    late_year1 = int(LATE_END[:4])

    use_tiles = USE_TILE_BASEMAP_DEFAULT

    fig = build_figure(
        extent, labels, pts,
        df_tot, df_wd, df_sdii,
        base_year0, base_year1,
        late_year0, late_year1,
        BOOT_N, BOOT_SEED,
        use_tiles=use_tiles
    )

    try:
        fig.savefig(FIG_PATH, dpi=FIG_DPI)
        plt.close(fig)
        print("DONE:", FIG_PATH)
    except Exception as e:
        print("\nWARNING: basemap tile rendering failed during savefig().")
        print("Reason:", repr(e))
        print("Re-rendering without tiles...\n")
        plt.close(fig)
        fig = build_figure(
            extent, labels, pts,
            df_tot, df_wd, df_sdii,
            base_year0, base_year1,
            late_year0, late_year1,
            BOOT_N, BOOT_SEED,
            use_tiles=False
        )
        fig.savefig(FIG_PATH, dpi=FIG_DPI)
        plt.close(fig)
        print("DONE (fallback):", FIG_PATH)


if __name__ == "__main__":
    main()
