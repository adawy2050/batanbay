# -*- coding: utf-8 -*-
# Copyright Ahmed Eladawy

"""
MUR GHRSST inlet analysis and Hobday MHW diagnostics (2003-2023).

What this script does:
- Finds the nearest valid ocean pixel to the inlet (avoids land-mask NaNs).
- Extracts the daily SST series and computes monthly anomalies.
- Detects marine heatwave events using Hobday-style thresholds.
- Exports CSV outputs, a 2x2 summary figure, and a LaTeX table of top events since 2015.
"""

import os
import glob
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.stats import linregress, theilslopes

warnings.filterwarnings("ignore", message=".*'Y' is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*resample.*", category=FutureWarning)

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "ghrsst")
PATTERN  = os.path.join(DATA_DIR, "*JPL-L4_GHRSST-SSTfnd-MUR-GLOB*.nc4")

INLET_LATLON = (11.596509, 122.492519)  # (lat, lon)
T0, T1 = "2003-01-01", "2023-12-31"

# Point-selection robustness
SEARCH_RADIUS_KM = 10.0   # local search radius around inlet
MAX_CANDIDATES   = 200    # cap candidate points checked (for speed)

# MHW parameters (Hobday-style)
MHW_PCT = 90
MIN_DURATION_D = 5
DOY_HALF_WINDOW = 5  # ±5 days around DOY (11-day window)

# Output paths
OUT_DIR = os.path.join(BASE_DIR, "outputs", "ghrsst_inlet")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_PNG = os.path.join(OUT_DIR, "MUR_inlet_point_2003_2023_HobdayMHW_Nature_v2.png")
FIG_PDF = os.path.join(OUT_DIR, "MUR_inlet_point_2003_2023_HobdayMHW_Nature_v2.pdf")

CSV_DAILY   = os.path.join(OUT_DIR, "inlet_daily_sst_2003_2023.csv")
CSV_MONTHLY = os.path.join(OUT_DIR, "inlet_monthly_sst_anom_2003_2023.csv")
CSV_EVENTS  = os.path.join(OUT_DIR, "hobday_mhw_events_2003_2023.csv")
TEX_EVENTS_2015 = os.path.join(OUT_DIR, "hobday_mhw_events_2015_present_top.tex")

# =============================================================================
# Helper functions
# =============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km)."""
    R = 6371.0088
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def _find_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords:
            return c
        if c in ds.variables and ds[c].dims == (c,):
            return c
    return None

def _get_lon_adjusted(ds_lon, target_lon):
    """Adjust target_lon to match ds lon convention (0..360 vs -180..180)."""
    lon_min = float(np.nanmin(ds_lon))
    lon_max = float(np.nanmax(ds_lon))
    if lon_min >= 0 and lon_max > 180:  # 0..360
        return target_lon % 360.0
    # Otherwise assume -180..180 (or a limited regional domain)
    # Also handle the rare case where data are 0..180 and target lon is negative
    if target_lon > 180:
        return ((target_lon + 180) % 360) - 180
    return target_lon

def _decode_sst_to_c(da):
    """
    Convert analysed_sst-like variable to degC robustly.
    - If values look Kelvin -> subtract 273.15
    - Enforce float + mask unrealistic values
    """
    da = da.astype("float64")
    # Many GHRSST products are stored in Kelvin
    vmed = float(da.quantile(0.5, skipna=True))
    if vmed > 100:  # Kelvin-like
        da = da - 273.15
    # Mask physically unrealistic values
    da = da.where(np.isfinite(da))
    da = da.where((da > -5) & (da < 45))
    return da

def list_files_time_filtered(files, t0, t1):
    """Keep files that overlap [t0,t1] by quickly reading their time coord."""
    good = []
    t0 = np.datetime64(t0); t1 = np.datetime64(t1)
    for fn in files:
        try:
            with xr.open_dataset(fn, engine="netcdf4", decode_times=True, decode_timedelta=False) as ds0:
                if "time" not in ds0:
                    continue
                tt = ds0["time"].values
                if tt.size == 0:
                    continue
                if (tt.min() <= t1) and (tt.max() >= t0):
                    good.append(fn)
        except Exception:
            continue
    return good

def pick_nearest_valid_ocean_pixel(sample_file, inlet_lat, inlet_lon,
                                   search_radius_km=10.0, max_candidates=200):
    """
    Choose nearest (lat,lon) grid cell to inlet that has a valid SST value
    (i.e., ocean, not masked). Uses one representative file.
    """
    with xr.open_dataset(sample_file, engine="netcdf4", decode_times=True, decode_timedelta=False) as ds:
        lat_name = _find_coord_name(ds, ["lat", "latitude", "nav_lat"])
        lon_name = _find_coord_name(ds, ["lon", "longitude", "nav_lon"])
        if lat_name is None or lon_name is None:
            raise ValueError("Could not find lat/lon coordinates in dataset.")

        lat = ds[lat_name].values
        lon = ds[lon_name].values

        inlet_lon_adj = _get_lon_adjusted(lon, inlet_lon)

        # Build candidate indices in a bounding box first, then refine by distance
        # Approximate degrees per km
        dlat = search_radius_km / 110.574
        dlon = search_radius_km / (111.320 * np.cos(np.deg2rad(inlet_lat)) + 1e-12)

        lat_mask = (lat >= inlet_lat - dlat) & (lat <= inlet_lat + dlat)
        lon_mask = (lon >= inlet_lon_adj - dlon) & (lon <= inlet_lon_adj + dlon)

        ilat = np.where(lat_mask)[0]
        ilon = np.where(lon_mask)[0]
        if ilat.size == 0 or ilon.size == 0:
            # Fallback to nearest index (may still land on masked cells)
            ilat = np.array([int(np.argmin(np.abs(lat - inlet_lat)))])
            ilon = np.array([int(np.argmin(np.abs(lon - inlet_lon_adj)))])

        # Candidate grid points
        LAT2, LON2 = np.meshgrid(lat[ilat], lon[ilon], indexing="ij")
        dist = haversine_km(inlet_lat, inlet_lon_adj, LAT2, LON2).reshape(-1)

        order = np.argsort(dist)
        order = order[:max_candidates]

        # SST variable name
        var = None
        for v in ["analysed_sst", "sst", "sea_surface_temperature"]:
            if v in ds.data_vars:
                var = v
                break
        if var is None:
            raise ValueError("Could not find SST variable (analysed_sst/sst/sea_surface_temperature).")

        da = ds[var].isel(time=0) if "time" in ds[var].dims else ds[var]
        da = _decode_sst_to_c(da)

        # Return the first valid ocean candidate
        flat_ij = np.array(np.unravel_index(order, LAT2.shape)).T  # rows/cols in LAT2
        for rr, cc in flat_ij:
            lat_idx = ilat[rr]
            lon_idx = ilon[cc]
            val = da.isel({lat_name: lat_idx, lon_name: lon_idx}).values
            if np.isfinite(val):
                chosen_lat = float(lat[lat_idx])
                chosen_lon = float(lon[lon_idx])
                chosen_dist = float(haversine_km(inlet_lat, inlet_lon_adj, chosen_lat, chosen_lon))
                return lat_name, lon_name, chosen_lat, chosen_lon, chosen_dist

        raise RuntimeError(
            "No valid ocean pixel found within search radius. "
            "Increase SEARCH_RADIUS_KM or inspect land mask near inlet."
        )

def extract_point_timeseries(files, lat_name, lon_name, target_lat, target_lon):
    """
    Extract daily SST time series at (target_lat, target_lon) using nearest index in each file.
    Reads one value per time step per file (fast enough without open_mfdataset).
    """
    rows = []
    for fn in files:
        try:
            with xr.open_dataset(fn, engine="netcdf4", decode_times=True, decode_timedelta=False) as ds:
                # Find SST variable
                var = None
                for v in ["analysed_sst", "sst", "sea_surface_temperature"]:
                    if v in ds.data_vars:
                        var = v
                        break
                if var is None or "time" not in ds:
                    continue

                lat = ds[lat_name].values
                lon = ds[lon_name].values
                # Ensure longitude convention matches the dataset
                target_lon_adj = _get_lon_adjusted(lon, target_lon)

                i_lat = int(np.argmin(np.abs(lat - target_lat)))
                i_lon = int(np.argmin(np.abs(lon - target_lon_adj)))

                da = ds[var].isel({lat_name: i_lat, lon_name: i_lon})
                da = _decode_sst_to_c(da)

                t = pd.to_datetime(ds["time"].values)
                y = da.values

                # `y` may be scalar or array depending on file structure
                if np.ndim(y) == 0:
                    rows.append((t[0], float(y)))
                else:
                    rows.extend(list(zip(t, y.astype("float64"))))
        except Exception:
            continue

    df = pd.DataFrame(rows, columns=["time", "sst_c"]).dropna()
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

def monthly_climatology_anomaly(monthly_sst):
    """Monthly anomalies relative to 2003–2023 monthly climatology."""
    clim = monthly_sst.groupby(monthly_sst.index.month).mean()
    anom = monthly_sst.copy()
    anom[:] = [monthly_sst.iloc[i] - clim[monthly_sst.index[i].month] for i in range(len(monthly_sst))]
    return clim, anom

def within_year_cumulative_monthly(anom_monthly):
    """Cumulative monthly anomalies that reset each calendar year."""
    df = anom_monthly.to_frame("anom")
    df["year"] = df.index.year
    df["cum_within_year"] = df.groupby("year")["anom"].cumsum()
    return df

# =============================================================================
# Hobday MHW (single-point) implementation
# =============================================================================
@dataclass
class MHWEvent:
    start: pd.Timestamp
    end: pd.Timestamp
    duration: int
    peak_intensity: float
    mean_intensity: float
    cum_intensity: float
    category: str

def compute_hobday_threshold_and_clim(daily_sst: pd.Series,
                                     pct=90,
                                     doy_half_window=5):
    """
    Hobday-style: for each DOY, compute climatology mean and threshold percentile
    using all days within ±window around DOY across all years.
    Returns aligned series: clim_mean, thresh, and a "seasonal exceedance scale" (thresh - clim).
    """
    s = daily_sst.dropna().copy()
    # Drop Feb 29 to keep a stable 365-day DOY mapping
    is_feb29 = (s.index.month == 2) & (s.index.day == 29)
    s = s.loc[~is_feb29]

    doy = s.index.dayofyear.values
    vals = s.values

    clim_mean = np.full_like(vals, np.nan, dtype="float64")
    thresh = np.full_like(vals, np.nan, dtype="float64")
    delta = np.full_like(vals, np.nan, dtype="float64")

    # Pre-index by DOY for faster window lookups
    idx_by_doy = {}
    for d in range(1, 366):
        idx_by_doy[d] = np.where(doy == d)[0]

    for i, d in enumerate(doy):
        lo = d - doy_half_window
        hi = d + doy_half_window
        # Wrap around the year at boundaries
        window_days = []
        for dd in range(lo, hi + 1):
            if dd < 1:
                window_days.append(dd + 365)
            elif dd > 365:
                window_days.append(dd - 365)
            else:
                window_days.append(dd)

        widx = np.concatenate([idx_by_doy[dd] for dd in window_days if idx_by_doy[dd].size > 0])
        wvals = vals[widx]
        if wvals.size < 20:  # safety; should not happen with 21 yrs
            continue
        cm = np.nanmean(wvals)
        th = np.nanpercentile(wvals, pct)
        clim_mean[i] = cm
        thresh[i] = th
        delta[i] = th - cm

    clim_s = pd.Series(clim_mean, index=s.index, name="clim")
    thr_s  = pd.Series(thresh,    index=s.index, name="thresh")
    del_s  = pd.Series(delta,     index=s.index, name="delta")
    return clim_s, thr_s, del_s

def detect_mhw_events(daily_sst: pd.Series,
                      clim: pd.Series,
                      thresh: pd.Series,
                      delta: pd.Series,
                      min_duration=5):
    """
    Identify MHW events where SST > threshold for >= min_duration consecutive days.
    Intensity = SST - clim (Hobday definition).
    Category based on multiples of delta (threshold - clim):
      I: 1–2x (Moderate), II: 2–3x (Strong), III: 3–4x (Severe), IV: >=4x (Extreme)
    """
    # Align all series on a shared index
    idx = daily_sst.index.intersection(clim.index).intersection(thresh.index).intersection(delta.index)
    s = daily_sst.loc[idx]
    c = clim.loc[idx]
    t = thresh.loc[idx]
    d = delta.loc[idx]

    exceed = (s > t) & np.isfinite(s) & np.isfinite(t) & np.isfinite(c) & np.isfinite(d) & (d > 0)

    events = []
    if exceed.sum() == 0:
        return events, exceed

    # Find exceedance runs
    flag = exceed.values.astype(int)
    # Run boundaries
    starts = np.where(np.diff(np.r_[0, flag]) == 1)[0]
    ends   = np.where(np.diff(np.r_[flag, 0]) == -1)[0] - 1

    for st, en in zip(starts, ends):
        dur = en - st + 1
        if dur < min_duration:
            continue
        seg_idx = s.index[st:en+1]
        seg_s = s.iloc[st:en+1]
        seg_c = c.iloc[st:en+1]
        seg_d = d.iloc[st:en+1]

        intensity = (seg_s - seg_c)
        peak = float(np.nanmax(intensity.values))
        mean_int = float(np.nanmean(intensity.values))
        cum_int = float(np.nansum(intensity.values))  # °C·d

        # Assign Hobday category from peak intensity relative to local delta
        peak_day = intensity.idxmax()
        peak_delta = float(seg_d.loc[peak_day])
        if not np.isfinite(peak_delta) or peak_delta <= 0:
            cat = "I (Moderate)"
        else:
            ratio = peak / peak_delta
            if ratio < 2:
                cat = "I (Moderate)"
            elif ratio < 3:
                cat = "II (Strong)"
            elif ratio < 4:
                cat = "III (Severe)"
            else:
                cat = "IV (Extreme)"

        events.append(MHWEvent(
            start=seg_idx[0], end=seg_idx[-1], duration=dur,
            peak_intensity=peak, mean_intensity=mean_int, cum_intensity=cum_int,
            category=cat
        ))

    return events, exceed

def events_to_df(events):
    if len(events) == 0:
        return pd.DataFrame(columns=["start","end","duration_d","peak_intensity_c","mean_intensity_c","cum_intensity_cdays","category"])
    return pd.DataFrame({
        "start": [e.start for e in events],
        "end": [e.end for e in events],
        "duration_d": [e.duration for e in events],
        "peak_intensity_c": [e.peak_intensity for e in events],
        "mean_intensity_c": [e.mean_intensity for e in events],
        "cum_intensity_cdays": [e.cum_intensity for e in events],
        "category": [e.category for e in events],
    }).sort_values("start").reset_index(drop=True)

def df_to_latex_table(df, caption, label):
    """Return LaTeX table text (no file I/O)."""
    d = df.copy()
    d["start"] = pd.to_datetime(d["start"]).dt.strftime("%Y-%m-%d")
    d["end"] = pd.to_datetime(d["end"]).dt.strftime("%Y-%m-%d")
    # Format numeric columns
    for col in ["peak_intensity_c","mean_intensity_c","cum_intensity_cdays"]:
        d[col] = d[col].map(lambda x: f"{x:.2f}")
    d["duration_d"] = d["duration_d"].astype(int).astype(str)

    cols = ["start","end","duration_d","peak_intensity_c","mean_intensity_c","cum_intensity_cdays","category"]
    header = ["Start","End","Dur (d)","Peak ($^\\circ$C)","Mean ($^\\circ$C)","Cum ($^\\circ$C\\,d)","Cat"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lllllll}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for _, r in d[cols].iterrows():
        lines.append(" & ".join(r.values) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# =============================================================================
# Main execution
# =============================================================================
def main():
    all_files = sorted(glob.glob(PATTERN))
    if not all_files:
        raise FileNotFoundError(f"No files matched: {PATTERN}")

    good_files = list_files_time_filtered(all_files, T0, T1)
    if not good_files:
        raise RuntimeError("No files overlap requested time window. Check file time coverage.")

    inlet_lat, inlet_lon = INLET_LATLON

    # 1) Pick nearest valid ocean pixel (avoids empty/NaN series near land)
    lat_name, lon_name, px_lat, px_lon, px_dist_km = pick_nearest_valid_ocean_pixel(
        good_files[len(good_files)//2],
        inlet_lat, inlet_lon,
        search_radius_km=SEARCH_RADIUS_KM,
        max_candidates=MAX_CANDIDATES
    )
    print(f"[POINT] inlet=({inlet_lat:.6f},{inlet_lon:.6f}) -> ocean pixel=({px_lat:.6f},{px_lon:.6f}), dist={px_dist_km:.3f} km")

    # 2) Extract daily SST series
    df = extract_point_timeseries(good_files, lat_name, lon_name, px_lat, px_lon)
    df = df[(df["time"] >= pd.to_datetime(T0)) & (df["time"] <= pd.to_datetime(T1))].copy()
    df = df.sort_values("time").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("Extracted series is empty. This indicates masking/coordinate mismatch still exists.")

    sst_daily = pd.Series(df["sst_c"].values, index=pd.to_datetime(df["time"]), name="sst_c").sort_index()
    sst_daily = sst_daily[~sst_daily.index.duplicated(keep="first")]

    # Save daily SST
    df_out = pd.DataFrame({"time": sst_daily.index, "sst_c": sst_daily.values})
    df_out.to_csv(CSV_DAILY, index=False)

    # 3) Monthly mean and anomalies
    sst_monthly = sst_daily.resample("MS").mean()
    clim_month, anom_month = monthly_climatology_anomaly(sst_monthly)

    # Extreme thresholds for monthly anomalies
    warm_thr95 = float(np.nanpercentile(anom_month.values, 95))
    cold_thr05 = float(np.nanpercentile(anom_month.values, 5))

    # Within-year cumulative monthly anomalies
    cum_df = within_year_cumulative_monthly(anom_month)

    # 4) Trend estimates
    # SST trend on monthly means (more stable than daily)
    x_year = (sst_monthly.index - sst_monthly.index[0]).days.values / 365.25
    slope_sst, intercept_sst, r, p_sst, _ = linregress(x_year, sst_monthly.values)

    # Theil-Sen slope (monthly SST)
    ts_slope_sst, ts_intercept_sst, _, _ = theilslopes(sst_monthly.values, x_year)

    # Monthly anomaly trend
    slope_an, intercept_an, r2, p_an, _ = linregress(x_year, anom_month.values)
    ts_slope_an, ts_intercept_an, _, _ = theilslopes(anom_month.values, x_year)

    # 5) Hobday MHW detection on daily SST
    clim_d, thr_d, delta_d = compute_hobday_threshold_and_clim(
        sst_daily, pct=MHW_PCT, doy_half_window=DOY_HALF_WINDOW
    )
    events, exceed_flag = detect_mhw_events(
        sst_daily, clim_d, thr_d, delta_d, min_duration=MIN_DURATION_D
    )
    evdf = events_to_df(events)
    evdf.to_csv(CSV_EVENTS, index=False)

    # Series used in MHW-intensity plots
    aligned_idx = sst_daily.index.intersection(clim_d.index).intersection(thr_d.index)
    intensity = (sst_daily.loc[aligned_idx] - clim_d.loc[aligned_idx]).rename("intensity_c")
    above_thr = (sst_daily.loc[aligned_idx] > thr_d.loc[aligned_idx])
    intensity_mhw = intensity.where(above_thr)

    # 6) Top events since 2015 by cumulative intensity
    ev2015 = evdf[evdf["start"] >= pd.Timestamp("2015-01-01")].copy()
    ev2015_top = ev2015.sort_values("cum_intensity_cdays", ascending=False).head(12).copy()

    tex = df_to_latex_table(
        ev2015_top,
        caption=("Top marine heatwave (MHW) events since 2015 at the inlet-nearest MUR GHRSST pixel, "
                 f"ranked by cumulative intensity (threshold={MHW_PCT}th percentile; minimum duration={MIN_DURATION_D} d)."),
        label="tab:mhw_events_2015_present_top"
    )
    with open(TEX_EVENTS_2015, "w", encoding="utf-8") as f:
        f.write(tex)

    # 7) Save monthly products
    pd.DataFrame({
        "time": sst_monthly.index,
        "sst_monthly_c": sst_monthly.values,
        "anom_monthly_c": anom_month.values
    }).to_csv(CSV_MONTHLY, index=False)

    # =============================================================================
    # Figure (clean 2x2 summary)
    # =============================================================================
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(14.5, 8.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1.25, 0.95],
                  hspace=0.28, wspace=0.22)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[0, 1])
    axD = fig.add_subplot(gs[1, 1])

    # (A) SST with MHW shading
    # Daily series in faint line
    axA.plot(sst_daily.index, sst_daily.values, linewidth=0.6, alpha=0.12, label="Daily SST")
    # Monthly means in bold
    axA.plot(sst_monthly.index, sst_monthly.values, linewidth=1.8, label="Monthly mean SST")

    # Monthly OLS trend line
    yfit = intercept_sst + slope_sst * x_year
    axA.plot(sst_monthly.index, yfit, linewidth=1.6, label=f"OLS trend: {slope_sst:+.3f} °C/yr")

    # Shade MHW events
    for e in events:
        axA.axvspan(e.start, e.end, alpha=0.10, linewidth=0)

    axA.set_title(f"(A) Inlet-nearest SST with Hobday MHW shading ({px_lat:.4f}°N, {px_lon:.4f}°E), 2003–2023")
    axA.set_ylabel("SST (°C)")
    axA.grid(True, alpha=0.25)
    axA.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.2, columnspacing=1.2)

    # (B) Monthly anomalies, extreme thresholds, and trend
    axB.axhline(0, linewidth=0.9)
    axB.bar(anom_month.index, anom_month.values, width=25, alpha=0.85, label="Monthly SST anomaly")

    # Extreme thresholds (95th and 5th percentile of anomalies)
    axB.axhline(warm_thr95, linestyle="--", linewidth=1.0, label="95th anomaly thr")
    axB.axhline(cold_thr05, linestyle="--", linewidth=1.0, label="5th anomaly thr")

    # Mark extreme months
    warm_m = anom_month[anom_month >= warm_thr95]
    cold_m = anom_month[anom_month <= cold_thr05]
    axB.scatter(warm_m.index, warm_m.values, marker="^", s=22)
    axB.scatter(cold_m.index, cold_m.values, marker="v", s=22)

    # Trend line
    yfit_an = intercept_an + slope_an * x_year
    axB.plot(anom_month.index, yfit_an, linewidth=1.6, label=f"Anom OLS: {slope_an:+.3f} °C/yr (p={p_an:.2g})")

    axB.set_title("(B) Monthly SST anomalies (relative to 2003–2023 monthly climatology)")
    axB.set_ylabel("Anomaly (°C)")
    axB.set_xlabel("Time")
    axB.grid(True, alpha=0.25)
    axB.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.2, columnspacing=1.2)

    # (C) Within-year cumulative monthly anomalies (reset each year)
    # Show earlier years faintly and highlight the last four years
    years = np.sort(cum_df["year"].unique())
    highlight_years = years[-4:] if years.size >= 4 else years

    for y in years:
        sub = cum_df[cum_df["year"] == y]
        if y in highlight_years:
            axC.plot(sub.index, sub["cum_within_year"].values, linewidth=2.0, label=str(y))
        else:
            axC.plot(sub.index, sub["cum_within_year"].values, linewidth=0.8, alpha=0.12)

    axC.axhline(0, linewidth=0.9)
    axC.set_title("(C) Within-year cumulative monthly anomalies (reset each year)")
    axC.set_ylabel("Cumulative anomaly (°C·month)")
    axC.grid(True, alpha=0.25)
    axC.legend(loc="upper left", frameon=False, ncol=1)

    # (D) MHW event intensity through time
    # Plot cumulative intensity as bars and peak intensity as points
    if not evdf.empty:
        mid = evdf["start"] + (evdf["end"] - evdf["start"]) / 2
        axD.bar(mid, evdf["cum_intensity_cdays"].values, width=20, alpha=0.75, label="Cum intensity (°C·d)")
        axD.scatter(mid, evdf["peak_intensity_c"].values, s=26, label="Peak intensity (°C)")
        axD.set_xlim(pd.Timestamp(T0), pd.Timestamp(T1))

    axD.set_title("(D) Hobday MHW events time series (cumulative & peak intensity)")
    axD.set_ylabel("Intensity metric")
    axD.set_xlabel("Time")
    axD.grid(True, alpha=0.25)
    axD.legend(loc="upper left", frameon=False)

    # Suptitle with key metrics
    n_events = len(events)
    fig.suptitle(
        f"Inlet SST + Hobday MHW diagnostics (MUR GHRSST, 2003–2023) | "
        f"SST trend={slope_sst:+.3f} °C/yr (p={p_sst:.3g}) | "
        f"MHW threshold={MHW_PCT}th pct; min duration={MIN_DURATION_D} d | "
        f"Events={n_events}",
        y=0.98, fontsize=12
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_PNG, dpi=350)
    fig.savefig(FIG_PDF, dpi=350)
    plt.show()

    print("\n[SAVED]")
    print("Daily CSV   :", CSV_DAILY)
    print("Monthly CSV :", CSV_MONTHLY)
    print("Events CSV  :", CSV_EVENTS)
    print("Events TEX  :", TEX_EVENTS_2015)
    print("Figure PNG  :", FIG_PNG)
    print("Figure PDF  :", FIG_PDF)

if __name__ == "__main__":
    main()
