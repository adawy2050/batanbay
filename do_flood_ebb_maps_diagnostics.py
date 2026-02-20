# -*- coding: utf-8 -*-
# Copyright Ahmed Eladawy

"""
Figure 6-style dissolved oxygen plot for 2024.

The figure has:
- Row 1: flood-tide maps (top/mid/bottom depth thirds)
- Row 2: ebb-tide maps (top/mid/bottom depth thirds)
- Row 3: spatial diagnostics (DO vs distance to inlet)

This version uses one shared map extent for all six maps so the panels are directly comparable.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point

import rasterio
from rasterio.plot import show as rioshow

from scipy.stats import spearmanr
from matplotlib.patches import Rectangle


# =========================
# Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "AAQ", "EXTRACTED2024")
BASEMAP_TIF = os.path.join(BASE_DIR, "data", "basemap", "L15-1720E-1090N.tif")

INLET_LAT = 11.603698
INLET_LON = 122.498061

DO_VMIN = 2.0
DO_VMAX = 7.0

FALLBACK_FLOOD_MIN_COMMENT = 49
FALLBACK_EBB_MIN_COMMENT = 1
FALLBACK_EBB_MAX_COMMENT = 41

MS_MAP = 90
EDGE_COLOR = "k"
EDGE_LW = 0.6

FLOOD_MARKER = "o"
EBB_MARKER = "^"
MS_DIAG = 35

# Basemap wash style (kept consistent with Fig. 5)
BASEMAP_ALPHA = 0.70
WHITE_WASH_ALPHA = 0.28

# Shared crop for all map panels
MAP_PAD_FRAC = 0.06  # 6% padding around station envelope (in map CRS)

# Diagnostics axis limits
X_LIM = (0.0, 13.5)     # km
Y_LIM = (3.4, 7.2)      # mg/L

OUT_DIR = os.path.join(BASE_DIR, "outputs", "do_figures")
OUT_FIG = os.path.join(OUT_DIR, "Fig6_DO_2024_Tide_DepthThirds_MapsPlusDiagnostics.png")
DPI = 400
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Helper functions
# =========================
def dms_to_dd_robust(x):
    if pd.isna(x):
        return np.nan
    try:
        val = float(str(x).strip())
    except Exception:
        return np.nan
    if abs(val) <= 90.0:
        return val
    deg = int(val // 100)
    minutes = val - deg * 100
    return deg + minutes / 60.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def extract_comment_number(comment):
    if pd.isna(comment):
        return np.nan
    s = str(comment).strip()
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan


def compute_depth_thirds_station_means(df_station, depth_col, do_col):
    dmax = df_station[depth_col].max()
    if not np.isfinite(dmax) or dmax <= 0:
        return {"top": np.nan, "mid": np.nan, "bot": np.nan}
    z1 = dmax / 3.0
    z2 = 2.0 * dmax / 3.0
    top = df_station.loc[df_station[depth_col] <= z1, do_col].mean()
    mid = df_station.loc[(df_station[depth_col] > z1) & (df_station[depth_col] <= z2), do_col].mean()
    bot = df_station.loc[df_station[depth_col] > z2, do_col].mean()
    return {"top": top, "mid": mid, "bot": bot}


def safe_spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan, np.nan
    rho, p = spearmanr(x[m], y[m])
    return rho, p


def ols_line(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x[m], y[m], 1)
    return slope, intercept


def pick_csv_with_required_columns(folder, required_cols):
    candidates = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    for p in candidates:
        try:
            tmp = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = set(tmp.columns)
        if all(rc in cols for rc in required_cols):
            return p

    msg = ["No CSV in DATA_DIR contains all required columns."]
    msg.append(f"Required: {required_cols}")
    msg.append("Found these CSVs and their columns (first 10 cols shown):")
    for p in candidates:
        try:
            tmp = pd.read_csv(p, nrows=5)
            msg.append(f" - {os.path.basename(p)}: {list(tmp.columns)[:10]}")
        except Exception:
            msg.append(f" - {os.path.basename(p)}: [unreadable]")
    raise KeyError("\n".join(msg))


# =========================
# Load and standardize data
# =========================
LAT_COL = "Latitude"
LON_COL = "Longitude"
DEPTH_COL = "Depth [m]"
DO_COL = "G&G-DO"
COMMENT_COL = "Comment"

required_cols = [LAT_COL, LON_COL, DEPTH_COL, DO_COL, COMMENT_COL]
CSV_PATH = pick_csv_with_required_columns(DATA_DIR, required_cols)

df = pd.read_csv(CSV_PATH)

df["lat_dd"] = df[LAT_COL].apply(dms_to_dd_robust)
df["lon_dd"] = df[LON_COL].apply(dms_to_dd_robust)

df[DEPTH_COL] = pd.to_numeric(df[DEPTH_COL], errors="coerce")
df[DO_COL] = pd.to_numeric(df[DO_COL], errors="coerce")

df["station_id"] = df[COMMENT_COL].astype(str).str.strip()
df["comment_num"] = df[COMMENT_COL].apply(extract_comment_number)

df = df.dropna(subset=["lat_dd", "lon_dd", DEPTH_COL, DO_COL, "station_id"])


# =========================
# Split flood and ebb subsets
# =========================
tide_col = None
for c in df.columns:
    if c.strip().lower() in ["tide", "tidalphase", "phase"]:
        tide_col = c
        break

if tide_col is not None:
    tide_vals = df[tide_col].astype(str).str.lower()
    flood_df = df[tide_vals.str.contains("flood")].copy()
    ebb_df = df[tide_vals.str.contains("ebb")].copy()
else:
    flood_df = df[df["comment_num"] >= FALLBACK_FLOOD_MIN_COMMENT].copy()
    ebb_df = df[(df["comment_num"] >= FALLBACK_EBB_MIN_COMMENT) & (df["comment_num"] <= FALLBACK_EBB_MAX_COMMENT)].copy()

if flood_df.empty or ebb_df.empty:
    raise RuntimeError(
        "Flood/Ebb split produced empty set.\n"
        "Check Tide column or Comment ranges.\n"
        f"Flood rows={len(flood_df)}, Ebb rows={len(ebb_df)}\n"
        f"CSV used: {CSV_PATH}"
    )


# =========================
# Aggregate per station and depth third
# =========================
def build_station_thirds_table(df_phase, phase_name):
    records = []
    for sid in sorted(df_phase["station_id"].unique()):
        dsi = df_phase[df_phase["station_id"] == sid].copy()

        lat = dsi["lat_dd"].iloc[0]
        lon = dsi["lon_dd"].iloc[0]
        thirds = compute_depth_thirds_station_means(dsi, DEPTH_COL, DO_COL)
        dist_km = haversine_km(lat, lon, INLET_LAT, INLET_LON)

        records.append({
            "phase": phase_name,
            "station_id": sid,
            "lat_dd": lat,
            "lon_dd": lon,
            "dist_km": dist_km,
            "do_top": thirds["top"],
            "do_mid": thirds["mid"],
            "do_bot": thirds["bot"],
        })
    return pd.DataFrame.from_records(records)

tab_flood = build_station_thirds_table(flood_df, "FLOOD")
tab_ebb = build_station_thirds_table(ebb_df, "EBB")

for col in ["do_top", "do_mid", "do_bot"]:
    tab_flood[col] = tab_flood[col].clip(DO_VMIN, DO_VMAX)
    tab_ebb[col] = tab_ebb[col].clip(DO_VMIN, DO_VMAX)


# =========================
# Load basemap and project station coordinates
# =========================
with rasterio.open(BASEMAP_TIF) as src:
    raster_crs = src.crs
    raster_bounds = src.bounds

gdf_flood = gpd.GeoDataFrame(
    tab_flood,
    geometry=[Point(xy) for xy in zip(tab_flood["lon_dd"], tab_flood["lat_dd"])],
    crs="EPSG:4326"
).to_crs(raster_crs)

gdf_ebb = gpd.GeoDataFrame(
    tab_ebb,
    geometry=[Point(xy) for xy in zip(tab_ebb["lon_dd"], tab_ebb["lat_dd"])],
    crs="EPSG:4326"
).to_crs(raster_crs)

# Compute one shared crop extent from flood + ebb points
geom_all = pd.concat([gdf_flood.geometry, gdf_ebb.geometry], ignore_index=True)
xmin, ymin, xmax, ymax = gpd.GeoSeries(geom_all, crs=raster_crs).total_bounds

dx = max(xmax - xmin, 1.0)
dy = max(ymax - ymin, 1.0)
padx = dx * MAP_PAD_FRAC
pady = dy * MAP_PAD_FRAC

crop_left = max(xmin - padx, raster_bounds.left)
crop_right = min(xmax + padx, raster_bounds.right)
crop_bottom = max(ymin - pady, raster_bounds.bottom)
crop_top = min(ymax + pady, raster_bounds.top)

MAP_EXTENT = (crop_left, crop_right, crop_bottom, crop_top)


# =========================
# Plot: 3 rows x 3 columns
# =========================
fig = plt.figure(figsize=(16, 15.5))
gs = fig.add_gridspec(
    nrows=3, ncols=3,
    height_ratios=[1.0, 1.0, 1.20],
    hspace=0.28, wspace=0.08
)

third_keys = [("do_top", "Top Third"), ("do_mid", "Middle Third"), ("do_bot", "Bottom Third")]
cmap = plt.colormaps.get_cmap("viridis")
norm = plt.Normalize(DO_VMIN, DO_VMAX)

def draw_map(ax, gdf, col, title):
    with rasterio.open(BASEMAP_TIF) as src:
        rioshow(src, ax=ax, alpha=BASEMAP_ALPHA, zorder=0)

    left, right, bottom, top = MAP_EXTENT

    ax.add_patch(
        Rectangle((left, bottom), right - left, top - bottom,
                  facecolor="white", edgecolor="none",
                  alpha=WHITE_WASH_ALPHA, zorder=1)
    )

    ax.scatter(
        gdf.geometry.x, gdf.geometry.y,
        c=gdf[col].values, cmap=cmap, norm=norm,
        s=MS_MAP, edgecolors=EDGE_COLOR, linewidths=EDGE_LW,
        zorder=5
    )

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

# Row 1: flood maps
axes_row1 = []
for j, (col, name) in enumerate(third_keys):
    ax = fig.add_subplot(gs[0, j])
    draw_map(ax, gdf_flood, col, f"Flood {name}")
    axes_row1.append(ax)

# Row 2: ebb maps
axes_row2 = []
for j, (col, name) in enumerate(third_keys):
    ax = fig.add_subplot(gs[1, j])
    draw_map(ax, gdf_ebb, col, f"Ebb {name}")
    axes_row2.append(ax)

# Colorbars
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar1 = fig.colorbar(sm, ax=axes_row1, orientation="vertical", fraction=0.025, pad=0.015)
cbar1.set_label("Flood Average DO [mg/L]", fontsize=11)

cbar2 = fig.colorbar(sm, ax=axes_row2, orientation="vertical", fraction=0.025, pad=0.015)
cbar2.set_label("Ebb Average DO [mg/L]", fontsize=11)

# Diagnostics
def draw_diag(ax, flood_tab, ebb_tab, col, title, show_legend=False, show_ylabel=True, show_yticklabels=True):
    xF, yF = flood_tab["dist_km"].values, flood_tab[col].values
    xE, yE = ebb_tab["dist_km"].values, ebb_tab[col].values

    ax.scatter(xF, yF, s=MS_DIAG, marker=FLOOD_MARKER, edgecolors="k", linewidths=0.4, alpha=0.9, label="FLOOD")
    ax.scatter(xE, yE, s=MS_DIAG, marker=EBB_MARKER, edgecolors="k", linewidths=0.4, alpha=0.9, label="EBB")

    mF, bF = ols_line(xF, yF)
    mE, bE = ols_line(xE, yE)
    xx = np.linspace(X_LIM[0], X_LIM[1], 100)

    if np.isfinite(mF):
        ax.plot(xx, mF * xx + bF, linewidth=1.7)
    if np.isfinite(mE):
        ax.plot(xx, mE * xx + bE, linewidth=1.7, linestyle="--")

    rhoF, pF = safe_spearman(xF, yF)
    rhoE, pE = safe_spearman(xE, yE)

    txt = (
        f"$\\rho_F$={rhoF:+.2f}, $p_F$={pF:.1e}\n"
        f"$\\rho_E$={rhoE:+.2f}, $p_E$={pE:.1e}"
    )
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0)
    )

    ax.set_title(title, fontsize=12, pad=14)
    ax.set_xlabel("Distance to inlet (km)", fontsize=11)

    if show_ylabel:
        ax.set_ylabel("Average DO (mg/L)", fontsize=11)
    else:
        ax.set_ylabel("")

    if not show_yticklabels:
        ax.tick_params(axis="y", labelleft=False)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.grid(True, alpha=0.25)

    if show_legend:
        ax.legend(frameon=False, loc="upper right")

for j, (col, name) in enumerate(third_keys):
    ax = fig.add_subplot(gs[2, j])
    is_left = (j == 0)
    draw_diag(
        ax, tab_flood, tab_ebb, col, f"DO vs distance — {name}",
        show_legend=(j == 2),
        show_ylabel=is_left,
        show_yticklabels=is_left
    )

# Row labels
fig.text(0.015, 0.82, "FLOOD", rotation=90, fontsize=13, fontweight="bold", va="center")
fig.text(0.015, 0.50, "EBB", rotation=90, fontsize=13, fontweight="bold", va="center")
fig.text(0.015, 0.19, "SPATIAL\nDIAGNOSTICS", rotation=90, fontsize=12, fontweight="bold", va="center")

fig.suptitle(
    "Depth-stratified dissolved oxygen (DO): tide-resolved maps and longitudinal diagnostics (Batan Estuary, 2024)",
    fontsize=14, y=0.985
)

plt.savefig(OUT_FIG, dpi=DPI, bbox_inches="tight")
plt.show()

print("Saved:", OUT_FIG)
print("CSV used:", CSV_PATH)
print("Shared MAP_EXTENT (left, right, bottom, top):", MAP_EXTENT)
