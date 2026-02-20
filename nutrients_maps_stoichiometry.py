# Copyright Ahmed Eladawy

import os
import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

import cartopy.crs as ccrs
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
tiff_path = os.path.join(DATA_DIR, "basemap", "L15-1720E-1090N.tif")

# Inlet coordinates given as (lat, lon)
inlet_lat, inlet_lon = 11.596509, 122.492519

# If True, replace PO4 <= 0 with a small epsilon for DIN:PO4 ratio calculations.
# This removes open circles in the ratio map but adds an assumption.
SUBSTITUTE_NONPOSITIVE_PO4_FOR_RATIO = False
PO4_EPS_UMOL = 0.05  # µmol P/L used if substitution is enabled (choose conservatively)

# Robust color scaling (percentiles) to prevent single outliers from compressing the colormap
ROBUST_LIMITS = True
ROBUST_PCT = (2, 98)

# Output
out_dir = os.path.join(BASE_DIR, "outputs", "nutrients")
fig_dpi = 450

# ============================================================
# Data (ordered list, not a set)
# ============================================================
SAMPLES = [f"PH{i}" for i in range(21, 52)] + [f"PH{i}" for i in range(1, 21)]

NO2 = np.array([0.067160098,-0.018266329,0.033051532,-0.014772398,-0.023410952,0.02010359,0.093292164,0.018675482,0.14521133,0.107330532,0.272745052,0.24230579,0.048764247,0.155445617,-0.020041715,0.001052214,-0.028339577,0.019893507,-0.024022398,0.07605798,0.007667156,-0.042548615,-0.031475883,0.023419664,0.047146474,-0.023164225,0.174380627,0.26028864,0.015670208,0.303496395,-0.029161627,0.296906222,0.22294393,0.185984476,0.431705756,0.010174001,0.430270678,0.11188263,0.029055953,-0.018014868,0.486358292,0.412691107,0.347358245,0.12232264,-0.000384549,-0.024171064,0.226373827,-0.017575917,0.141343291,0.104804536,0.057968484])

NH4 = np.array([0.185162038,-0.181688988,2.354342003,0.349827423,0.170968443,-0.318140479,2.568665303,2.450880516,1.075550821,-0.424894912,0.350050038,-0.159556426,-0.583292982,-0.031048122,0.159506422,-0.035561239,-0.326744196,-0.265955499,-0.194566948,-0.323433898,0.049644257,-0.54542513,-0.356427305,0.151374813,-0.502356344,-0.278854449,0.244055749,2.25476784,0.327256019,4.678951621,-0.330743536,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

NO2_NO3 = np.array([0.465390471,0.017305028,0.694925516,-0.019667519,0.090939299,0.197646862,0.719936768,1.140139795,0.934403824,0.485598479,1.999859383,2.275723192,0.515848582,0.972372945,0.199554388,0.325191357,-0.058966498,0.579878295,0.001225059,0.388799566,0.525079612,-0.072551097,-0.01148729,0.68797878,0.205010546,0.273719631,2.287181145,3.865358709,0.272974001,3.247446813,0.243922968,1.440990348,1.041154363,1.2771256,3.333640036,0.601960796,3.541594773,0.783040135,0.783647519,0.264595679,4.506455664,3.682469643,2.838615005,0.386579742,0.534143163,0.012192234,1.310794701,0.16658223,0.982733213,0.651689685,0.688636145])

PO4 = np.array([0.196543309,0.010486156,-0.03698828,0.028519579,-0.027045705,0.119822716,0.387432206,-0.019592084,0.133762906,0.170644479,0.367858199,0.179046679,0.108264498,0.246275588,0.049421351,0.129936246,0.110601306,0.118496909,0.112273668,0.400506006,0.048024512,0.059444741,0.116281032,-0.002484617,0.185617891,0.394015816,0.400489682,0.423133656,0.246684631,1.142757829,0.413273021,0.231232371,0.237040027,0.554120598,0.633908519,-0.001428937,0.615574346,0.18691697,-0.017433434,0.027881498,0.635917277,0.51918586,0.388930404,0.149254005,0.049375595,0.111356215,0.252118846,0.143117564,0.597824222,0.220278808,-0.003134197])

SiO2 = np.array([6.933348862,3.061625535,2.379610696,2.724025238,2.214954867,8.137730707,15.91334155,6.270774466,8.357506867,10.0395616,23.83218269,30.34811334,32.73617902,25.66305167,26.99719574,29.04165398,21.69581752,36.66706376,28.90494435,20.4468618,18.86840081,26.36186789,18.35457299,15.91515387,18.96801904,21.62533962,29.86295354,52.8252343,21.72072245,55.27699305,38.93839062,22.5902173,22.62010919,22.65010678,22.68021063,22.71042131,19.97728086,8.111129892,21.3464348,21.5473139,21.35905824,16.4625882,15.15273057,18.27907423,19.98014836,22.54946282,12.36605391,22.3957684,11.52335621,5.939024563,2.411355436])

NO3 = np.array([0.397514206,0.03489186,0.658161796,-0.005314734,0.113579797,0.176644466,0.624107917,1.110226066,0.786792704,0.37888583,1.719388335,2.022221946,0.46466739,0.815511864,0.21690609,0.322903576,-0.030804373,0.560831008,0.025410321,0.312769289,0.52030115,-0.029283956,0.019555403,0.666563444,0.158836025,0.298688831,2.113324873,3.606960488,0.258263169,2.941786445,0.272310811,1.144084126,0.818210433,1.091141124,2.90193428,0.591786795,3.111324095,0.671157505,0.754591566,0.282610547,4.020097373,3.269778536,2.49125676,0.264257102,0.534527712,0.036363298,1.084420873,0.184158147,0.841389922,0.546885149,0.630667661])

Lat = np.array([11.575491,11.589536,11.59897,11.613281,11.601877,11.60379,11.591708,11.583148,11.59061,11.602786,11.609179,11.609071,11.608669,11.622434,11.624364,11.5906,11.60102,11.596803,11.60401,11.617858,11.621319,11.651851,11.653142,11.645174,11.635551,11.635925,11.634218,11.633879,11.615901,11.614584,11.611412,11.63357,11.63818,11.64629,11.65018,11.6443,11.63893,11.63481,11.62801,11.62148,11.574538,11.576926,11.574352,11.572148,11.564623,11.559806,11.558274,11.558422,11.562255,11.604024,11.61082])

Lon = np.array([122.492891,122.491819,122.49434,122.492076,122.499438,122.51008,122.487343,122.4787,122.471365,122.471807,122.450569,122.443848,122.440228,122.435937,122.426047,122.43986,122.42996,122.423967,122.41643,122.427856,122.413589,122.393809,122.408788,122.408486,122.396703,122.408292,122.412004,122.408047,122.407781,122.38256,122.394101,122.41728,122.41652,122.41584,122.42347,122.43062,122.43263,122.43606,122.44161,122.44855,122.446321,122.451914,122.456838,122.465624,122.488586,122.451098,122.466785,122.479122,122.477287,122.459473,122.463607])

# ============================================================
# Helper functions
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km)."""
    R = 6371.0
    lat1r = np.deg2rad(lat1); lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2); lon2r = np.deg2rad(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def robust_vmin_vmax(x, pct=(2,98)):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    return np.nanpercentile(x, pct[0]), np.nanpercentile(x, pct[1])

def add_scalebar_km(ax, length_km=2.0, loc=(0.08, 0.10), lw=3, text_offset=0.02):
    """
    Draw a simple scale bar in PlateCarree using an approximate conversion at mid-latitude.
    loc is in axes fraction.
    """
    xmin, xmax, ymin, ymax = ax.get_extent(ccrs.PlateCarree())
    x0 = xmin + (xmax-xmin)*loc[0]
    y0 = ymin + (ymax-ymin)*loc[1]

    # Approximate meters per degree longitude at this latitude
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(y0))
    deg_len = (length_km*1000.0)/m_per_deg_lon

    ax.plot([x0, x0+deg_len], [y0, y0], transform=ccrs.PlateCarree(), lw=lw, color="k", solid_capstyle="butt")
    ax.text(x0, y0 + (ymax-ymin)*text_offset, f"{int(length_km)} km", transform=ccrs.PlateCarree(),
            ha="left", va="bottom", fontsize=9, color="k")

# ============================================================
# Unit conversions and derived variables
# ============================================================
mgN_to_umolN = 1000.0 / 14.0
mgP_to_umolP = 1000.0 / 31.0

NO2_umol = NO2 * mgN_to_umolN
NO3_umol = NO3 * mgN_to_umolN
NH4_umol = NH4 * mgN_to_umolN
PO4_umol = PO4 * mgP_to_umolP

# DIN in umol N/L (nitrogen species only)
DIN_umol = NO2_umol + NO3_umol + NH4_umol

# Distance to inlet (km)
dist_km = haversine_km(Lat, Lon, inlet_lat, inlet_lon)

# DIN:PO4 ratio (molar)
PO4_for_ratio = PO4_umol.copy()
ratio_mask = np.isfinite(DIN_umol) & np.isfinite(PO4_for_ratio)

if SUBSTITUTE_NONPOSITIVE_PO4_FOR_RATIO:
    PO4_for_ratio = np.where((PO4_for_ratio <= 0) & ratio_mask, PO4_EPS_UMOL, PO4_for_ratio)
    DIN_PO4 = np.where(ratio_mask & (PO4_for_ratio != 0), DIN_umol / PO4_for_ratio, np.nan)
    ratio_undefined = np.zeros_like(DIN_PO4, dtype=bool)  # none shown as undefined
else:
    # Ratio is undefined where PO4 <= 0
    ratio_undefined = ratio_mask & (PO4_for_ratio <= 0)
    DIN_PO4 = np.where(ratio_mask & (PO4_for_ratio > 0), DIN_umol / PO4_for_ratio, np.nan)

# ============================================================
# Load and reproject basemap
# ============================================================
with rasterio.open(tiff_path) as src:
    dst_crs = "EPSG:4326"
    transform, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    dst_data = np.empty((src.count, h, w), dtype=src.dtypes[0])

    for b in range(1, src.count + 1):
        reproject(
            source=rasterio.band(src, b),
            destination=dst_data[b-1],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

# Use RGB if available, otherwise single-band raster
if dst_data.shape[0] >= 3:
    basemap_img = np.dstack([dst_data[0], dst_data[1], dst_data[2]])
else:
    basemap_img = dst_data[0]

xmin = transform.c
xmax = xmin + w * transform.a
ymax = transform.f
ymin = ymax + h * transform.e
raster_extent = [xmin, xmax, ymin, ymax]

# Map extent focused on station locations
lon_min, lon_max = Lon.min(), Lon.max()
lat_min, lat_max = Lat.min(), Lat.max()
lon_buf = (lon_max - lon_min) * 0.07
lat_buf = (lat_max - lat_min) * 0.07
data_extent = [lon_min - lon_buf, lon_max + lon_buf, lat_min - lat_buf, lat_max + lat_buf]

# ============================================================
# Figure 1: six nutrient maps (filled points only)
# ============================================================
params_maps = [
    ("NO₂ (mg L⁻¹)", NO2),
    ("NH₄ (µmol L⁻¹)", NH4_umol),
    ("NO₂+NO₃ (mg L⁻¹)", NO2_NO3),
    ("PO₄ (µmol L⁻¹)", PO4_umol),
    ("SiO₂ (mg L⁻¹)", SiO2),
    ("NO₃ (µmol L⁻¹)", NO3_umol),
]

fig, axes = plt.subplots(
    2, 3, figsize=(16, 9),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)

panel_letters = list("abcdef")

for ax, (lab, vals), p in zip(axes.flat, params_maps, panel_letters):
    ax.imshow(
        basemap_img, origin="upper",
        extent=raster_extent, transform=ccrs.PlateCarree(),
        alpha=0.35
    )
    ax.set_extent(data_extent, crs=ccrs.PlateCarree())

    # Plot all finite values, including negatives and zeros
    finite = np.isfinite(vals)

    if ROBUST_LIMITS:
        vmin, vmax = robust_vmin_vmax(vals[finite], ROBUST_PCT)
    else:
        vmin, vmax = np.nanmin(vals[finite]), np.nanmax(vals[finite])

    norm = Normalize(vmin=vmin, vmax=vmax)

    ax.scatter(
        Lon[finite], Lat[finite],
        c=vals[finite], cmap="viridis", norm=norm,
        s=55, edgecolor="k", linewidth=0.5,
        transform=ccrs.PlateCarree(), zorder=3
    )

    # Inlet marker
    ax.scatter([inlet_lon], [inlet_lat], marker="^", s=110, facecolor="white",
               edgecolor="k", linewidth=1.2, transform=ccrs.PlateCarree(), zorder=5)
    ax.text(inlet_lon, inlet_lat, " Inlet", transform=ccrs.PlateCarree(),
            fontsize=10, va="center", ha="left")

    # Draw scale bar on the first panel only
    if p == "a":
        add_scalebar_km(ax, 2.0)

    ax.set_title(lab, fontsize=12)
    ax.text(0.01, 0.98, p, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold")

    # Horizontal colorbar per panel
    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array(vals[finite])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, fraction=0.06)
    cbar.set_label(lab, fontsize=10)

    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Spatial distribution of dissolved nutrients across Batan Estuary stations", fontsize=15, y=1.02)

# Save figure
os.makedirs(out_dir, exist_ok=True)
fig1_png = os.path.join(out_dir, "Fig_Nutrients_Maps.png")
fig1_pdf = os.path.join(out_dir, "Fig_Nutrients_Maps.pdf")
fig.savefig(fig1_png, dpi=fig_dpi, bbox_inches="tight")
fig.savefig(fig1_pdf, dpi=fig_dpi, bbox_inches="tight")
plt.show()

print("[OK] Saved:", fig1_png)
print("[OK] Saved:", fig1_pdf)

# ============================================================
# Figure 2: stoichiometry diagnostics (2 panels)
# Left: DIN vs PO4, colored by distance to inlet
# Right: DIN:PO4 map; points with PO4 <= 0 are shown as open circles
# ============================================================
fig2 = plt.figure(figsize=(16, 6), constrained_layout=True)
gs = fig2.add_gridspec(1, 2, width_ratios=[1.05, 1.0])

# Panel a: DIN vs PO4
axa = fig2.add_subplot(gs[0, 0])

finite_np = np.isfinite(DIN_umol) & np.isfinite(PO4_umol)
# Keep all finite points in the scatter; draw Redfield line only for x >= 0
sc = axa.scatter(
    PO4_umol[finite_np], DIN_umol[finite_np],
    c=dist_km[finite_np], cmap="viridis",
    s=70, edgecolor="k", linewidth=0.5
)

# Redfield reference (DIN = 16 * PO4), shown only for x >= 0
x_ref = np.linspace(max(0, np.nanmin(PO4_umol[finite_np])), np.nanmax(PO4_umol[finite_np]), 200)
axa.plot(x_ref, 16.0*x_ref, ls="--", lw=2.0, color="k")
axa.text(0.02, 0.05, "Redfield N:P = 16", transform=axa.transAxes, fontsize=10)

axa.set_xlabel("PO₄ (µmol P L⁻¹)")
axa.set_ylabel("DIN (µmol N L⁻¹)")
axa.set_title("N–P structure (colored by distance to inlet)")
cbar = plt.colorbar(sc, ax=axa)
cbar.set_label("Distance to inlet (km)")

axa.grid(True, alpha=0.25)
axa.text(0.01, 0.98, "a", transform=axa.transAxes, ha="left", va="top",
         fontsize=12, fontweight="bold")

# Panel b: DIN:PO4 map
axb = fig2.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

axb.imshow(
    basemap_img, origin="upper",
    extent=raster_extent, transform=ccrs.PlateCarree(),
    alpha=0.35
)
axb.set_extent(data_extent, crs=ccrs.PlateCarree())

finite_ratio = np.isfinite(DIN_PO4)

if ROBUST_LIMITS:
    vmin_r, vmax_r = robust_vmin_vmax(DIN_PO4[finite_ratio], ROBUST_PCT)
else:
    vmin_r, vmax_r = np.nanmin(DIN_PO4[finite_ratio]), np.nanmax(DIN_PO4[finite_ratio])

norm_r = Normalize(vmin=vmin_r, vmax=vmax_r)

# Colored points where the ratio is defined
axb.scatter(
    Lon[finite_ratio], Lat[finite_ratio],
    c=DIN_PO4[finite_ratio], cmap="viridis", norm=norm_r,
    s=55, edgecolor="k", linewidth=0.5,
    transform=ccrs.PlateCarree(), zorder=3
)

# Show undefined ratios (PO4 <= 0) as open circles
if np.any(ratio_undefined):
    axb.scatter(
        Lon[ratio_undefined], Lat[ratio_undefined],
        facecolor="none", edgecolor="0.35", linewidth=1.0,
        s=55, transform=ccrs.PlateCarree(), zorder=4,
        label="PO₄ ≤ 0 (ratio undefined)"
    )
    axb.legend(loc="lower left", frameon=True, fontsize=9)

# Inlet marker and scale bar
axb.scatter([inlet_lon], [inlet_lat], marker="^", s=110, facecolor="white",
            edgecolor="k", linewidth=1.2, transform=ccrs.PlateCarree(), zorder=5)
axb.text(inlet_lon, inlet_lat, " Inlet", transform=ccrs.PlateCarree(), fontsize=10, va="center", ha="left")
add_scalebar_km(axb, 2.0)

axb.set_title("Spatial pattern of DIN:PO₄ (molar)")
axb.text(0.01, 0.98, "b", transform=axb.transAxes, ha="left", va="top",
         fontsize=12, fontweight="bold")
axb.set_xticks([]); axb.set_yticks([])

smr = ScalarMappable(norm=norm_r, cmap="viridis")
smr.set_array(DIN_PO4[finite_ratio])
cbr = plt.colorbar(smr, ax=axb, orientation="vertical", fraction=0.045, pad=0.02)
cbr.set_label("DIN:PO₄ (molar)")

fig2.suptitle("Nutrient stoichiometry diagnostics (pre-PCA)", fontsize=15)

fig2_png = os.path.join(out_dir, "Fig_Nutrients_Stoichiometry_prePCA.png")
fig2_pdf = os.path.join(out_dir, "Fig_Nutrients_Stoichiometry_prePCA.pdf")
fig2.savefig(fig2_png, dpi=fig_dpi, bbox_inches="tight")
fig2.savefig(fig2_pdf, dpi=fig_dpi, bbox_inches="tight")
plt.show()

print("[OK] Saved:", fig2_png)
print("[OK] Saved:", fig2_pdf)

# ============================================================
# Quick QC printout to explain open-circle counts in the ratio map
# ============================================================
print("\nQC (counts):")
print("N stations:", len(SAMPLES))
print("PO4 <= 0 (µmol):", int(np.sum(np.isfinite(PO4_umol) & (PO4_umol <= 0))))
print("DIN finite:", int(np.sum(np.isfinite(DIN_umol))))
print("DIN:PO4 defined:", int(np.sum(np.isfinite(DIN_PO4))))
print("DIN:PO4 undefined (PO4<=0):", int(np.sum(ratio_undefined)))
