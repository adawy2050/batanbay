# -*- coding: utf-8 -*-
# Copyright Ahmed Eladawy

"""
AAQ and nutrient analysis for Batan Bay (v5.2).

This script builds the main and supplementary figures and exports all analysis tables.
Key updates in this version:
- Filters AAQ and nutrient points to the basemap extent and exports removed points.
- Enforces one-to-one AAQ station matching on the nutrient side (closest match kept).
- Uses a 4-panel main figure and a 4-panel supplementary figure.
- Adds optional block permutation (within regimes) to reduce spatial-dependence bias.

Main outputs:
- {OUT_DIR}/figures/AAQ_nutrients_MAIN_v5_2_k{K_BEST}.png/pdf
- {OUT_DIR}/figures/AAQ_nutrients_SUPP_v5_2_k{K_BEST}.png/pdf
- {OUT_DIR}/tables/aaq_station_features_all.csv
- {OUT_DIR}/tables/aaq_out_of_domain_stations.csv (if any removed)
- {OUT_DIR}/tables/nutr_out_of_domain_points.csv (if any removed)
- Additional matched, sensitivity, variation-partitioning, partial-Spearman, and regime-fingerprint tables.
"""

import os
import warnings
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.spatial import ConvexHull

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import SplineTransformer

import matplotlib.pyplot as plt

# Basemap
import rasterio
from rasterio.plot import show as rioshow
from rasterio.warp import transform as rio_transform


# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AAQ_DIR = os.path.join(BASE_DIR, "data", "AAQ", "EXTRACTED2024")

TEMP_CSV = os.path.join(AAQ_DIR, "extracted_temp_data.csv")
SAL_CSV  = os.path.join(AAQ_DIR, "extracted_sal_data.csv")
CHLA_CSV = os.path.join(AAQ_DIR, "extracted_chla_data.csv")
TURB_CSV = os.path.join(AAQ_DIR, "extracted_turb_data.csv")

# Basemap GeoTIFF used for plotting and domain filtering
BASEMAP_TIF = os.path.join(BASE_DIR, "data", "basemap", "L15-1720E-1090N.tif")

INLET_LATLON = (11.596509, 122.492519)  # (lat, lon)

OUT_DIR = os.path.join(BASE_DIR, "outputs", "aaq_nutrients")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

RANDOM_SEED = 42

# Smoothing
K_BEST = 7
K_LIST = [1, 3, 5, 7]

# Number of AAQ-state regimes for clustering
N_REGIMES = 4

# Nutrient pairing settings
USE_N_CLOSEST = True
N_CLOSEST = 15
MAX_DIST_KM = 2.0

# Enforce one-to-one AAQ station matching in the paired dataset
ENFORCE_UNIQUE_AAQ_STATIONS = True

# Distance thresholds (km) used in sensitivity checks
MATCH_THRESHOLDS = [0.25, 0.5, 1.0, 2.0, None]

# permutation settings
N_PERM_NAIVE = 999
N_PERM_SENS  = 499

# Optional: block permutation within regimes to reduce inflated significance under spatial dependence
DO_BLOCK_PERM = True
N_PERM_BLOCK = 999

# Domain filter: remove points outside basemap bounds (recommended)
FILTER_TO_BASEMAP_EXTENT = True
BASEMAP_BUFFER = 0.0  # in CRS units (meters for projected CRS); keep 0 unless needed

VERSION_TAG = "v5_2"


# =============================================================================
# Nutrient arrays (input values)
# =============================================================================

NO2 = [0.067160098, -0.018266329, 0.033051532, -0.014772398, -0.023410952, 0.02010359,
       0.093292164, 0.018675482, 0.14521133, 0.107330532, 0.272745052, 0.24230579,
       0.048764247, 0.155445617, -0.020041715, 0.001052214, -0.028339577, 0.019893507,
       -0.024022398, 0.07605798, 0.007667156, -0.042548615, -0.031475883, 0.023419664,
       0.047146474, -0.023164225, 0.174380627, 0.26028864, 0.015670208, 0.303496395,
       -0.029161627, 0.296906222, 0.22294393, 0.185984476, 0.431705756, 0.010174001,
       0.430270678, 0.11188263, 0.029055953, -0.018014868, 0.486358292, 0.412691107,
       0.347358245, 0.12232264, -0.000384549, -0.024171064, 0.226373827, -0.017575917,
       0.141343291, 0.104804536, 0.057968484]

NH4 = [0.185162038, -0.181688988, 2.354342003, 0.349827423, 0.170968443, -0.318140479,
       2.568665303, 2.450880516, 1.075550821, -0.424894912, 0.350050038, -0.159556426,
       -0.583292982, -0.031048122, 0.159506422, -0.035561239, -0.326744196, -0.265955499,
       -0.194566948, -0.323433898, 0.049644257, -0.54542513, -0.356427305, 0.151374813,
       -0.502356344, -0.278854449, 0.244055749, 2.25476784, 0.327256019, 4.678951621,
       -0.330743536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

NO2_NO3 = [0.465390471, 0.017305028, 0.694925516, -0.019667519, 0.090939299, 0.197646862,
           0.719936768, 1.140139795, 0.934403824, 0.485598479, 1.999859383, 2.275723192,
           0.515848582, 0.972372945, 0.199554388, 0.325191357, -0.058966498, 0.579878295,
           0.001225059, 0.388799566, 0.525079612, -0.072551097, -0.01148729, 0.68797878,
           0.205010546, 0.273719631, 2.287181145, 3.865358709, 0.272974001, 3.247446813,
           0.243922968, 1.440990348, 1.041154363, 1.2771256, 3.333640036, 0.601960796,
           3.541594773, 0.783040135, 0.783647519, 0.264595679, 4.506455664, 3.682469643,
           2.838615005, 0.386579742, 0.534143163, 0.012192234, 1.310794701, 0.16658223,
           0.982733213, 0.651689685, 0.688636145]

PO4 = [0.196543309, 0.010486156, -0.03698828, 0.028519579, -0.027045705, 0.119822716,
       0.387432206, -0.019592084, 0.133762906, 0.170644479, 0.367858199, 0.179046679,
       0.108264498, 0.246275588, 0.049421351, 0.129936246, 0.110601306, 0.118496909,
       0.112273668, 0.400506006, 0.048024512, 0.059444741, 0.116281032, -0.002484617,
       0.185617891, 0.394015816, 0.400489682, 0.423133656, 0.246684631, 1.142757829,
       0.413273021, 0.231232371, 0.237040027, 0.554120598, 0.633908519, -0.001428937,
       0.615574346, 0.18691697, -0.017433434, 0.027881498, 0.635917277, 0.51918586,
       0.388930404, 0.149254005, 0.049375595, 0.111356215, 0.252118846, 0.143117564,
       0.597824222, 0.220278808, -0.003134197]

SiO2 = [6.933348862, 3.061625535, 2.379610696, 2.724025238, 2.214954867, 8.137730707,
        15.91334155, 6.270774466, 8.357506867, 10.0395616, 23.83218269, 30.34811334,
        32.73617902, 25.66305167, 26.99719574, 29.04165398, 21.69581752, 36.66706376,
        28.90494435, 20.4468618, 18.86840081, 26.36186789, 18.35457299, 15.91515387,
        18.96801904, 21.62533962, 29.86295354, 52.8252343, 21.72072245, 55.27699305,
        38.93839062, 22.5902173, 22.62010919, 22.65010678, 22.68021063, 22.71042131,
        19.97728086, 8.111129892, 21.3464348, 21.5473139, 21.35905824, 16.4625882,
        15.15273057, 18.27907423, 19.98014836, 22.54946282, 12.36605391, 22.3957684,
        11.52335621, 5.939024563, 2.411355436]

NO3 = [0.397514206, 0.03489186, 0.658161796, -0.005314734, 0.113579797, 0.176644466,
       0.624107917, 1.110226066, 0.786792704, 0.37888583, 1.719388335, 2.022221946,
       0.46466739, 0.815511864, 0.21690609, 0.322903576, -0.030804373, 0.560831008,
       0.025410321, 0.312769289, 0.52030115, -0.029283956, 0.019555403, 0.666563444,
       0.158836025, 0.298688831, 2.113324873, 3.606960488, 0.258263169, 2.941786445,
       0.272310811, 1.144084126, 0.818210433, 1.091141124, 2.90193428, 0.591786795,
       3.111324095, 0.671157505, 0.754591566, 0.282610547, 4.020097373, 3.269778536,
       2.49125676, 0.264257102, 0.534527712, 0.036363298, 1.084420873, 0.184158147,
       0.841389922, 0.546885149, 0.630667661]

Lat = [11.575491, 11.589536, 11.59897, 11.613281, 11.601877, 11.60379, 11.591708, 11.583148,
       11.59061, 11.602786, 11.609179, 11.609071, 11.608669, 11.622434, 11.624364, 11.5906,
       11.60102, 11.596803, 11.60401, 11.617858, 11.621319, 11.651851, 11.653142, 11.645174,
       11.635551, 11.635925, 11.634218, 11.633879, 11.615901, 11.614584, 11.611412, 11.63357,
       11.63818, 11.64629, 11.65018, 11.6443, 11.63893, 11.63481, 11.62801, 11.62148, 11.574538,
       11.576926, 11.574352, 11.572148, 11.564623, 11.559806, 11.558274, 11.558422, 11.562255,
       11.604024, 11.61082]

Long = [122.492891, 122.491819, 122.49434, 122.492076, 122.499438, 122.51008, 122.487343, 122.4787,
        122.471365, 122.471807, 122.450569, 122.443848, 122.440228, 122.435937, 122.426047, 122.43986,
        122.42996, 122.423967, 122.41643, 122.427856, 122.413589, 122.393809, 122.408788, 122.408486,
        122.396703, 122.408292, 122.412004, 122.408047, 122.407781, 122.38256, 122.394101, 122.41728,
        122.41652, 122.41584, 122.42347, 122.43062, 122.43263, 122.43606, 122.44161, 122.44855,
        122.446321, 122.451914, 122.456838, 122.465624, 122.488586, 122.451098, 122.466785,
        122.479122, 122.477287, 122.459473, 122.463607]


# =============================================================================
# Plot style
# =============================================================================

def set_style():
    plt.rcParams.update({
        "font.size": 11.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.titlesize": 15,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# =============================================================================
# Helper functions
# =============================================================================

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def ddmm_to_dd(x):
    x = pd.to_numeric(x, errors="coerce")
    deg = np.floor(x / 100.0)
    minutes = x - deg * 100.0
    return deg + minutes / 60.0

def clean_latlon_if_needed(df, lat_col, lon_col):
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    if (lat.abs().max() > 90) or (lon.abs().max() > 180):
        df[lat_col] = ddmm_to_dd(lat)
        df[lon_col] = ddmm_to_dd(lon)
    return df

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def local_xy(latlon, lat0=None):
    lat = latlon[:, 0]
    lon = latlon[:, 1]
    if lat0 is None:
        lat0 = np.nanmean(lat)
    x = lon * np.cos(np.deg2rad(lat0))
    y = lat
    return np.column_stack([y, x])

def profile_features(df, group_col, depth_col, value_col, prefix):
    out = []
    dfa = df.copy()
    dfa[group_col] = dfa[group_col].astype(str).str.strip()
    dfa[depth_col] = pd.to_numeric(dfa[depth_col], errors="coerce")
    dfa[value_col] = pd.to_numeric(dfa[value_col], errors="coerce")

    for sid, g in dfa.groupby(group_col):
        g2 = g.dropna(subset=[depth_col, value_col]).sort_values(depth_col)
        v = g2[value_col].to_numpy()
        n = len(v)
        if n == 0:
            continue

        if n < 3:
            mid3 = float(np.nanmean(v))
            surf = float(np.nanmean(v))
            bot  = float(np.nanmean(v))
        else:
            a = n // 3
            b = (2 * n) // 3
            surf = float(np.nanmean(v[:a])) if a > 0 else float(v[0])
            mid3 = float(np.nanmean(v[a:b])) if b > a else float(np.nanmean(v))
            bot  = float(np.nanmean(v[b:])) if b < n else float(v[-1])

        grad = bot - surf
        sd = float(np.nanstd(v, ddof=1)) if n >= 2 else 0.0

        out.append({
            group_col: sid,
            f"{prefix}_mid3": mid3,
            f"{prefix}_surf": surf,
            f"{prefix}_bot": bot,
            f"{prefix}_grad": grad,
            f"{prefix}_sd": sd,
            f"{prefix}_n": int(n),
        })
    return pd.DataFrame(out)

def idw_knn_smooth(values, coords_latlon, k=7, power=2.0):
    coords = np.asarray(coords_latlon, float)
    tree = cKDTree(local_xy(coords))
    dist, idx = tree.query(local_xy(coords), k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    dist = np.maximum(dist, 1e-12)
    w = 1.0 / (dist ** power)
    w = w / w.sum(axis=1, keepdims=True)

    sm = np.zeros_like(values, dtype=float)
    for i in range(values.shape[0]):
        sm[i, :] = np.sum(values[idx[i, :], :] * w[i, :, None], axis=0)
    return sm

def pcoa(D):
    D2 = D**2
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J @ D2 @ J
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    pos = evals > 1e-12
    evals = evals[pos]
    evecs = evecs[:, pos]
    Z = evecs * np.sqrt(evals)
    return Z, evals

def adj_r2(r2, n, p):
    if (not np.isfinite(r2)) or (n <= p + 1):
        return np.nan
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

def _permute_within_blocks(rng, arr, blocks):
    """Permute rows of arr within each block label."""
    out = arr.copy()
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        if len(idx) < 2:
            continue
        out[idx, :] = arr[rng.permutation(idx), :]
    return out

def dbRDA_like(DY, XZ, n_axes_keep=8, n_perm=999, seed=42, blocks=None):
    """
    Distance-based RDA-like model using PCoA(Y) -> linear fit from XZ.
    Returns R2, AdjR2, naive permutation p, and optional block-permutation p (within blocks).
    """
    rng = np.random.default_rng(seed)
    Z, _ = pcoa(DY)
    m = min(n_axes_keep, Z.shape[1])
    Zm = Z[:, :m]

    X = np.column_stack([np.ones((XZ.shape[0], 1)), XZ])
    B = np.linalg.lstsq(X, Zm, rcond=None)[0]
    Zhat = X @ B

    sse = np.sum((Zm - Zhat)**2)
    sst = np.sum((Zm - Zm.mean(axis=0))**2)
    r2 = 1.0 - sse/sst if sst > 0 else np.nan
    ar2 = adj_r2(r2, n=Zm.shape[0], p=XZ.shape[1])

    p_naive = np.nan
    p_block = np.nan

    if n_perm and n_perm > 0 and np.isfinite(r2):
        r2_perm = []
        for _ in range(n_perm):
            perm = rng.permutation(XZ.shape[0])
            Xp = np.column_stack([np.ones((XZ.shape[0], 1)), XZ[perm, :]])
            Bp = np.linalg.lstsq(Xp, Zm, rcond=None)[0]
            Zp = Xp @ Bp
            sse_p = np.sum((Zm - Zp)**2)
            r2_p = 1.0 - sse_p/sst if sst > 0 else np.nan
            r2_perm.append(r2_p)
        r2_perm = np.asarray(r2_perm)
        p_naive = (np.sum(r2_perm >= r2) + 1) / (len(r2_perm) + 1)

        # Optional block permutation
        if blocks is not None:
            blocks = np.asarray(blocks)
            # Run only if at least two blocks have size >= 2
            ok_blocks = [b for b in np.unique(blocks) if np.sum(blocks == b) >= 2]
            if len(ok_blocks) >= 2:
                r2_perm_b = []
                for _ in range(n_perm):
                    XZp = _permute_within_blocks(rng, XZ, blocks)
                    Xp = np.column_stack([np.ones((XZ.shape[0], 1)), XZp])
                    Bp = np.linalg.lstsq(Xp, Zm, rcond=None)[0]
                    Zp = Xp @ Bp
                    sse_p = np.sum((Zm - Zp)**2)
                    r2_p = 1.0 - sse_p/sst if sst > 0 else np.nan
                    r2_perm_b.append(r2_p)
                r2_perm_b = np.asarray(r2_perm_b)
                p_block = (np.sum(r2_perm_b >= r2) + 1) / (len(r2_perm_b) + 1)

    return {
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "AdjR2": float(ar2) if np.isfinite(ar2) else np.nan,
        "p_naive": float(p_naive) if np.isfinite(p_naive) else np.nan,
        "p_block": float(p_block) if np.isfinite(p_block) else np.nan,
        "Z_fit2": Zhat[:, :2] if Zhat.shape[1] >= 2 else None,
        "Z_fit": Zhat,
        "Zm": Zm
    }

def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, int)
    ranks[order] = np.arange(1, n+1)
    q = p * n / ranks
    q[order[::-1]] = np.minimum.accumulate(q[order[::-1]])
    return np.clip(q, 0, 1)

def partial_spearman_matrix(df, x_cols, y_cols, control_col):
    c = pd.to_numeric(df[control_col], errors="coerce").to_numpy(float)
    C = np.column_stack([np.ones_like(c), c])

    out_rho = np.full((len(x_cols), len(y_cols)), np.nan)
    out_p   = np.full((len(x_cols), len(y_cols)), np.nan)

    for i, xc in enumerate(x_cols):
        x = pd.to_numeric(df[xc], errors="coerce").to_numpy(float)
        okx = np.isfinite(x) & np.isfinite(c)
        if okx.sum() < 6:
            continue
        bx = np.linalg.lstsq(C[okx, :], x[okx], rcond=None)[0]
        rx = x.copy()
        rx[okx] = x[okx] - (C[okx, :] @ bx)

        for j, yc in enumerate(y_cols):
            y = pd.to_numeric(df[yc], errors="coerce").to_numpy(float)
            ok = okx & np.isfinite(y)
            if ok.sum() < 6:
                continue
            by = np.linalg.lstsq(C[ok, :], y[ok], rcond=None)[0]
            ry = y.copy()
            ry[ok] = y[ok] - (C[ok, :] @ by)

            rho, pval = spearmanr(rx[ok], ry[ok])
            out_rho[i, j] = rho
            out_p[i, j] = pval

    flat_p = out_p.flatten()
    mask = np.isfinite(flat_p)
    q = np.full_like(flat_p, np.nan)
    if mask.sum() > 0:
        q[mask] = benjamini_hochberg(flat_p[mask])
    out_q = q.reshape(out_p.shape)
    return out_rho, out_p, out_q

def space_basis(coords, n_knots=4, max_pc=4, seed=42):
    spl = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
    S = spl.fit_transform(coords)
    S = StandardScaler().fit_transform(S)
    max_pc = int(max(1, max_pc))
    pca = PCA(n_components=min(max_pc, S.shape[1]), random_state=seed)
    Sp = pca.fit_transform(S)
    Sp = StandardScaler().fit_transform(Sp)
    return Sp

def regime_colors(n_regimes):
    cmap = plt.get_cmap("viridis", n_regimes)
    return [cmap(i) for i in range(n_regimes)]

def plot_convex_hull(ax, x, y, color, alpha=0.10, lw=1.2):
    pts = np.column_stack([x, y])
    if pts.shape[0] < 3:
        return
    try:
        hull = ConvexHull(pts)
        poly = pts[hull.vertices]
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, zorder=0)
        ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
                color=color, lw=lw, alpha=0.8, zorder=1)
    except Exception:
        return

def heatmap_with_text(ax, mat, vmin=-0.6, vmax=0.6, fmt="{:+.2f}"):
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        fontsize=9.5, color="white" if abs(mat[i, j]) > 0.25 else "black")
    return im

def domain_mask_from_basemap(tif_path, lons, lats, buffer=0.0):
    """
    Returns boolean mask for points that fall within the raster bounds (plus optional buffer).
    """
    lons = np.asarray(lons, float)
    lats = np.asarray(lats, float)
    mask = np.isfinite(lons) & np.isfinite(lats)
    if not os.path.exists(tif_path):
        return mask

    with rasterio.open(tif_path) as src:
        b = src.bounds
        crs = src.crs

        if (crs is None) or ("epsg:4326" in str(crs).lower()):
            x = lons
            y = lats
            # Buffer in degrees if the raster CRS is geographic (lat/lon)
            buf = float(buffer)
            inb = (x >= (b.left - buf)) & (x <= (b.right + buf)) & (y >= (b.bottom - buf)) & (y <= (b.top + buf))
            return mask & inb

        # Projected CRS: transform lon/lat to raster CRS, then bounds-check in CRS units
        x, y = rio_transform("EPSG:4326", crs, lons.tolist(), lats.tolist())
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        buf = float(buffer)
        inb = (x >= (b.left - buf)) & (x <= (b.right + buf)) & (y >= (b.bottom - buf)) & (y <= (b.top + buf))
        return mask & inb

def plot_basemap(ax, tif_path, alpha=0.95):
    if (tif_path is None) or (not os.path.exists(tif_path)):
        return None, None

    src = rasterio.open(tif_path)
    rioshow(src, ax=ax, alpha=alpha)

    if src.crs is None or "epsg:4326" in src.crs.to_string().lower():
        def tf(lon, lat):
            return np.asarray(lon), np.asarray(lat)
        return src, tf

    def tf(lon, lat):
        x, y = rio_transform("EPSG:4326", src.crs,
                             np.asarray(lon).tolist(),
                             np.asarray(lat).tolist())
        return np.asarray(x), np.asarray(y)

    return src, tf

def add_north_arrow(ax, xy=(0.94, 0.12), size=0.10):
    x, y = xy
    ax.annotate("", xy=(x, y+size), xytext=(x, y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="white"))
    ax.text(x, y+size+0.02, "N", transform=ax.transAxes,
            ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")

def add_scalebar(ax, length_m=1000, location=(0.06, 0.08), linewidth=4):
    x0, y0 = location
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xr = xmax - xmin
    yr = ymax - ymin
    xb0 = xmin + x0 * xr
    yb0 = ymin + y0 * yr
    xb1 = xb0 + length_m

    ax.plot([xb0, xb1], [yb0, yb0], lw=linewidth, color="white", solid_capstyle="butt")
    ax.plot([xb0, xb1], [yb0, yb0], lw=linewidth/2, color="black", solid_capstyle="butt")
    ax.text((xb0+xb1)/2, yb0 + 0.02*yr, f"{int(length_m/1000)} km",
            ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")

def variation_partitioning_3sets(DY, X1, X2, X3, seed=42, blocks=None):
    """
    Uses AdjR2 from dbRDA-like fits.
    Returns raw uniques and shared; plus plot-friendly (clipped) uniques.
    """
    def ar2(XZ):
        out = dbRDA_like(DY, XZ, n_axes_keep=8, n_perm=0, seed=seed, blocks=blocks)
        return out["AdjR2"]

    R_full = ar2(np.column_stack([X1, X2, X3]))
    R_23   = ar2(np.column_stack([X2, X3]))
    R_13   = ar2(np.column_stack([X1, X3]))
    R_12   = ar2(np.column_stack([X1, X2]))

    a = R_full - R_23 if np.isfinite(R_full) and np.isfinite(R_23) else np.nan
    b = R_full - R_13 if np.isfinite(R_full) and np.isfinite(R_13) else np.nan
    c = R_full - R_12 if np.isfinite(R_full) and np.isfinite(R_12) else np.nan
    shared = R_full - (a + b + c) if all(np.isfinite(v) for v in [R_full, a, b, c]) else np.nan

    def nz(x):
        return 0.0 if (x is None or (not np.isfinite(x))) else float(x)

    return {
        "AdjR2_full": R_full,
        "uniq_X1": a, "uniq_X2": b, "uniq_X3": c,
        "shared_raw": shared,
        "plot_uniq_X1": max(0.0, nz(a)),
        "plot_uniq_X2": max(0.0, nz(b)),
        "plot_uniq_X3": max(0.0, nz(c)),
    }


# =============================================================================
# Load and prepare data
# =============================================================================

def load_aaq():
    temp = pd.read_csv(TEMP_CSV)
    sal  = pd.read_csv(SAL_CSV)
    chla = pd.read_csv(CHLA_CSV)
    turb = pd.read_csv(TURB_CSV)

    group_col = find_col(temp, ["Comment", "comment"])
    if group_col is None:
        raise ValueError("AAQ files must include a 'Comment' station ID column.")

    depth_col = None
    for df_ in [temp, sal, chla, turb]:
        dc = find_col(df_, ["Depth [m]", "Depth", "depth", "DEPTH"])
        if dc is not None:
            depth_col = dc
            break
    if depth_col is None:
        raise ValueError("Could not find Depth column in AAQ files.")

    temp_col = find_col(temp, ["Temp. [degC]", "Temp", "Temperature", "temp"])
    sal_col  = find_col(sal,  ["Sal.", "Sal", "Salinity", "sal"])
    chla_col = find_col(chla, ["Chl-a [µg/l]", "Chl-a", "Chla", "Chlorophyll", "chl"])
    turb_col = find_col(turb, ["Turb. [FTU]", "Turb", "Turbidity", "turb"])

    lat_col = find_col(temp, ["Latitude", "Lat", "lat"])
    lon_col = find_col(temp, ["Longitude", "Lon", "Long", "lon", "lng"])

    if any(c is None for c in [temp_col, sal_col, chla_col, turb_col, lat_col, lon_col]):
        raise ValueError("Missing required columns in AAQ files (check headers).")

    temp = clean_latlon_if_needed(temp, lat_col, lon_col)

    temp[group_col] = temp[group_col].astype(str).str.strip()
    aaq_ll = temp.groupby(group_col)[[lat_col, lon_col]].mean().reset_index()
    aaq_ll = clean_latlon_if_needed(aaq_ll, lat_col, lon_col)

    temp_feat = profile_features(temp, group_col, depth_col, temp_col, prefix="AAQ_Temp")
    sal_feat  = profile_features(sal,  group_col, depth_col, sal_col,  prefix="AAQ_Sal")
    chla_feat = profile_features(chla, group_col, depth_col, chla_col, prefix="AAQ_Chla")
    turb_feat = profile_features(turb, group_col, depth_col, turb_col, prefix="AAQ_Turb")

    aaq = (temp_feat.merge(sal_feat, on=group_col, how="inner")
                 .merge(chla_feat, on=group_col, how="inner")
                 .merge(turb_feat, on=group_col, how="inner")
                 .merge(aaq_ll, on=group_col, how="left"))

    aaq["AAQ_Chla_log1p_mid3"] = np.log1p(np.maximum(0, aaq["AAQ_Chla_mid3"]))
    aaq["AAQ_Turb_log1p_mid3"] = np.log1p(np.maximum(0, aaq["AAQ_Turb_mid3"]))

    aaq["dist_inlet_km"] = haversine_km(
        pd.to_numeric(aaq[lat_col], errors="coerce").to_numpy(float),
        pd.to_numeric(aaq[lon_col], errors="coerce").to_numpy(float),
        INLET_LATLON[0], INLET_LATLON[1]
    )
    return aaq, group_col, lat_col, lon_col

def build_nutrients():
    df = pd.DataFrame({
        "nut_lat": pd.Series(Lat),
        "nut_lon": pd.Series(Long),
        "NO2": pd.Series(NO2),
        "NH4": pd.Series(NH4),
        "NO2_NO3": pd.Series(NO2_NO3),
        "PO4": pd.Series(PO4),
        "SiO2": pd.Series(SiO2),
        "NO3": pd.Series(NO3),
    }).dropna().reset_index(drop=True)

    df["nut_dist_inlet_km"] = haversine_km(
        df["nut_lat"].to_numpy(float),
        df["nut_lon"].to_numpy(float),
        INLET_LATLON[0], INLET_LATLON[1]
    )
    return df

def match_nutrients_to_aaq(nutr, aaq, group_col, lat_col, lon_col, max_match_km=None, enforce_unique=True):
    # Select nutrient points
    if USE_N_CLOSEST:
        selected = nutr.nsmallest(N_CLOSEST, "nut_dist_inlet_km").copy()
    else:
        selected = nutr[nutr["nut_dist_inlet_km"] <= MAX_DIST_KM].copy()

    # Build KDTree for AAQ station lookups
    aaq_valid = aaq.dropna(subset=[lat_col, lon_col]).copy()
    aaq_latlon = aaq_valid[[lat_col, lon_col]].to_numpy(float)
    tree = cKDTree(local_xy(aaq_latlon))

    nut_latlon = selected[["nut_lat", "nut_lon"]].to_numpy(float)
    _, idx = tree.query(local_xy(nut_latlon, lat0=np.nanmean(aaq_latlon[:, 0])), k=1)

    selected["Nearest_AAQ_ID"] = aaq_valid.iloc[idx][group_col].astype(str).to_numpy()
    selected["match_km"] = haversine_km(
        selected["nut_lat"].to_numpy(float),
        selected["nut_lon"].to_numpy(float),
        aaq_valid.iloc[idx][lat_col].to_numpy(float),
        aaq_valid.iloc[idx][lon_col].to_numpy(float),
    )

    if max_match_km is not None:
        selected = selected[selected["match_km"] <= max_match_km].copy()

    # Keep one nutrient point per AAQ station (closest match only)
    if enforce_unique:
        selected = (selected.sort_values("match_km")
                            .drop_duplicates(subset=["Nearest_AAQ_ID"], keep="first")
                            .reset_index(drop=True))

    merged = selected.merge(aaq, left_on="Nearest_AAQ_ID", right_on=group_col, how="inner")
    return merged


# =============================================================================
# Main execution
# =============================================================================

def main():
    ensure_dirs()
    set_style()

    # Load raw data
    aaq_all, group_col, lat_col, lon_col = load_aaq()
    nutr = build_nutrients()

    # Ensure numeric latitude/longitude values
    aaq_all[lat_col] = pd.to_numeric(aaq_all[lat_col], errors="coerce")
    aaq_all[lon_col] = pd.to_numeric(aaq_all[lon_col], errors="coerce")
    nutr["nut_lat"] = pd.to_numeric(nutr["nut_lat"], errors="coerce")
    nutr["nut_lon"] = pd.to_numeric(nutr["nut_lon"], errors="coerce")

    # Filter to basemap extent (removes out-of-domain points)
    if FILTER_TO_BASEMAP_EXTENT and os.path.exists(BASEMAP_TIF):
        maskA = domain_mask_from_basemap(BASEMAP_TIF, aaq_all[lon_col].to_numpy(float), aaq_all[lat_col].to_numpy(float), buffer=BASEMAP_BUFFER)
        removed = aaq_all.loc[~maskA, [group_col, lat_col, lon_col, "dist_inlet_km"]].copy()
        if len(removed) > 0:
            removed.to_csv(os.path.join(TAB_DIR, "aaq_out_of_domain_stations.csv"), index=False)
        aaq_all = aaq_all.loc[maskA].copy()

        maskN = domain_mask_from_basemap(BASEMAP_TIF, nutr["nut_lon"].to_numpy(float), nutr["nut_lat"].to_numpy(float), buffer=BASEMAP_BUFFER)
        removedN = nutr.loc[~maskN, :].copy()
        if len(removedN) > 0:
            removedN.to_csv(os.path.join(TAB_DIR, "nutr_out_of_domain_points.csv"), index=False)
        nutr = nutr.loc[maskN].copy()

    # Define regimes using all in-domain AAQ stations
    AAQ_STATE_COLS = [
        "AAQ_Temp_mid3", "AAQ_Sal_mid3",
        "AAQ_Chla_log1p_mid3", "AAQ_Turb_log1p_mid3",
        "AAQ_Temp_grad", "AAQ_Sal_grad",
        "AAQ_Temp_sd", "AAQ_Sal_sd",
    ]
    NUTR_COLS = ["NO2", "NH4", "NO2_NO3", "PO4", "SiO2", "NO3"]

    aaq_use = aaq_all.dropna(subset=AAQ_STATE_COLS + [lat_col, lon_col]).copy()
    aaq_use["lat"] = aaq_use[lat_col].to_numpy(float)
    aaq_use["lon"] = aaq_use[lon_col].to_numpy(float)
    aaq_use = aaq_use.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Export AAQ features for all in-domain stations
    export_cols = [group_col, "lat", "lon", "dist_inlet_km"] + AAQ_STATE_COLS
    aaq_use[export_cols].to_csv(os.path.join(TAB_DIR, "aaq_station_features_all.csv"), index=False)

    Y_all = aaq_use[AAQ_STATE_COLS].to_numpy(float)
    coords_all = aaq_use[["lat", "lon"]].to_numpy(float)

    kbest = int(K_BEST)
    Ysm_all = idw_knn_smooth(Y_all, coords_all, k=kbest, power=2.0)
    YZ_all = StandardScaler().fit_transform(Ysm_all)

    pca_all = PCA(n_components=2, random_state=RANDOM_SEED)
    scores_all = pca_all.fit_transform(YZ_all)

    km = KMeans(n_clusters=N_REGIMES, random_state=RANDOM_SEED, n_init=50)
    aaq_use["regime"] = km.fit_predict(scores_all)

    # Attach regime labels back to the AAQ table
    aaq_all = aaq_all.merge(aaq_use[[group_col, "regime"]], on=group_col, how="inner")

    # Build matched subset for nutrient inference (unique AAQ stations)
    df_all = match_nutrients_to_aaq(
        nutr, aaq_all, group_col, lat_col, lon_col,
        max_match_km=None,
        enforce_unique=ENFORCE_UNIQUE_AAQ_STATIONS
    )

    needed = AAQ_STATE_COLS + NUTR_COLS + ["dist_inlet_km", lat_col, lon_col, "match_km", "regime"]
    d0 = df_all.dropna(subset=needed).copy()
    d0["lat"] = pd.to_numeric(d0[lat_col], errors="coerce")
    d0["lon"] = pd.to_numeric(d0[lon_col], errors="coerce")
    d0 = d0.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    matched_ids = set(d0["Nearest_AAQ_ID"].astype(str).tolist())

    # Save matched table
    d0.to_csv(os.path.join(TAB_DIR, "aaq_nutrients_matched_primary_uniqueAAQ.csv"), index=False)

    print(f"AAQ usable stations (in-domain): n={len(aaq_use)}")
    print(f"Matched rows (unique AAQ stations): n={len(d0)}")
    if len(d0) > 0:
        print("Match distance km summary:",
              float(d0["match_km"].min()), float(d0["match_km"].median()), float(d0["match_km"].max()))

    # Constrained structure on the matched subset
    Y = d0[AAQ_STATE_COLS].to_numpy(float)
    coords = d0[["lat", "lon"]].to_numpy(float)

    # Block labels used for block permutation (matched subset)
    blocks = d0["regime"].to_numpy(int) if (DO_BLOCK_PERM and len(d0) > 0) else None

    db_out = {}
    for k in K_LIST:
        Ysm = idw_knn_smooth(Y, coords, k=k, power=2.0)
        YZ = StandardScaler().fit_transform(Ysm)
        DY = squareform(pdist(YZ, metric="euclidean"))

        X = d0[["dist_inlet_km"] + NUTR_COLS].to_numpy(float)
        XZ = StandardScaler().fit_transform(X)

        out = dbRDA_like(
            DY, XZ,
            n_axes_keep=8,
            n_perm=N_PERM_NAIVE,
            seed=RANDOM_SEED,
            blocks=(blocks if DO_BLOCK_PERM else None)
        )
        db_out[k] = out

    out_best = db_out[kbest]
    Zfit2 = out_best["Z_fit2"]

    # Variation partitioning (matched subset)
    n = len(d0)
    max_space_pc = min(4, max(1, n // 3))
    Sp = space_basis(coords, n_knots=4, max_pc=max_space_pc, seed=RANDOM_SEED)

    Ysm_best = idw_knn_smooth(Y, coords, k=kbest, power=2.0)
    YZ_best = StandardScaler().fit_transform(Ysm_best)
    DY_best = squareform(pdist(YZ_best, metric="euclidean"))

    X1 = StandardScaler().fit_transform(d0[["dist_inlet_km"]].to_numpy(float))
    X2 = StandardScaler().fit_transform(d0[NUTR_COLS].to_numpy(float))
    X3 = Sp

    vp = variation_partitioning_3sets(DY_best, X1, X2, X3, seed=RANDOM_SEED, blocks=None)

    pd.DataFrame([{
        "kbest": kbest, "n": n, "space_pc": int(max_space_pc),
        **vp
    }]).to_csv(os.path.join(TAB_DIR, "variation_partitioning_3set.csv"), index=False)

    # Sensitivity to match threshold (unique AAQ stations)
    sens_rows = []
    for thr in MATCH_THRESHOLDS:
        df_thr = match_nutrients_to_aaq(
            nutr, aaq_all, group_col, lat_col, lon_col,
            max_match_km=thr,
            enforce_unique=ENFORCE_UNIQUE_AAQ_STATIONS
        )

        d_thr = df_thr.dropna(subset=AAQ_STATE_COLS + NUTR_COLS + ["dist_inlet_km", lat_col, lon_col]).copy()
        d_thr["lat"] = pd.to_numeric(d_thr[lat_col], errors="coerce")
        d_thr["lon"] = pd.to_numeric(d_thr[lon_col], errors="coerce")
        d_thr = d_thr.dropna(subset=["lat", "lon"]).reset_index(drop=True)

        if len(d_thr) < 9:
            sens_rows.append({"thr_km": thr if thr is not None else 999,
                              "n": len(d_thr), "AdjR2": np.nan, "p_naive": np.nan, "p_block": np.nan})
            continue

        Yt = d_thr[AAQ_STATE_COLS].to_numpy(float)
        ct = d_thr[["lat", "lon"]].to_numpy(float)
        Ysm = idw_knn_smooth(Yt, ct, k=kbest, power=2.0)
        YZt = StandardScaler().fit_transform(Ysm)
        DYt = squareform(pdist(YZt, metric="euclidean"))

        Xt = d_thr[["dist_inlet_km"] + NUTR_COLS].to_numpy(float)
        XZt = StandardScaler().fit_transform(Xt)

        blk_thr = d_thr["regime"].to_numpy(int) if (DO_BLOCK_PERM and "regime" in d_thr.columns) else None
        out = dbRDA_like(
            DYt, XZt,
            n_axes_keep=8,
            n_perm=N_PERM_SENS,
            seed=RANDOM_SEED,
            blocks=(blk_thr if DO_BLOCK_PERM else None)
        )
        sens_rows.append({"thr_km": thr if thr is not None else 999, "n": len(d_thr),
                          "AdjR2": out["AdjR2"], "p_naive": out["p_naive"], "p_block": out["p_block"]})

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(os.path.join(TAB_DIR, "match_distance_sensitivity_uniqueAAQ.csv"), index=False)

    # Partial Spearman analysis (matched subset)
    rho, pval, qval = partial_spearman_matrix(d0, NUTR_COLS, AAQ_STATE_COLS, "dist_inlet_km")
    pd.DataFrame(rho, index=NUTR_COLS, columns=AAQ_STATE_COLS).to_csv(os.path.join(TAB_DIR, "partial_spearman_rho.csv"))
    pd.DataFrame(qval, index=NUTR_COLS, columns=AAQ_STATE_COLS).to_csv(os.path.join(TAB_DIR, "partial_spearman_qFDR.csv"))

    # Top six predictor arrows
    top_arrows = []
    Xpred = d0[["dist_inlet_km"] + NUTR_COLS].to_numpy(float)
    XpredZ = StandardScaler().fit_transform(Xpred)
    if Zfit2 is not None:
        arrows = []
        for j, name in enumerate(["dist_inlet_km"] + NUTR_COLS):
            r1 = np.corrcoef(XpredZ[:, j], Zfit2[:, 0])[0, 1]
            r2 = np.corrcoef(XpredZ[:, j], Zfit2[:, 1])[0, 1]
            mag = np.sqrt(r1*r1 + r2*r2)
            arrows.append((name, r1, r2, mag))
        arrows = sorted(arrows, key=lambda x: x[3], reverse=True)
        top_arrows = arrows[:6]

    # Regime fingerprint (matched subset)
    fingerprint_cols = AAQ_STATE_COLS + ["dist_inlet_km"] + NUTR_COLS
    F = d0[fingerprint_cols].copy()
    Fz = pd.DataFrame(StandardScaler().fit_transform(F.to_numpy(float)), columns=fingerprint_cols)
    Fz["regime"] = d0["regime"].astype(int).to_numpy()

    reg_levels = sorted(Fz["regime"].unique().tolist())
    reg_med = Fz.groupby("regime")[fingerprint_cols].median().loc[reg_levels, :]
    reg_med.to_csv(os.path.join(TAB_DIR, "regime_fingerprint_standardized_medians.csv"))

    # =============================================================================
    # Figure 1 (main): 4 panels
    # =============================================================================
    cols = regime_colors(N_REGIMES)

    fig1 = plt.figure(figsize=(20, 11))
    gs1 = fig1.add_gridspec(2, 2, wspace=0.25, hspace=0.30)

    # A) Regime map (all AAQ stations)
    axA = fig1.add_subplot(gs1[0, 0])
    src, tf = (None, None)
    try:
        src, tf = plot_basemap(axA, BASEMAP_TIF, alpha=0.95)
    except Exception as e:
        print("Basemap plot failed:", e)

    lon_all = aaq_use["lon"].to_numpy(float)
    lat_all = aaq_use["lat"].to_numpy(float)
    reg_all = aaq_use["regime"].to_numpy(int)
    mask_match = aaq_use[group_col].astype(str).isin(matched_ids).to_numpy(bool)

    if tf is not None:
        x_all, y_all = tf(lon_all, lat_all)

        for r in range(N_REGIMES):
            m = reg_all == r
            axA.scatter(x_all[m], y_all[m], s=95, alpha=0.98,
                        edgecolor="black", linewidth=0.35, color=cols[r], zorder=3,
                        label=f"Regime {r}")

        # Rings show matched subset stations
        axA.scatter(x_all[mask_match], y_all[mask_match],
                    facecolors="none", edgecolors="white", s=260, linewidth=2.0, zorder=6)

        # Inlet marker
        xi, yi = tf([INLET_LATLON[1]], [INLET_LATLON[0]])
        axA.scatter(xi, yi, marker="*", s=420, edgecolor="black", linewidth=0.8,
                    color="gold", zorder=7)

        # Nutrient site markers
        xN, yN = tf(nutr["nut_lon"].to_numpy(float), nutr["nut_lat"].to_numpy(float))
        axA.scatter(xN, yN, marker="^", s=70, facecolor="none",
                    edgecolor="white", linewidth=1.1, alpha=0.85, zorder=4)

        # Keep map extent aligned to raster bounds when available
        if src is not None:
            b = src.bounds
            axA.set_xlim(b.left, b.right)
            axA.set_ylim(b.bottom, b.top)
            try:
                add_scalebar(axA, length_m=1000, location=(0.06, 0.08))
            except Exception:
                pass
            add_north_arrow(axA, xy=(0.93, 0.10), size=0.11)

    axA.set_title("A) AAQ hydrographic regimes (all stations)\nRings: nutrient-matched stations; ^ nutrient sites; * inlet")
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_aspect("equal", adjustable="box")
    axA.legend(frameon=False, loc="upper left", ncol=2)

    # Close raster handle to avoid file locks on Windows
    if src is not None:
        try:
            src.close()
        except Exception:
            pass

    # B) PCA regimes (all AAQ stations)
    axB = fig1.add_subplot(gs1[0, 1])
    for r in range(N_REGIMES):
        m = reg_all == r
        axB.scatter(scores_all[m, 0], scores_all[m, 1],
                    s=80, alpha=0.95, edgecolor="k", linewidth=0.25,
                    color=cols[r], label=f"Regime {r}")
        plot_convex_hull(axB, scores_all[m, 0], scores_all[m, 1], color=cols[r], alpha=0.10)

    axB.scatter(scores_all[mask_match, 0], scores_all[mask_match, 1],
                facecolors="none", edgecolors="k", s=200, linewidth=1.6, zorder=5)

    axB.set_title("B) AAQ-state PCA regimes (all stations)\nRings: nutrient-matched subset")
    axB.set_xlabel(f"PC1 ({pca_all.explained_variance_ratio_[0]*100:.1f}%)")
    axB.set_ylabel(f"PC2 ({pca_all.explained_variance_ratio_[1]*100:.1f}%)")
    axB.grid(True, alpha=0.20)

    # C) Mixing axis by regime (distance to inlet)
    axC = fig1.add_subplot(gs1[1, 0])
    dist = aaq_use["dist_inlet_km"].to_numpy(float)

    # Boxplot by regime
    data = [dist[reg_all == r] for r in range(N_REGIMES)]
    bp = axC.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)

    for r, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cols[r])
        patch.set_alpha(0.25)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    # Faint jittered points for all stations
    rng = np.random.default_rng(RANDOM_SEED)
    for r in range(N_REGIMES):
        yy = dist[reg_all == r]
        xx = (r + 1) + rng.normal(0, 0.06, size=len(yy))
        axC.scatter(xx, yy, s=18, alpha=0.25, color=cols[r], edgecolor="none", zorder=2)

    # Open circles for matched subset
    aaq_match = aaq_use.loc[mask_match, [group_col, "regime", "dist_inlet_km"]].copy()
    for r in range(N_REGIMES):
        yy = aaq_match.loc[aaq_match["regime"] == r, "dist_inlet_km"].to_numpy(float)
        xx = (r + 1) + rng.normal(0, 0.05, size=len(yy))
        axC.scatter(xx, yy, s=65, facecolors="none", edgecolors="k", linewidth=1.1, zorder=5)

    axC.set_xticks(np.arange(1, N_REGIMES+1))
    axC.set_xticklabels([f"Regime {r}" for r in range(N_REGIMES)])
    axC.set_ylabel("Distance to inlet (km)")
    axC.set_title("C) Mixing axis: distance-to-inlet differs across regimes\n(open circles = matched subset)")
    axC.grid(True, axis="y", alpha=0.20)

    # D) Variation-partitioning bars (unique contributions) + total dashed line
    axD = fig1.add_subplot(gs1[1, 1])
    uniq = [vp["plot_uniq_X1"], vp["plot_uniq_X2"], vp["plot_uniq_X3"]]
    labels = ["Unique\nDistance", "Unique\nNutrients", "Unique\nSpace"]
    axD.bar(np.arange(3), uniq, alpha=0.95)

    # Total explained fraction line
    if np.isfinite(vp["AdjR2_full"]):
        axD.axhline(vp["AdjR2_full"], linestyle="--", linewidth=1.6)
    axD.set_xticks(np.arange(3)); axD.set_xticklabels(labels)
    axD.set_ylabel("AdjR² fraction")
    axD.set_title(
        f"D) Variation partitioning (matched unique stations)\n"
        f"Total AdjR²={vp['AdjR2_full']:.3f}; shared(raw)={vp['shared_raw']:.3f}\n"
        f"n_uniqueStations={len(d0)}"
    )
    axD.grid(True, axis="y", alpha=0.20)

    fig1_path_png = os.path.join(FIG_DIR, f"AAQ_nutrients_MAIN_{VERSION_TAG}_k{kbest}.png")
    fig1_path_pdf = os.path.join(FIG_DIR, f"AAQ_nutrients_MAIN_{VERSION_TAG}_k{kbest}.pdf")
    fig1.savefig(fig1_path_png, dpi=450, bbox_inches="tight")
    fig1.savefig(fig1_path_pdf, bbox_inches="tight")
    plt.close(fig1)

    # =============================================================================
    # Figure 2 (supplementary): 4 panels in a 2x2 layout
    # =============================================================================
    fig2 = plt.figure(figsize=(20, 12))
    gs2 = fig2.add_gridspec(2, 2, wspace=0.25, hspace=0.30)

    # S1) Constrained ordination (matched subset)
    axS1 = fig2.add_subplot(gs2[0, 0])
    if Zfit2 is not None:
        sc = axS1.scatter(Zfit2[:, 0], Zfit2[:, 1],
                          c=d0["dist_inlet_km"], s=120, alpha=0.95,
                          edgecolor="k", linewidth=0.25)
        cb = fig2.colorbar(sc, ax=axS1, fraction=0.046, pad=0.03)
        cb.set_label("Distance to inlet (km)")

        # Scale arrows to fit panel range
        xr = np.ptp(Zfit2[:, 0]) if np.ptp(Zfit2[:, 0]) > 0 else 1.0
        yr = np.ptp(Zfit2[:, 1]) if np.ptp(Zfit2[:, 1]) > 0 else 1.0
        scale = 0.40 * min(xr, yr)

        for name, r1, r2, _mag in top_arrows:
            axS1.arrow(0, 0, r1*scale, r2*scale,
                       head_width=0.06*scale, head_length=0.08*scale,
                       length_includes_head=True, color="black", alpha=0.85)
            axS1.text(r1*scale*1.10, r2*scale*1.10, name, fontsize=10.5)

    axS1.set_xlabel("Constrained axis 1")
    axS1.set_ylabel("Constrained axis 2")
    axS1.grid(True, alpha=0.20)
    axS1.set_title(
        f"S1) Constrained ordination (matched unique stations)\n"
        f"AdjR²={out_best['AdjR2']:.3f}; p_naive={out_best['p_naive']:.4f}; p_block={out_best['p_block']:.4f}"
    )

    # S2) Sensitivity to match threshold
    axS2 = fig2.add_subplot(gs2[0, 1])
    xlab = ["All" if v == 999 else str(v) for v in sens_df["thr_km"].tolist()]
    xs = np.arange(len(sens_df))
    axS2.plot(xs, sens_df["AdjR2"], marker="o", linewidth=2.2)

    axS2.set_xticks(xs)
    axS2.set_xticklabels(xlab)
    axS2.set_xlabel("Max match distance (km)")
    axS2.set_ylabel("AdjR² (distance + nutrients → AAQ-state)")
    axS2.grid(True, alpha=0.20)
    axS2.set_title("S2) Sensitivity to nutrient→AAQ match distance\n")

    for i, row in sens_df.iterrows():
        if np.isfinite(row["AdjR2"]):
            ptxt = f"n={int(row['n'])}\n"
            if np.isfinite(row.get("p_block", np.nan)):
                ptxt += f"blk p={row['p_block']:.3f}"
            else:
                ptxt += f"p={row['p_naive']:.3f}"
            axS2.text(i, row["AdjR2"] + 0.015, ptxt, ha="center", fontsize=10.2)
        else:
            axS2.text(i, 0.02, f"n={int(row['n'])}", ha="center", fontsize=10.2)

    # S3) Partial Spearman heatmap
    axS3 = fig2.add_subplot(gs2[1, 0])
    im = heatmap_with_text(axS3, rho, vmin=-0.6, vmax=0.6, fmt="{:+.2f}")
    axS3.set_yticks(np.arange(len(NUTR_COLS))); axS3.set_yticklabels(NUTR_COLS)
    axS3.set_xticks(np.arange(len(AAQ_STATE_COLS))); axS3.set_xticklabels(AAQ_STATE_COLS, rotation=45, ha="right")
    axS3.set_title("S3) Partial Spearman (nutrients ↔ AAQ | distance)\nCells show ρ; * qFDR<0.10; ** qFDR<0.05")
    fig2.colorbar(im, ax=axS3, fraction=0.046, pad=0.03)

    for i in range(len(NUTR_COLS)):
        for j in range(len(AAQ_STATE_COLS)):
            q = qval[i, j]
            if np.isfinite(q):
                if q < 0.05:
                    axS3.text(j, i, "**", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
                elif q < 0.10:
                    axS3.text(j, i, "*", ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    # S4) Regime fingerprint (standardized medians)
    axS4 = fig2.add_subplot(gs2[1, 1])
    hm = axS4.imshow(reg_med.to_numpy(float), aspect="auto", vmin=-1.5, vmax=1.5)
    axS4.set_yticks(np.arange(len(reg_levels)))
    axS4.set_yticklabels([f"Regime {r}" for r in reg_levels])
    axS4.set_xticks(np.arange(len(reg_med.columns)))
    axS4.set_xticklabels(reg_med.columns, rotation=60, ha="right", fontsize=9.5)
    axS4.set_title("S4) Regime fingerprint (standardized medians; matched subset)")
    fig2.colorbar(hm, ax=axS4, fraction=0.046, pad=0.03)

    fig2_path_png = os.path.join(FIG_DIR, f"AAQ_nutrients_SUPP_{VERSION_TAG}_k{kbest}.png")
    fig2_path_pdf = os.path.join(FIG_DIR, f"AAQ_nutrients_SUPP_{VERSION_TAG}_k{kbest}.pdf")
    fig2.savefig(fig2_path_png, dpi=450, bbox_inches="tight")
    fig2.savefig(fig2_path_pdf, bbox_inches="tight")
    plt.close(fig2)

    print("Saved MAIN figure:", fig1_path_png)
    print("Saved MAIN figure:", fig1_path_pdf)
    print("Saved SUPP figure:", fig2_path_png)
    print("Saved SUPP figure:", fig2_path_pdf)
    print("Exported ALL-stations AAQ features:", os.path.join(TAB_DIR, "aaq_station_features_all.csv"))
    print("Saved tables to:", TAB_DIR)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
