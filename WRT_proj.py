#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:35:59 2025

@author: savannahsouthward
@author: sarahzeko
"""

###############################################################################
# NEXRAD Level-II → VAD + EVAD
# - reads in Level II files with Py-ART
# - computes VAD (binwise harmonic fit)
# - computes EVAD (global vertical LSQ w// smoothing, deformation, divergence)
###############################################################################

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pyart

############################### user-config ####################################
VEL_FIELD        = "velocity"      # radial velocity field name in L2
REF_FIELD        = "reflectivity"  # optional (not heavily used here)
Z_BIN_WIDTH      = 250.0           # [m] vertical bin width (VAD/EVAD)
Z_MAX            = 12000.0         # [m AGL] max height for profile
MAX_ELEV_DEG     = 30.0            # ignore very steep tilts (> this elevation)
MIN_GATES        = 60              # minimum gates per z-bin for a solution
MIN_AZ_SECTORS   = 6               # minimum 30° az sectors for coverage
USE_RANGE_WEIGHT = False           # weight gates by 1/r (closer - more weight)
DEALIAS_FIRST    = False           # attempts to dealias using pyart
INCLUDE_HARMONICS = True           # EVAD: include 2θ harmonic (deformation)
INCLUDE_R_TERM     = True          # EVAD: include r-term (divergence proxy)
LAMBDA_SMOOTH      = 0.001         # EVAD vertical smoothing strength: Tikhonov
###############################################################################

# functions for main()

def masked_to_nan(arr):
    """convert masked arrays to plain float arrays with NaNs."""
    if np.ma.isMaskedArray(arr):
        return arr.filled(np.nan).astype(float)
    return np.array(arr, dtype=float)


def gate_height_43earth(range_m, elev_rad, Re=6_371_000.0, k=4.0/3.0):
    """
    4/3-Earth beam height model (Doviak & Zrnić):
      h = sqrt(r^2 + (kRe)^2 + 2 r kRe sin(e)) - kRe
    returns height above radar (m) for given range and elevation.
    """
    kRe = k * Re
    r = range_m
    return np.sqrt(r*r + kRe*kRe + 2.0*r*kRe*np.sin(elev_rad)) - kRe


def robust_dealias(radar, vel_field):
    """
    attempts to dealias radial velocity; if it fails, return original field name.
    """
    try:
        corr = pyart.correct.dealias_unwrap_phase(radar, vel_field=vel_field)
        field_name = vel_field + "_dealiased"
        radar.add_field_like(vel_field, field_name, corr["data"], replace_existing=True)
        print(f"[INFO] Dealiasing succeeded. Using field '{field_name}'.")
        return field_name
    except Exception as e:
        print(f"[WARN] Dealiasing failed ({e}); using original field '{vel_field}'.")
        return vel_field


def build_geometry(radar, vel_field, max_elev_deg=MAX_ELEV_DEG):
    """
    build per-gate geometry arrays:
      az    : (nrays, ngates) azimuth in radians
      r     : (nrays, ngates) range in meters
      z_agl : (nrays, ngates) height AGL in meters
      vr    : (nrays, ngates) radial velocity (m/s, NaN where missing)
      ok_ray: (nrays,) boolean ray-quality mask
    """
    vr = masked_to_nan(radar.fields[vel_field]["data"])       # (nr, ng)
    rng = np.array(radar.range["data"], dtype=float)          # (ng,)
    elev_deg = np.array(radar.elevation["data"], dtype=float) # (nr,)
    azim_deg = np.array(radar.azimuth["data"], dtype=float)   # (nr,)

    nr, ng = vr.shape

    # basic ray QC: finite elev/az and not too steep
    ok_ray = np.isfinite(elev_deg) & np.isfinite(azim_deg) & (np.abs(elev_deg) <= max_elev_deg)

    # broadcast azimuth and range to gate shape
    az = np.radians(azim_deg)[:, None]          # (nr, 1)
    az = np.broadcast_to(az, (nr, ng))          # (nr, ng)
    r = np.broadcast_to(rng[None, :], (nr, ng)) # (nr, ng)

    # height above radar using 4/3-Earth; use median elev per sweep
    z_agl = np.empty_like(vr, dtype=float)
    elev_rad = np.radians(elev_deg)
    s0 = radar.sweep_start_ray_index["data"]
    s1 = radar.sweep_end_ray_index["data"] + 1

    for sw in range(radar.nsweeps):
        a = int(s0[sw])
        b = int(s1[sw])
        if a >= b:
            continue
        el_med = np.nanmedian(elev_rad[a:b])
        if not np.isfinite(el_med):
            el_med = 0.0
        # r[a:b, :] already shape (nrays_in_sweep, ngates)
        z_above = gate_height_43earth(r[a:b, :], el_med)  # above radar
        z_agl[a:b, :] = z_above

    return az, r, z_agl, vr, ok_ray


def make_height_bins(z_agl, z_width=Z_BIN_WIDTH, z_max=Z_MAX):
    """
    create vertical bins and return:
      bins  : edges (0..Z_MAX)
      z_mid : bin midpoints (AGL)
      z_idx : per-gate bin index (or -1 if out-of-range)
    """
    bins = np.arange(0.0, z_max + z_width, z_width)
    z_mid = bins[:-1] + 0.5 * z_width
    # Digitize per gate
    z_idx = np.digitize(z_agl, bins) - 1
    z_idx[(z_idx < 0) | (z_idx >= len(z_mid))] = -1
    return bins, z_mid, z_idx


def azimuth_sector_count(az_values_rad, n_sectors=12):
    """
    count distinct azimuth sectors with data (e.g., 12 → 30° sectors).
    """
    if az_values_rad.size == 0:
        return 0
    a = az_values_rad % (2.0 * np.pi)
    sectors = np.floor(a * n_sectors / (2.0 * np.pi)).astype(int)
    return np.unique(sectors).size


# VAD (binwise) computation function

def vad_profile(radar, vel_field):
    """
    classic VAD:
      Vr = U cos(az) + V sin(az)
    fit independently per height bin (no vertical coupling).
    includes:
      - ray QC (max elevation)
      - minimum gate count per bin
      - azimuthal coverage check (sector count)
      - optional 1/r weighting
    returns:
      z_mid : [m AGL]
      U, V  : [m/s]
    """
    az, r, z_agl, vr, ok_ray = build_geometry(radar, vel_field)
    _, z_mid, z_idx = make_height_bins(z_agl)

    nb = len(z_mid)
    U = np.full(nb, np.nan, dtype=float)
    V = np.full(nb, np.nan, dtype=float)

    raymask = ok_ray[:, None]
    valid = np.isfinite(vr) & raymask
    weights = (1.0 / np.maximum(r, 1.0)) if USE_RANGE_WEIGHT else np.ones_like(r)

    for k in range(nb):
        m = (z_idx == k) & valid
        npts = int(m.sum())
        if npts < MIN_GATES:
            continue

        # azimuthal coverage QC
        if azimuth_sector_count(az[m], n_sectors=12) < MIN_AZ_SECTORS:
            continue

        y = vr[m]
        cos1 = np.cos(az[m])
        sin1 = np.sin(az[m])
        X = np.column_stack([cos1, sin1])

        if USE_RANGE_WEIGHT:
            w = weights[m]
            Xw = X * w[:, None]
            yw = y * w
            sol, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        else:
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        U[k], V[k] = sol

    return z_mid, U, V


# -EVAD (global) computation

def evad_profile(radar,
                 vel_field,
                 include_harmonics=INCLUDE_HARMONICS,
                 include_r_term=INCLUDE_R_TERM,
                 lambda_smooth=LAMBDA_SMOOTH):
    
    """
    EVAD (Extended VAD) as a global vertical inversion w// smoothing.

    per gate i in vertical bin k:

      Vr_i = U_k cos(az_i) + V_k sin(az_i)
             + 0.5 * r_i [ A_k cos(2 az_i) + B_k sin(2 az_i) ]    (optional 2θ)
             + D_k r_i                                          (optional r-term)

    unknowns per bin:
      U, V         : mean horizontal wind
      A, B (opt)   : 2θ deformation-like harmonics
      D (opt)      : r-term divergence proxy

    stacks all gates into one big linear system and applies vertical smoothness
    on each parameter w// a Tikhonov penalty (i.e., first differences).
    """
    az, r, z_agl, vr, ok_ray = build_geometry(radar, vel_field)
    _, z_mid, z_idx = make_height_bins(z_agl)
    nb = len(z_mid)

    # unknown structure per bin
    nvar_bin = 2 + (2 if include_harmonics else 0) + (1 if include_r_term else 0)
    OFF_U, OFF_V = 0, 1
    OFF_A = 2 if include_harmonics else None
    OFF_B = 3 if include_harmonics else None
    OFF_D = (nvar_bin - 1) if include_r_term else None

    raymask = ok_ray[:, None]
    valid = np.isfinite(vr) & raymask & (z_idx >= 0)

    if not np.any(valid):
        raise RuntimeError("No valid gates after QC for EVAD.")

    # flatten all obs
    azv = az[valid]
    rv  = r[valid]
    kv  = z_idx[valid]
    y   = vr[valid]

    if USE_RANGE_WEIGHT:
        W = 1.0 / np.maximum(rv, 1.0)
    else:
        W = np.ones_like(rv)

    cos1 = np.cos(azv)
    sin1 = np.sin(azv)
    if include_harmonics:
        cos2 = np.cos(2.0 * azv)
        sin2 = np.sin(2.0 * azv)
    if include_r_term:
        rfac = rv

    nobs = y.size
    A = np.zeros((nobs, nb * nvar_bin), dtype=float)

    # fill design matrix row-by-row
    for i in range(nobs):
        k = int(kv[i])         # bin index
        j0 = k * nvar_bin
        A[i, j0 + OFF_U] = cos1[i]
        A[i, j0 + OFF_V] = sin1[i]
        if include_harmonics:
            A[i, j0 + OFF_A] = 0.5 * rv[i] * cos2[i]
            A[i, j0 + OFF_B] = 0.5 * rv[i] * sin2[i]
        if include_r_term:
            A[i, j0 + OFF_D] = rfac[i]

    # apply weights
    Aw = A * W[:, None]
    yw = y * W

    # vertical smoothing (Tikhonov) on differences x_{k+1} - x_k
    def add_smoothness(Aw, yw, lam, nb, nvar_bin, offsets):
        if lam <= 0:
            return Aw, yw
        rows = []
        for off in offsets:
            for k in range(nb - 1):
                row = np.zeros(nb * nvar_bin, dtype=float)
                row[k * nvar_bin + off] = -1.0
                row[(k + 1) * nvar_bin + off] = +1.0
                rows.append(np.sqrt(lam) * row)
        if rows:
            Sm = np.vstack(rows)
            zeros = np.zeros(Sm.shape[0], dtype=float)
            Aw = np.vstack([Aw, Sm])
            yw = np.concatenate([yw, zeros])
        return Aw, yw

    smooth_offsets = [OFF_U, OFF_V]
    if include_harmonics:
        smooth_offsets += [OFF_A, OFF_B]
    if include_r_term:
        smooth_offsets += [OFF_D]

    Aw, yw = add_smoothness(Aw, yw, lambda_smooth, nb, nvar_bin, smooth_offsets)

    # solve normal equations
    sol, *_ = np.linalg.lstsq(Aw, yw, rcond=None)

    # unpack solutions
    U = np.full(nb, np.nan)
    V = np.full(nb, np.nan)
    A2 = np.full(nb, np.nan) if include_harmonics else None
    B2 = np.full(nb, np.nan) if include_harmonics else None
    Dp = np.full(nb, np.nan) if include_r_term else None

    for k in range(nb):
        j0 = k * nvar_bin
        U[k] = sol[j0 + OFF_U]
        V[k] = sol[j0 + OFF_V]
        if include_harmonics:
            A2[k] = sol[j0 + OFF_A]
            B2[k] = sol[j0 + OFF_B]
        if include_r_term:
            Dp[k] = sol[j0 + OFF_D]

    out = {
        "z_agl": z_mid,
        "U": U,
        "V": V,
        "speed": np.hypot(U, V),
        "dir_from_deg": (np.degrees(np.arctan2(-U, -V)) % 360.0),
        "A2": A2,                    # 2θ deformation-like term (cos component)
        "B2": B2,                    # 2θ deformation-like term (sin component)
        "D_proxy": Dp,               # r-term coefficient; div ≈ -2*D_proxy (very approximate)
        "meta": {
            "bin_width_m": Z_BIN_WIDTH,
            "z_max_m": Z_MAX,
            "min_gates": MIN_GATES,
            "min_az_sectors": MIN_AZ_SECTORS,
            "weights": "1/r" if USE_RANGE_WEIGHT else "none",
            "include_harmonics": include_harmonics,
            "include_r_term": include_r_term,
            "lambda_smooth": lambda_smooth
        }
    }
    return out


# additional plotting functions 

def plot_profile(z_agl, U, V, title="Wind Profile"):
    spd = np.hypot(U, V)
    drn = (np.degrees(np.arctan2(-U, -V)) % 360.0)
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 8))

    ax[0].plot(spd, z_agl, "-o")
    ax[1].plot(drn, z_agl, "-o")

    ax[0].set_xlabel("Wind speed (m/s)")
    ax[1].set_xlabel("Direction-from (deg)")
    ax[0].set_ylabel("Height AGL (m)")

    for a in ax:
        a.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_divergence_proxy(z_agl, D_proxy, title="Divergence Proxy (EVAD)"):
    if D_proxy is None:
        print("[INFO] D_proxy not available (include_r_term=False).")
        return
    # crude mapping: div ~ -2 * D_proxy (units s^-1)
    div_est = -2.0 * D_proxy
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.plot(div_est * 1e5, z_agl, "-o")
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("Divergence (×10⁻⁵ s⁻¹) [approximate]")
    ax.set_ylabel("Height AGL (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --------------------------- main --------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", 
        default="/Users/savannahsouthward/Downloads/KTLX20211027_100132_V06")
    args = parser.parse_args()

    filename = args.filename 

    print(f"[INFO] Reading radar file: {args.filename}")
    radar = pyart.io.read_nexrad_archive(args.filename)

    # optional dealias
    vel_field = VEL_FIELD
    if DEALIAS_FIRST:
        vel_field = robust_dealias(radar, vel_field)

    # VAD
    print("[INFO] Computing VAD profile...")
    z_vad, U_vad, V_vad = vad_profile(radar, vel_field)
    #plot_profile(z_vad, U_vad, V_vad, title="VAD (binwise) Wind Profile")

    # EVAD
    print("[INFO] Computing EVAD profile...")
    ev = evad_profile(radar,
                      vel_field=vel_field,
                      include_harmonics=INCLUDE_HARMONICS,
                      include_r_term=INCLUDE_R_TERM,
                      lambda_smooth=LAMBDA_SMOOTH)

    # DEBUG SECTION (reduce lambda if data is too smoothed)
    print("\n====== EVAD DIAGNOSTICS ======")
    print("EVAD U (m/s):", ev["U"])
    print("EVAD V (m/s):", ev["V"])
    print("EVAD speed (m/s):", np.hypot(ev["U"], ev["V"]))
    print("================================\n")

    #plot_profile(ev["z_agl"], ev["U"], ev["V"], title="EVAD (global, smoothed) Wind Profile")
    #plot_divergence_proxy(ev["z_agl"], ev["D_proxy"], title="EVAD Divergence Proxy")
 
    print("[INFO] Done.")
    
    ###########################################################################
    # 4-PANEL SUMMARY FIGURE FOR REPORT: reflectivity, VAD, EVAD, & divergence
    ###########################################################################

    rcParams["font.family"] = "Times New Roman"
    rcParams["figure.dpi"] = 500
    
    fig = plt.figure(figsize=(18, 12), dpi=500)
    
    # reflectivity plot 
    ax1 = fig.add_subplot(2, 2, 1)
    display = pyart.graph.RadarDisplay(radar)
    display.plot_ppi(
        REF_FIELD,
        sweep=0,
        ax=ax1,
        cmap="pyart_NWSRef",
        vmin=-10,
        vmax=70,
        title="Reflectivity (dBZ)",
        colorbar_label="dBZ"
    )
    # change axis if needed 
    ax1.set_xlim(-50, 50)   # km east-west
    ax1.set_ylim(-50, 50)   # km north-south
    
    # VAD plot
    ax2 = fig.add_subplot(2, 2, 2)
    vad_speed = np.hypot(U_vad, V_vad)
    
    ax2.plot(vad_speed, z_vad, lw=2, color="navy")
    ax2.set_xlabel("Wind Speed (m/s)")
    ax2.set_ylabel("Height (m AGL)")
    ax2.set_title("VAD Wind Profile")
    ax2.grid(True, ls="--", alpha=0.4)
    
    # EVAD plot
    ax3 = fig.add_subplot(2, 2, 3)
    evad_speed = np.hypot(ev["U"], ev["V"])
    
    ax3.plot(evad_speed, ev["z_agl"], lw=2, color="darkred")
    ax3.set_xlabel("Wind Speed (m/s)")
    ax3.set_ylabel("Height (m AGL)")
    ax3.set_title("EVAD Wind Profile")
    ax3.grid(True, ls="--", alpha=0.4)
    
    # divergence proxy plot
    ax4 = fig.add_subplot(2, 2, 4)
    
    if ev["D_proxy"] is not None:
        div_est = -2.0 * ev["D_proxy"]   # approximate divergence
        ax4.plot(div_est * 1e5, ev["z_agl"], lw=2, color="purple")
        ax4.axvline(0.0, color="black", lw=1)
        ax4.set_xlabel(r"Divergence ($\times10^{-5}\ \mathrm{s^{-1}}$)")
        ax4.set_ylabel("Height (m AGL)")
        ax4.set_title("Divergence Proxy (EVAD)")
        ax4.grid(True, ls="--", alpha=0.4)
    else:
        ax4.text(0.5, 0.5, "No divergence term", ha="center")
        

    timestamp = radar.time['units'].split()[-1].replace('T',' ').replace('Z',' UTC')
    radar_id  = radar.metadata['instrument_name']  # usually 'KTLX', 'KINX', etc.

    fig.suptitle(
        f"{radar_id} – {REF_FIELD.upper()}, VAD, EVAD, and Divergence Diagnostics\n"
        f"{timestamp}",
        fontsize=20, fontweight='bold', y=0.97
    )

    plt.tight_layout()
    plt.savefig("nllj_case.png", dpi=500, bbox_inches="tight")
    for ax in (ax2, ax3, ax4):  # VAD, EVAD, divergence axes
        ax.set_ylim(0, 6000)    # or 0, 5000, etc. depending on case
    plt.show()

if __name__ == "__main__":
    main()
