"""
GRB260207A — cand41148 joint fit
Run interactively to zoom in on problem areas.

Usage: python plot_GRB260207A_cand41148.py
Dependencies: numpy, matplotlib, scipy, astropy
Input file:   lc_GRB260207A_cand41148
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.time import Time

# ---------------------------------------------------------------
# Trigger time (non-barycentric)
# ---------------------------------------------------------------
trigger_tjd = Time("2026-02-07 05:40:16.947").jd - 2457000

# ---------------------------------------------------------------
# MASTER reference point  (GCN 43633 / Lipunov et al.)
# Obs time: 2026-02-07.24049 UT  =>  T+~6 min
# Magnitude: 17.3 unfiltered (magnitude system not specified)
# Flux: approximate, ~30% systematic uncertainty
# May include prompt emission tail at this early epoch
# ---------------------------------------------------------------
master_day_frac = 0.24049
master_hr  = master_day_frac * 24
master_h   = int(master_hr)
master_m   = int((master_hr - master_h) * 60)
master_s   = ((master_hr - master_h) * 60 - master_m) * 60
master_dt  = Time(f"2026-02-07 {master_h:02d}:{master_m:02d}:{master_s:06.3f}")
t_master   = master_dt.jd - 2457000 - trigger_tjd
F_master   = 3631.0 * 10**(-0.4 * (17.3 + 0.16))   # approx AB flux, Jy
eF_master  = F_master * 0.30                          # 30% systematic

# ---------------------------------------------------------------
# Load and convert TESS data
# ---------------------------------------------------------------
rawdata  = np.loadtxt('lc_GRB260207A_cand41148')
x_all    = rawdata[:, 0] - trigger_tjd
y_all    = -rawdata[:, 1] / (200 * 0.8 * 0.99)   # cts/s
yerr_all =  rawdata[:, 2] / (200 * 0.8 * 0.99)   # cts/s

zp        = 2416 * 10**(-0.4 * 20.44)             # Jy per cts/s
flux_all  = y_all    * zp
eflux_all = yerr_all * zp

# ---------------------------------------------------------------
# Cut to 1e-3 to 1 day
# ---------------------------------------------------------------
mask_cut = (x_all >= 1e-3) & (x_all <= 1.0)
x     = x_all[mask_cut]
flux  = flux_all[mask_cut]
eflux = eflux_all[mask_cut]

# ---------------------------------------------------------------
# Log-binning from 0.055 days
# ---------------------------------------------------------------
BIN_START = 0.055

x_e  = x[x <  BIN_START];  y_e  = flux[x <  BIN_START];  ye_e  = eflux[x <  BIN_START]
x_l  = x[x >= BIN_START];  y_l  = flux[x >= BIN_START];  ye_l  = eflux[x >= BIN_START]

def log_bin(x, y, ye, bpd=10):
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()),
                        max(2, int(round(np.log10(x.max()/x.min()) * bpd)) + 1))
    xb, yb, yeb = [], [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i+1])
        if m.sum() == 0: continue
        xb.append(np.mean(x[m])); yb.append(np.mean(y[m]))
        yeb.append(np.sqrt(np.sum(ye[m]**2)) / m.sum())
    return np.array(xb), np.array(yb), np.array(yeb)

x_bin, y_bin, ye_bin = log_bin(x_l, y_l, ye_l)

x_plot  = np.concatenate([x_e,   x_bin])
y_plot  = np.concatenate([y_e,   y_bin])
ye_plot = np.concatenate([ye_e,  ye_bin])

# ---------------------------------------------------------------
# Model functions
# Indices in decay-positive convention:
#   negative alpha = rising, positive alpha = decaying
# ---------------------------------------------------------------
S1 = 0.1    # Peak 1 smoothness
S2 = 0.02   # Peak 2 smoothness

def sbpl(t, F0, tb, a1, a2):
    """Single smoothly broken power law — Peak 1."""
    S = S1
    return F0 * (t/tb)**(-a1) * (0.5*(1 + (t/tb)**(1/S)))**(-(a2-a1)*S)

def tsbpl(t, F0, tb1, tb2, tb3, a1, a2, a3, a4):
    """Triple smoothly broken power law — Peak 2 (4 segments)."""
    S = S2
    def bkn(t, tb, da): return (0.5*(1 + (t/tb)**(1/S)))**(-da*S)
    return F0*(t/tb1)**(-a1)*bkn(t,tb1,a2-a1)*bkn(t,tb2,a3-a2)*bkn(t,tb3,a4-a3)

def F_total(t, F0, tb_1, a1_1, a2_1,
               F1, tb1_2, tb2_2, tb3_2, a1_2, a2_2, a3_2, a4_2):
    """Joint model: Peak1 (SBPL) + Peak2 (TSBPL)."""
    pk1 = sbpl( t, F0, tb_1,  a1_1, a2_1)
    pk2 = tsbpl(t, F1, tb1_2, tb2_2, tb3_2, a1_2, a2_2, a3_2, a4_2)
    return np.where(pk1>0, pk1, 0) + np.where(pk2>0, pk2, 0)

# ---------------------------------------------------------------
# Joint fit
# Strictly ordered break times: tb1_2 < tb2_2 < tb3_2
# ---------------------------------------------------------------
mask_fit = (x_plot >= 1e-3) & (x_plot <= 1.0) & (ye_plot > 0)
xf = x_plot[mask_fit]; yf = y_plot[mask_fit]; yef = ye_plot[mask_fit]

#            F0      tb_1   a1_1  a2_1    F1      tb1_2   tb2_2   tb3_2   a1_2   a2_2  a3_2   a4_2
p0        = [6e-4,  0.010, -0.5,  2.0,   4e-4,   0.032,  0.049,  0.080,  -3.0,  0.0,   2.0,   1.0]
bounds_lo = [0,     0.005, -3.0,  0.5,   0,      0.030,  0.046,  0.055,  -8.0, -0.5,   0.5,   0.0]
bounds_hi = [5e-3,  0.020,  0.0,  5.0,   5e-3,   0.035,  0.052,  0.200,   0.0,  1.5,   8.0,   3.0]
# Break ordering enforced:
#   tb1 max (0.035 d) < tb2 min (0.046 d)
#   tb2 max (0.052 d) < tb3 min (0.055 d)

r, pcov = curve_fit(F_total, xf, yf, p0=p0, bounds=(bounds_lo, bounds_hi),
                    sigma=yef, absolute_sigma=True, maxfev=500000)
e   = np.sqrt(np.diag(pcov))
rc  = np.sum(((yf - F_total(xf, *r)) / yef)**2) / (len(xf) - len(r))

print("JOINT FIT RESULTS:")
print(f"  chi2_r = {rc:.3f}  (N={len(xf)}, dof={len(xf)-len(r)})")
print(f"\nPeak 1 (SBPL, s={S1}):")
print(f"  F0  = {r[0]*1e6:.2f} +/- {e[0]*1e6:.2f} uJy")
print(f"  tb  = {r[1]*1440:.2f} +/- {e[1]*1440:.2f} min")
print(f"  a1  = {r[2]:.3f} +/- {e[2]:.3f}  (rise)")
print(f"  a2  = {r[3]:.3f} +/- {e[3]:.3f}  (decay)")
print(f"\nPeak 2 (TSBPL, s={S2}):")
print(f"  F0  = {r[4]*1e6:.2f} +/- {e[4]*1e6:.2f} uJy")
print(f"  tb1 = {r[5]*1440:.2f} +/- {e[5]*1440:.2f} min  (rise -> plateau)")
print(f"  tb2 = {r[6]*1440:.2f} +/- {e[6]*1440:.2f} min  (plateau -> decay)")
print(f"  tb3 = {r[7]*1440:.2f} +/- {e[7]*1440:.2f} min  (decay -> late)")
print(f"  a1  = {r[8]:.3f} +/- {e[8]:.3f}  (rise)")
print(f"  a2  = {r[9]:.3f} +/- {e[9]:.3f}  (plateau)")
print(f"  a3  = {r[10]:.3f} +/- {e[10]:.3f}  (decay)")
print(f"  a4  = {r[11]:.3f} +/- {e[11]:.3f}  (late)")

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
t_model = np.logspace(np.log10(1e-3), np.log10(1.0), 1000)
pk1_m   = sbpl( t_model, r[0], r[1], r[2], r[3])
pk2_m   = tsbpl(t_model, r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11])
tot_m   = np.where(pk1_m>0, pk1_m, 0) + np.where(pk2_m>0, pk2_m, 0)

fig, ax = plt.subplots(figsize=(9, 6))

pos = y_plot > 0
neg = ~pos

if neg.any():
    ax.errorbar(x_plot[neg], np.abs(y_plot[neg]), yerr=ye_plot[neg],
                fmt='.', color='lightgray', markersize=3, elinewidth=0.5,
                alpha=0.4, capsize=0, zorder=1)
ax.errorbar(x_plot[pos], y_plot[pos], yerr=ye_plot[pos],
            fmt='.', color='black', markersize=3, elinewidth=0.5,
            alpha=0.8, capsize=0, zorder=2, label='TESS data (cand41148)')

ax.errorbar(t_master, F_master, yerr=eF_master, fmt='D',
            color='mediumseagreen', markersize=9, elinewidth=1.4,
            capsize=4, zorder=7, markeredgecolor='k', markeredgewidth=0.6,
            label=f'MASTER unfiltered T+{t_master*1440:.1f} min\n'
                   '(ref. only; mag system unknown;\n'
                   ' may include prompt emission)')

ax.plot(t_model, tot_m, '-',  color='crimson',   lw=2,   zorder=6,
        label=f'Total (χ²_r={rc:.2f})')
ax.plot(t_model, np.where(pk1_m>0, pk1_m, np.nan), '--', color='goldenrod', lw=1.8, zorder=5,
        label=f'P1 SBPL  tb={r[1]*1440:.1f} min')
ax.plot(t_model, np.where(pk2_m>0, pk2_m, np.nan), '--', color='steelblue', lw=1.8, zorder=5,
        label=f'P2 TSBPL  tb1={r[5]*1440:.1f}, tb2={r[6]*1440:.1f}, tb3={r[7]*1440:.1f} min')

for tb, col in [(r[1],'goldenrod'), (r[5],'steelblue'), (r[6],'steelblue'), (r[7],'steelblue')]:
    ax.axvline(tb, color=col, lw=0.8, ls=':', alpha=0.6)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1e-3, 1.0)
vis = (x_plot >= 1e-3) & (x_plot <= 1.0) & (y_plot > 0)
yhi = max(y_plot[vis].max(), F_master + eF_master) * 3
ax.set_ylim(ye_plot[vis].min() * 0.3, yhi)
ax.set_xlabel('Days post-burst', fontsize=12)
ax.set_ylabel('Flux density (Jy)', fontsize=12)

# Twin x-axis in minutes
ax_min = ax.twiny()
ax_min.set_xscale('log')
ax_min.set_xlim(1e-3, 1.0)
min_ticks = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
ax_min.set_xticks([m/1440. for m in min_ticks])
ax_min.set_xticklabels([str(m) for m in min_ticks], fontsize=9)
ax_min.set_xlabel('Minutes post-burst', fontsize=11)

ax.set_title('GRB260207A — cand41148 joint fit', fontsize=12, pad=28)
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, which='both', alpha=0.2, lw=0.5)
plt.tight_layout()
plt.show()
