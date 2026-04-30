"""
GRB260207A — cand41148 light curve analysis
Generates: GRB260207A_cand41148.png

Trigger time: 2026-02-07 05:40:16.947 UT (non-barycentric, non-geocentric)
Timing note: inter-instrument light travel time corrections (Fermi/TESS/MASTER)
             are at most ~seconds and negligible at the scale of this plot.

Data conversion:
    y    = -rawdata[:, 1] / (200 * 0.8 * 0.99)   [cts/s]
    yerr =  rawdata[:, 2] / (200 * 0.8 * 0.99)   [cts/s]
    flux = y * zp  where zp = 2416 * 10^(-0.4 * 20.44)  [Jy per cts/s]

No baseline subtraction applied.
Log-binning applied from 0.03 days onwards (10 bins/decade).

All points whose 200s integration window overlaps with t > 0 are shown:
  - Straddle point (centre T+0.17 min, window T-1.5 to T+1.84 min):
    shown as open red diamond, excluded from fit — integration spans trigger
  - T+3.5 min point: shown as open black circle, excluded from fit
  - t >= 4e-3 days (~5.8 min): included in fit

Fit: joint SBPL + SBPL with shared decay index, s=0.1 for both peaks.

MASTER point (GCN 43633, Lipunov et al.):
    T+~6 min, MASTER W = 0.2B + 0.8R (Vega), 17.3 mag
    Vega zp: W_zp = 0.2*4063 + 0.8*3080 = 3276 Jy
    ~30% systematic flux uncertainty
    Shown for reference only, not fitted.

Dependencies: numpy, matplotlib, scipy, astropy
Input file:   lc_GRB260207A_cand41148
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.time import Time

parser = argparse.ArgumentParser()
parser.add_argument('--unbinned', action='store_true',
                    help='Skip log-binning; plot and fit all data points')
parser.add_argument('--separate_decay', action='store_true',
                    help='Fit Peak 1 and Peak 2 decay indices independently')
args = parser.parse_args()
UNBINNED       = args.unbinned
SEPARATE_DECAY = args.separate_decay

# ---------------------------------------------------------------
# Trigger time (non-barycentric)
# ---------------------------------------------------------------
trigger_tjd = Time("2026-02-07 05:40:16.947").jd - 2457000

# ---------------------------------------------------------------
# MASTER reference point  (GCN 43633 / Lipunov et al.)
# MASTER W = 0.2B + 0.8R in Vega system
# Vega zp: B=4063 Jy, R=3080 Jy => W_zp = 0.2*4063 + 0.8*3080 = 3276 Jy
# ---------------------------------------------------------------
master_obs = Time("2026-02-07 05:46:04.3")   # 2026-02-07.24049 UT
t_master   = master_obs.jd - 2457000 - trigger_tjd
F_master   = 3276.0 * 10**(-0.4 * 17.3)     # Vega flux, Jy
eF_master  = F_master * 0.30                  # 30% systematic

# ---------------------------------------------------------------
# Load and convert TESS data
# ---------------------------------------------------------------
rawdata  = np.loadtxt('lc_GRB260207A_cand41148_geo')
x_all    = rawdata[:, 0] - trigger_tjd
y_all    = -rawdata[:, 1] / (200 * 0.8 * 0.99)   # cts/s
yerr_all =  rawdata[:, 2] / (200 * 0.8 * 0.99)   # cts/s

zp        = 2416 * 10**(-0.4 * 20.44)             # Jy per cts/s
flux_all  = y_all    * zp
eflux_all = yerr_all * zp

# ---------------------------------------------------------------
# Select all points whose integration window overlaps post-burst
# TESS cadence = 200s; integration runs [centre - 100s, centre + 100s]
# Window overlaps t>0 when: centre + 100s > 0 => centre > -100s
# ---------------------------------------------------------------
CADENCE = 200 / 86400   # days
HALF_C  = CADENCE / 2   # days

mask_plot = (x_all > -HALF_C) & (x_all <= 1.0)
x_plot_all  = x_all[mask_plot]
y_plot_all  = flux_all[mask_plot]
ye_plot_all = eflux_all[mask_plot]

# ---------------------------------------------------------------
# Binning (or not)
# ---------------------------------------------------------------
BIN_START = 0.03

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

if UNBINNED:
    x_plot  = x_plot_all
    y_plot  = y_plot_all
    ye_plot = ye_plot_all
else:
    x_e  = x_plot_all[x_plot_all <  BIN_START]
    y_e  = y_plot_all[x_plot_all <  BIN_START]
    ye_e = ye_plot_all[x_plot_all < BIN_START]
    x_l  = x_plot_all[x_plot_all >= BIN_START]
    y_l  = y_plot_all[x_plot_all >= BIN_START]
    ye_l = ye_plot_all[x_plot_all >= BIN_START]
    x_bin, y_bin, ye_bin = log_bin(x_l, y_l, ye_l)
    x_plot  = np.concatenate([x_e,   x_bin])
    y_plot  = np.concatenate([y_e,   y_bin])
    ye_plot = np.concatenate([ye_e,  ye_bin])

# ---------------------------------------------------------------
# Model: two SBPL, s=0.1
# Indices in decay-positive convention:
#   negative alpha = rising, positive alpha = decaying
# ---------------------------------------------------------------
S = 0.1

def sbpl(t, F0, tb, a1, a2):
    """Smoothly broken power law."""
    return F0 * (t/tb)**(-a1) * (0.5*(1 + (t/tb)**(1/S)))**(-(a2-a1)*S)

def F_total(t, F0, tb_1, a1_1, F1, tb_2, a1_2, alpha_decay):
    """Joint model: Peak1 + Peak2, shared decay index (7 params)."""
    pk1 = sbpl(t, F0, tb_1, a1_1, alpha_decay)
    pk2 = sbpl(t, F1, tb_2, a1_2, alpha_decay)
    return np.where(pk1>0, pk1, 0) + np.where(pk2>0, pk2, 0)

def F_total_separate(t, F0, tb_1, a1_1, a2_1, F1, tb_2, a1_2, a2_2):
    """Joint model: Peak1 + Peak2, independent decay indices (8 params)."""
    pk1 = sbpl(t, F0, tb_1, a1_1, a2_1)
    pk2 = sbpl(t, F1, tb_2, a1_2, a2_2)
    return np.where(pk1>0, pk1, 0) + np.where(pk2>0, pk2, 0)

# ---------------------------------------------------------------
# Fit: t >= 4e-3 days (~5.8 min)
# Excluded:
#   - Straddle point (T+0.17 min): integration spans T=0
#   - T+3.5 min point: may include prompt emission / early excess
# ---------------------------------------------------------------
FIT_START = 4e-3   # days

mask_fit = (x_plot >= FIT_START) & (x_plot <= 1.0) & (ye_plot > 0)
xf = x_plot[mask_fit]; yf = y_plot[mask_fit]; yef = ye_plot[mask_fit]

if SEPARATE_DECAY:
    #               F0      tb_1   a1_1  a2_1    F1      tb_2    a1_2   a2_2
    p0        = [6e-4,  0.010, -0.5,  1.0,   4e-4,  0.055,  -8.0,   1.0]
    bounds_lo = [0,     0.005, -3.0,  0.2,   0,     0.030,  -20.0,   0.2]
    bounds_hi = [5e-3,  0.020,  0.0,  5.0,   5e-3,  0.100,   0.0,   5.0]
    r, pcov = curve_fit(F_total_separate, xf, yf, p0=p0,
                        bounds=(bounds_lo, bounds_hi),
                        sigma=yef, absolute_sigma=True, maxfev=500000)
    e   = np.sqrt(np.diag(pcov))
    rc  = np.sum(((yf - F_total_separate(xf, *r)) / yef)**2) / (len(xf) - len(r))

    print(f"Trigger TJD:  {trigger_tjd:.6f}")
    print(f"MASTER:       t={t_master*1440:.2f} min,  F={F_master*1e6:.0f} uJy (Vega)")
    print(f"\nJOINT FIT (t >= {FIT_START*1440:.0f} min, two SBPL, separate decay, s={S}):")
    print(f"  chi2_r     = {rc:.3f}  (N={len(xf)}, dof={len(xf)-len(r)})")
    print(f"\nPeak 1 (SBPL):")
    print(f"  F0         = {r[0]*1e6:.2f} +/- {e[0]*1e6:.2f} uJy")
    print(f"  tb         = {r[1]*1440:.2f} +/- {e[1]*1440:.2f} min")
    print(f"  a1 (rise)  = {r[2]:.3f} +/- {e[2]:.3f}")
    print(f"  a2 (decay) = {r[3]:.3f} +/- {e[3]:.3f}")
    print(f"  => p1 (ISM slow cooling) = {r[3]*4/3 + 1:.3f}")
    print(f"\nPeak 2 (SBPL):")
    print(f"  F0         = {r[4]*1e6:.2f} +/- {e[4]*1e6:.2f} uJy")
    print(f"  tb         = {r[5]*1440:.2f} +/- {e[5]*1440:.2f} min")
    print(f"  a1 (rise)  = {r[6]:.3f} +/- {e[6]:.3f}")
    print(f"  a2 (decay) = {r[7]:.3f} +/- {e[7]:.3f}")
    print(f"  => p2 (ISM slow cooling) = {r[7]*4/3 + 1:.3f}")

else:
    #            F0      tb_1   a1_1    F1      tb_2    a1_2   alpha_decay
    p0        = [6e-4,  0.010, -0.5,   4e-4,  0.055,  -3.0,   1.0]
    bounds_lo = [0,     0.005, -3.0,   0,     0.030,  -8.0,   0.2]
    bounds_hi = [5e-3,  0.020,  0.0,   5e-3,  0.100,   0.0,   5.0]
    r, pcov = curve_fit(F_total, xf, yf, p0=p0, bounds=(bounds_lo, bounds_hi),
                        sigma=yef, absolute_sigma=True, maxfev=500000)
    e   = np.sqrt(np.diag(pcov))
    rc  = np.sum(((yf - F_total(xf, *r)) / yef)**2) / (len(xf) - len(r))

    print(f"Trigger TJD:  {trigger_tjd:.6f}")
    print(f"MASTER:       t={t_master*1440:.2f} min,  F={F_master*1e6:.0f} uJy (Vega)")
    print(f"\nJOINT FIT (t >= {FIT_START*1440:.0f} min, two SBPL, shared decay, s={S}):")
    print(f"  chi2_r     = {rc:.3f}  (N={len(xf)}, dof={len(xf)-len(r)})")
    print(f"\nPeak 1 (SBPL):")
    print(f"  F0         = {r[0]*1e6:.2f} +/- {e[0]*1e6:.2f} uJy")
    print(f"  tb         = {r[1]*1440:.2f} +/- {e[1]*1440:.2f} min")
    print(f"  a1 (rise)  = {r[2]:.3f} +/- {e[2]:.3f}")
    print(f"\nPeak 2 (SBPL):")
    print(f"  F0         = {r[3]*1e6:.2f} +/- {e[3]*1e6:.2f} uJy")
    print(f"  tb         = {r[4]*1440:.2f} +/- {e[4]*1440:.2f} min")
    print(f"  a1 (rise)  = {r[5]:.3f} +/- {e[5]:.3f}")
    print(f"\nShared decay:")
    print(f"  alpha      = {r[6]:.3f} +/- {e[6]:.3f}")
    print(f"  => p (ISM slow cooling) = {r[6]*4/3 + 1:.3f}")

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
t_model = np.logspace(np.log10(x_plot[x_plot>0].min()*0.9), np.log10(1.0), 2000)
if SEPARATE_DECAY:
    pk1_m    = sbpl(t_model, r[0], r[1], r[2], r[3])
    pk2_m    = sbpl(t_model, r[4], r[5], r[6], r[7])
    tb1, tb2 = r[1], r[5]
    label_p2 = f'P2 SBPL  tb={tb2*1440:.1f} min  |  α₁={r[3]:.2f}, α₂={r[7]:.2f} (sep.)'
else:
    pk1_m    = sbpl(t_model, r[0], r[1], r[2], r[6])
    pk2_m    = sbpl(t_model, r[3], r[4], r[5], r[6])
    tb1, tb2 = r[1], r[4]
    label_p2 = f'P2 SBPL  tb={tb2*1440:.1f} min  |  α={r[6]:.2f} (shared)'
tot_m = np.where(pk1_m>0, pk1_m, 0) + np.where(pk2_m>0, pk2_m, 0)

fig, ax = plt.subplots(figsize=(9, 6))

pos      = y_plot > 0
neg      = ~pos
excl     = x_plot < FIT_START
straddle = (x_plot > 0) & (x_plot < CADENCE)   # integration straddles T=0

# Negative flux — grey
if neg.any():
    ax.errorbar(x_plot[neg], np.abs(y_plot[neg]), yerr=ye_plot[neg],
                fmt='.', color='lightgray', markersize=2, elinewidth=0.4,
                alpha=0.3, capsize=0, zorder=1)

# Straddle point — open red diamond
if (straddle & pos).any():
    ax.errorbar(x_plot[straddle & pos], y_plot[straddle & pos],
                yerr=[ye_plot[straddle & pos],
                      ye_plot[straddle & pos]+(y_plot[straddle & pos]*((200/60/60/24)/((100/60/60/24)-x_plot[straddle & pos])) - y_plot[straddle & pos])],
                fmt='D', color='tomato', markersize=6, elinewidth=0.6,
                alpha=1, capsize=2, zorder=4, markerfacecolor='none',
                markeredgewidth=1.2,
                label='Straddles T=0 (excl. from fit)')

# Other excluded points — open black circles
other_excl = excl & ~straddle & pos
if other_excl.any():
    ax.errorbar(x_plot[other_excl], y_plot[other_excl],
                yerr=ye_plot[other_excl],
                fmt='o', color='black', markersize=4, elinewidth=0.5,
                alpha=1, capsize=2, zorder=3, markerfacecolor='none',
                label=f'Excl. from fit (t < {FIT_START*1440:.0f} min)')

# Fitted data — filled black dots
ax.errorbar(x_plot[~excl & pos], y_plot[~excl & pos],
            yerr=ye_plot[~excl & pos],
            fmt='.', color='black', markersize=3, elinewidth=0.4,
            alpha=0.7, capsize=0, zorder=2, label='Data (fitted)')

# MASTER reference point
ax.errorbar(t_master, F_master, yerr=eF_master, fmt='D',
            color='mediumseagreen', markersize=9, elinewidth=1.4,
            capsize=4, zorder=7, markeredgecolor='k', markeredgewidth=0.6,
            label=f'MASTER T+{t_master*1440:.1f} min\n'
                   '(ref. only; ~30% syst.)')

# Model curves
ax.plot(t_model, tot_m, '-', color='crimson', lw=2, zorder=6,
        label=f'Total (χ²_r={rc:.2f})')
ax.plot(t_model, np.where(pk1_m>0, pk1_m, np.nan), '--', color='goldenrod',
        lw=1.8, zorder=5, label=f'P1 SBPL  tb={tb1*1440:.1f} min')
ax.plot(t_model, np.where(pk2_m>0, pk2_m, np.nan), ':', color='steelblue',
        lw=1.8, zorder=5, label=label_p2)

# Break time markers
ax.axvline(tb1, color='goldenrod', lw=0.8, ls=':', alpha=0.6)
ax.axvline(tb2, color='steelblue', lw=0.8, ls=':', alpha=0.6)
ax.axvline(FIT_START, color='gray', lw=0.8, ls='--', alpha=0.4)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(x_plot[pos].min() * 0.7, 1.0)
vis = (x_plot >= 1e-4) & (x_plot <= 1.0) & (y_plot > 0)
yhi = max(y_plot[vis].max(), F_master + eF_master) * 3
ax.set_ylim(ye_plot[vis].min() * 0.3, yhi)
ax.set_xlabel('Days post-burst', fontsize=12)
ax.set_ylabel('Flux density (Jy)', fontsize=12)

# Twin x-axis in minutes
ax_min = ax.twiny()
ax_min.set_xscale('log')
ax_min.set_xlim(ax.get_xlim())
min_ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
ax_min.set_xticks([m/1440. for m in min_ticks])
ax_min.set_xticklabels([str(m) for m in min_ticks], fontsize=8)
ax_min.set_xlabel('Minutes post-burst', fontsize=11)

bin_str = 'unbinned' if UNBINNED else 'log-binned'
dec_str = 'separate decay' if SEPARATE_DECAY else 'shared decay'
ax.set_title(f'GRB260207A — cand41148  ({bin_str}, {dec_str})', fontsize=12, pad=28)
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, which='both', alpha=0.2, lw=0.5)
plt.tight_layout()
suffix = ('_unbinned' if UNBINNED else '') + ('_sepdecay' if SEPARATE_DECAY else '')
plt.savefig(f"GRB260207A{suffix}.png")
plt.close()
