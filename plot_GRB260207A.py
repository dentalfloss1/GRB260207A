"""
GRB260207A — TESS + VLA + MASTER light curve analysis
Generates 3 PNG files:
  GRB260207A_tess_full.png    — TESS full view
  GRB260207A_tess_peaks.png   — TESS peak region
  GRB260207A_radio.png        — VLA 6 GHz with ISM vs Wind extrapolations

Fitting approach:
  Peak 1 and Peak 2 are fit JOINTLY to all data simultaneously:
    F_total(t) = F_pk1(t, F0, tb, a1, a2, s=0.1)
               + F_pk2(t, F0, tb1, tb2, a1, a2, a3, s=0.02)
  Wind FS model fit jointly to TESS (0.055-0.1 d) + VLA radio.

Dependencies: numpy, pandas, matplotlib, scipy
Input files:  lc_GRB260207A_cand47734_cleaned
              GRB260207A_-_Sheet1.csv

Notes on MASTER point (GCN 35684, Lipunov et al.):
  - Observed at T+3.6 min, unfiltered (broad V-R-I bandpass)
  - Magnitude system not specified in GCN circular
  - Flux conversion approximate (~30% systematic uncertainty)
  - At T+3.6 min may include prompt emission tail
  - Plotted for reference only — not included in any fit
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import math
from datetime import datetime, timezone

# ---------------------------------------------------------------
# Trigger time
# ---------------------------------------------------------------
def _jd(dt):
    a = math.floor((14 - dt.month) / 12)
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = (dt.day + math.floor((153*m + 2)/5) + 365*y
           + math.floor(y/4) - math.floor(y/100) + math.floor(y/400) - 32045)
    return jdn + (dt.hour-12)/24 + dt.minute/1440 + (dt.second + dt.microsecond/1e6)/86400

GRB_BTJD = _jd(datetime(2026, 2, 7, 5, 47, 42, 796000, tzinfo=timezone.utc)) - 2457000
GRB_TJD = _jd(datetime(2026, 2, 7, 5, 40, 16, 947000, tzinfo=timezone.utc)) - 2457000

# ---------------------------------------------------------------
# MASTER reference point  (GCN 35684 / Lipunov et al.)
# Obs time: 2026-02-07.24049 UT  =>  T+3.6 min
# Magnitude: 17.3 unfiltered (magnitude system not specified)
# Flux: approximate, ~30% systematic uncertainty
# May include prompt emission tail at this early epoch
# ---------------------------------------------------------------
master_day_frac = 0.24049
master_hr  = master_day_frac * 24
master_h   = int(master_hr)
master_m   = int((master_hr - master_h) * 60)
master_s   = ((master_hr - master_h) * 60 - master_m) * 60
master_dt  = datetime(2026, 2, 7, master_h, master_m, int(master_s), tzinfo=timezone.utc)
t_master   = _jd(master_dt) - 2457000 - GRB_TJD
mag_master_approx_AB = 17.3 + 0.16
F_master   = 3631.0 * 10**(-0.4 * mag_master_approx_AB)
eF_master  = F_master * 0.30

# ---------------------------------------------------------------
# Load TESS data
# ---------------------------------------------------------------
df = pd.read_csv('lc_GRB260207A_cand47734_cleaned', comment='#', sep=r'\s+',
    names=['BTJD','TJD','cts_per_s','e_cts_per_s','mag','e_mag',
           'bkg','bkg_model','bkg2','e_bkg2'])

x     = df['BTJD'].values - GRB_BTJD
# bdiff     = np.average(df['BTJD'].values - df['TJD'].values)
# x     = df['TJD'].values - GRB_TJD
cts   = df['cts_per_s'].values.copy()
ects  = df['e_cts_per_s'].values
cts  -= np.median(cts[(x >= -1) & (x <= -2/24)])
zp       = 2416 * 10**(-0.4 * 20.44)
flux_jy  = cts  * zp
eflux_jy = ects * zp
sigma1   = np.std(flux_jy[(x >= -1) & (x <= -1/24)])

# ---------------------------------------------------------------
# Load VLA radio data
# ---------------------------------------------------------------
rdf = pd.read_csv('GRB260207A_-_Sheet1.csv')
t_radio    = rdf['Epoch (delta T)'].values.astype(float)
nu_radio   = rdf['Frequency '].values.astype(float) * 1e9
F_radio    = rdf['Flux density (microJy)'].values.astype(float) * 1e-6
Ferr_stat  = rdf['Flux density error'].values.astype(float) * 1e-6
Ferr_radio = np.sqrt(Ferr_stat**2 + (0.05 * F_radio)**2)

nu_TESS = 5e14
nu_6GHz = 6e9
S1 = 0.1    # Peak 1 smoothness (sparse data)
S2 = 0.02   # Peak 2 smoothness (well-sampled)

# ---------------------------------------------------------------
# Working data
# ---------------------------------------------------------------
mask_all = (x > 1e-3) & (x < 0.1)
xall = x[mask_all]; yall = flux_jy[mask_all]; yeall = eflux_jy[mask_all]

# Positive-flux data for joint fit
mask_fit = (x > 1e-3) & (x < 0.1) & (flux_jy > 0)
xfit = x[mask_fit]; yfit = flux_jy[mask_fit]; yefit = eflux_jy[mask_fit]

# ---------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------
def sbpl(t, F0, t_b, a1, a2):
    """Single smoothly broken power law, s=S1 (Peak 1)."""
    S = S1
    return F0 * (t/t_b)**(-a1) * (0.5*(1 + (t/t_b)**(1/S)))**(-(a2-a1)*S)

def dbpl(t, F0, tb1, tb2, a1, a2, a3):
    """Double smoothly broken power law, s=S2 (Peak 2)."""
    S = S2
    def bkn(t, tb, da): return (0.5*(1 + (t/tb)**(1/S)))**(-da*S)
    return F0 * (t/tb1)**(-a1) * bkn(t, tb1, a2-a1) * bkn(t, tb2, a3-a2)

def F_total(t, F0, tb1_1, a1_1, a2_1, F1, tb1_2, tb2_2, a1_2, a2_2, a3_2):
    """Joint model: Peak1 + Peak2, both positive-definite."""
    pk1 = sbpl(t, F0, tb1_1, a1_1, a2_1)
    pk2 = dbpl(t, F1, tb1_2, tb2_2, a1_2, a2_2, a3_2)
    return np.where(pk1>0, pk1, 0) + np.where(pk2>0, pk2, 0)

def fs_model(F0, p, t, nu):
    """Wind forward shock: F(t,nu) = F0*(nu/nu_TESS)^((1-p)/2)*(t/t0)^((1-3p)/4)."""
    return F0 * (nu/nu_TESS)**((1-p)/2) * (t/t0_anchor)**((1-3*p)/4)

# ---------------------------------------------------------------
# Joint morphological fit  (10 free parameters)
# ---------------------------------------------------------------
p0 = [5e-5, 0.020, -4.0, 2.7,          # P1: F0, tb, a1, a2
      5e-5, 0.047, 0.060, -3.2, -0.1, 1.9]  # P2: F0, tb1, tb2, a1, a2, a3

bounds_lo = [0,    0.010, -15.0, 0.1,   0,    0.040, 0.053, -12,  -2.0,  0.5]
bounds_hi = [1e-3, 0.034,   0.0, 15.0,  1e-3, 0.056, 0.072,  0.0,  1.0,  4.0]

r_j, pcov_j = curve_fit(F_total, xfit, yfit,
    p0=p0, bounds=(bounds_lo, bounds_hi),
    sigma=yefit, absolute_sigma=True, maxfev=200000)
e_j = np.sqrt(np.diag(pcov_j))

(F0_j, tb1_1_j, a1_1_j, a2_1_j,
 F1_j, tb1_2_j, tb2_2_j, a1_2_j, a2_2_j, a3_2_j) = r_j

ymodel_j = F_total(xfit, *r_j)
rc_j     = np.sum(((yfit - ymodel_j)/yefit)**2) / (len(xfit) - len(r_j))

# ---------------------------------------------------------------
# Wind FS model — joint optical + radio fit
# ---------------------------------------------------------------
mask_anchor = (x > 0.055) & (x < 0.065) & (flux_jy > 0)
t0_anchor   = np.median(x[mask_anchor])

mask_fs    = (x > 0.055) & (x < 0.1) & (flux_jy > 0)
t_fs, F_fs = x[mask_fs], flux_jy[mask_fs]
t_all_fs   = np.concatenate([t_fs, t_radio])
F_all_fs   = np.concatenate([F_fs, F_radio])
nu_all_fs  = np.concatenate([np.full(len(t_fs), nu_TESS), nu_radio])

res_fs = least_squares(
    lambda pv: np.log10(fs_model(pv[0], pv[1], t_all_fs, nu_all_fs)) - np.log10(F_all_fs),
    x0=[6e-5, 2.3])
F0_fs, p_fs = res_fs.x
alpha_fs  = (3 * p_fs - 1) / 4
alpha_ism = 3 * (p_fs - 1) / 4
beta      = (p_fs - 1) / 2

# ---------------------------------------------------------------
# Print results
# ---------------------------------------------------------------
print(f"Trigger TJD:  {GRB_TJD:.6f}")
print(f"t0_anchor:    {t0_anchor*1440:.2f} min")
print(f"\nMASTER (reference, not fit):")
print(f"  t = {t_master*1440:.2f} min,  F ~ {F_master*1e6:.0f} uJy (approx, unfiltered)")
print(f"  Note: mag system unspecified; may include prompt tail")
print(f"\nJOINT MORPHOLOGICAL FIT (s1={S1}, s2={S2}):")
print(f"  chi2_r = {rc_j:.3f}  (N={len(xfit)}, dof={len(xfit)-len(r_j)})")
print(f"\n  Peak 1 (SBPL):")
print(f"    F0     = {F0_j*1e6:.3f} +/- {e_j[0]*1e6:.3f} uJy")
print(f"    t_b    = {tb1_1_j*1440:.2f} +/- {e_j[1]*1440:.2f} min")
print(f"    alpha1 = {a1_1_j:.3f} +/- {e_j[2]:.3f}  (rise)")
print(f"    alpha2 = {a2_1_j:.3f} +/- {e_j[3]:.3f}  (decay)")
print(f"\n  Peak 2 (DBPL):")
print(f"    F0     = {F1_j*1e6:.3f} +/- {e_j[4]*1e6:.3f} uJy")
print(f"    tb1    = {tb1_2_j*1440:.2f} +/- {e_j[5]*1440:.2f} min  (rise->plateau)")
print(f"    tb2    = {tb2_2_j*1440:.2f} +/- {e_j[6]*1440:.2f} min  (plateau->decay)")
print(f"    alpha1 = {a1_2_j:.3f} +/- {e_j[7]:.3f}  (rise)")
print(f"    alpha2 = {a2_2_j:.3f} +/- {e_j[8]:.3f}  (plateau)")
print(f"    alpha3 = {a3_2_j:.3f} +/- {e_j[9]:.3f}  (decay)")
print(f"\nWIND FS (optical+radio):")
print(f"  p          = {p_fs:.3f}")
print(f"  alpha_Wind = {alpha_fs:.3f}")
print(f"  alpha_ISM  = {alpha_ism:.3f}  (same p)")
print(f"  beta       = {beta:.3f}")
print(f"  F0         = {F0_fs*1e6:.2f} uJy  at t={t0_anchor*1440:.1f} min")

# ---------------------------------------------------------------
# Radio extrapolations
# ---------------------------------------------------------------
freq_factor = (nu_6GHz / nu_TESS)**(-beta)
def F_wind_6(t): return F0_fs * freq_factor * (t / t0_anchor)**(-alpha_fs)
def F_ism_6(t):  return F0_fs * freq_factor * (t / t0_anchor)**(-alpha_ism)

mask_6 = nu_radio == nu_6GHz
t_6 = t_radio[mask_6]; F_6 = F_radio[mask_6]; Ferr_6 = Ferr_radio[mask_6]
t_radio_range = np.logspace(np.log10(5), np.log10(35), 300)

col_wind = '#0072B2'
col_ism  = '#E69F00'

# ---------------------------------------------------------------
# Model curves for plotting
# ---------------------------------------------------------------
t_plot    = np.logspace(np.log10(xall[0]*0.9), np.log10(0.105), 1000)
t_fs_full = np.logspace(-3, np.log10(0.105), 500)

pk1_plot  = sbpl(t_plot, F0_j, tb1_1_j, a1_1_j, a2_1_j)
pk2_plot  = dbpl(t_plot, F1_j, tb1_2_j, tb2_2_j, a1_2_j, a2_2_j, a3_2_j)
tot_plot  = np.where(pk1_plot>0, pk1_plot, 0) + np.where(pk2_plot>0, pk2_plot, 0)

master_label = (f'MASTER unfiltered T+{t_master*1440:.1f} min\n'
                '(ref. only; mag system unknown;\n'
                'may include prompt emission)')

def make_tess_panel(ax, xlim, title):
    ax.axhspan(1e-20, 3*sigma1, color='lightgray', alpha=0.5, zorder=0,
               label=r'$<3\sigma$')
    # Data
    ax.errorbar(xall, yall, yerr=yeall, fmt='.', color='steelblue',
                markersize=3, elinewidth=0.6, alpha=0.55, capsize=0,
                zorder=2, label='TESS data')
    # Joint total model
    ax.plot(t_plot, tot_plot, '-', color='black', lw=2, zorder=6,
            label=f'Total model ($\\chi^2_r$={rc_j:.2f})')
    # Individual components
    ax.plot(t_plot, np.where(pk1_plot>0, pk1_plot, np.nan),
            '--', color='goldenrod', lw=1.8, zorder=5,
            label=f'P1: $t_b$={tb1_1_j*1440:.1f} min, '
                  f'$\\alpha_2$={a2_1_j:.2f}')
    ax.plot(t_plot, np.where(pk2_plot>0, pk2_plot, np.nan),
            '--', color='tomato', lw=1.8, zorder=5,
            label=f'P2: $t_{{b1}}$={tb1_2_j*1440:.1f}, '
                  f'$t_{{b2}}$={tb2_2_j*1440:.1f} min, '
                  f'$\\alpha_3$={a3_2_j:.2f}')
    # Wind FS
    ax.plot(t_fs_full, fs_model(F0_fs, p_fs, t_fs_full, nu_TESS),
            'k:', lw=1.5, zorder=4, label='Wind FS (optical+radio)')
    # MASTER
    ax.errorbar(t_master, F_master, yerr=eF_master, fmt='D',
                color='mediumseagreen', markersize=8, elinewidth=1.2,
                capsize=3, zorder=7, markeredgecolor='k', markeredgewidth=0.5,
                label=master_label)
    # Break lines
    ax.axvline(tb1_1_j, color='goldenrod', lw=0.8, ls=':', alpha=0.6)
    ax.axvline(tb1_2_j, color='tomato',    lw=0.8, ls=':', alpha=0.5)
    ax.axvline(tb2_2_j, color='tomato',    lw=0.8, ls=':', alpha=0.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(xlim)
    vis = (xall >= xlim[0]) & (xall <= xlim[1]) & (yall > 0)
    yhi = max(yall[vis].max(), F_master + eF_master) * 3
    ax.set_ylim(yeall[vis].min() * 0.3, yhi)
    ax.set_xlabel('Days post-burst', fontsize=12)
    ax.set_ylabel('Flux density (Jy)', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(True, which='both', alpha=0.2, lw=0.5)
    ax.tick_params(labelsize=10)

# ---------------------------------------------------------------
# PNG 1: TESS full view
# ---------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 5))
make_tess_panel(ax1, (1e-3, 0.1), 'GRB260207A — TESS full view')
plt.tight_layout()
fig1.savefig('GRB260207A_tess_full.png', dpi=180, bbox_inches='tight')
plt.close()
print("\nSaved: GRB260207A_tess_full.png")

# ---------------------------------------------------------------
# PNG 2: TESS peak region
# ---------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 5))
make_tess_panel(ax2, (8e-3, 0.1), 'GRB260207A — TESS peak region')
plt.tight_layout()
fig2.savefig('GRB260207A_tess_peaks.png', dpi=180, bbox_inches='tight')
plt.close()
print("Saved: GRB260207A_tess_peaks.png")

# ---------------------------------------------------------------
# PNG 3: VLA 6 GHz radio panel
# ---------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.errorbar(t_6, F_6, yerr=Ferr_6, fmt='o', color='black', markersize=8,
             elinewidth=1.4, capsize=4, zorder=5, label='VLA 6 GHz data')
ax3.plot(t_radio_range, F_wind_6(t_radio_range), '-', color=col_wind, lw=2.2,
         zorder=4, label=f'Wind ($\\alpha_{{6}}$ = {alpha_fs:.2f})')
ax3.plot(t_radio_range, F_ism_6(t_radio_range), '--', color=col_ism, lw=2.2,
         zorder=4, label=f'ISM ($\\alpha_{{6}}$ = {alpha_ism:.2f})')
ax3.set_xscale('log'); ax3.set_yscale('log')
ax3.set_xlim(5, 35)
ax3.set_ylim(min(F_6.min(), F_wind_6(30)) * 0.3, max(F_6.max(), F_ism_6(5)) * 3)
ax3.set_xlabel('Days post-burst', fontsize=12)
ax3.set_ylabel('Flux density (Jy)', fontsize=12)
ax3.set_title('GRB260207A — VLA 6 GHz: ISM vs Wind', fontsize=12)
ax3.legend(fontsize=10, loc='upper right')
ax3.grid(True, which='both', alpha=0.2, lw=0.5)
ax3.tick_params(labelsize=10)
plt.tight_layout()
fig3.savefig('GRB260207A_radio.png', dpi=180, bbox_inches='tight')
plt.close()
print("Saved: GRB260207A_radio.png")
