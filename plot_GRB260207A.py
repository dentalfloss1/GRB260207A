"""
GRB260207A — TESS + VLA + MASTER light curve analysis
Generates a 3-page PDF:
  Page 1 — TESS full view + peak region (stacked), with MASTER reference point
  Page 2 — VLA 6 GHz with ISM vs Wind extrapolations
  Page 3 — Parameter table

Dependencies: numpy, pandas, matplotlib, scipy, reportlab
Input files:  lc_GRB260207A_cand47734_cleaned
              GRB260207A_-_Sheet1.csv  (VLA radio data)
Output:       GRB260207A_lightcurve.pdf

Notes on MASTER point:
  - Observed at T+3.6 min, unfiltered (broad V-R-I bandpass)
  - Magnitude system not specified in GCN; flux conversion is approximate
  - Could represent prompt emission tail, reverse shock, or early afterglow onset
  - Plotted for reference only — not included in any fit
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import math, io
from datetime import datetime, timezone

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                 Paragraph, Spacer, Image as RLImage, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ---------------------------------------------------------------
# Trigger time  (using manual JD calculation — no astropy needed)
# ---------------------------------------------------------------
def _jd(dt):
    a = math.floor((14 - dt.month) / 12)
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = (dt.day + math.floor((153*m + 2)/5) + 365*y
           + math.floor(y/4) - math.floor(y/100) + math.floor(y/400) - 32045)
    return jdn + (dt.hour-12)/24 + dt.minute/1440 + (dt.second + dt.microsecond/1e6)/86400

GRB_TJD = _jd(datetime(2026, 2, 7, 5, 42, 41, 348000, tzinfo=timezone.utc)) - 2457000

# ---------------------------------------------------------------
# MASTER reference point  (GCN 35684 / Lipunov et al.)
# Obs time: 2026-02-07.24049 UT  =  05:46:18 UT  =>  T+3.6 min
# Magnitude: 17.3 unfiltered (no magnitude system specified)
# Flux conversion: approximate, using AB zeropoint ~3631 Jy
#   with +0.16 mag offset as rough Vega->AB correction for R-band
#   Systematic uncertainty ~30% to account for unknown bandpass
# NOTE: could be prompt emission tail, not Peak 1
# ---------------------------------------------------------------
master_day_frac = 0.24049
master_hr = master_day_frac * 24
master_h  = int(master_hr)
master_m  = int((master_hr - master_h) * 60)
master_s  = ((master_hr - master_h) * 60 - master_m) * 60
master_dt = datetime(2026, 2, 7, master_h, master_m, int(master_s), tzinfo=timezone.utc)
t_master  = _jd(master_dt) - 2457000 - GRB_TJD        # days post-burst
mag_master_approx_AB = 17.3 + 0.16                     # rough Vega->AB
F_master  = 3631.0 * 10**(-0.4 * mag_master_approx_AB) # Jy
eF_master = F_master * 0.30                             # 30% systematic

# ---------------------------------------------------------------
# Load TESS data
# ---------------------------------------------------------------
df = pd.read_csv('lc_GRB260207A_cand47734_cleaned', comment='#', sep=r'\s+',
    names=['BTJD','TJD','cts_per_s','e_cts_per_s','mag','e_mag',
           'bkg','bkg_model','bkg2','e_bkg2'])

x     = df['BTJD'].values - GRB_TJD
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
S1 = 0.1    # Peak 1 smoothness
S2 = 0.02   # Peak 2 smoothness

# ---------------------------------------------------------------
# Working data: unbinned, x < 0.1 d
# ---------------------------------------------------------------
mask_all = (x > 1e-3) & (x < 0.1)
xall = x[mask_all]; yall = flux_jy[mask_all]; yeall = eflux_jy[mask_all]

mask_between = (xall >= 0.028) & (xall <= 0.040)
t_local_min  = xall[mask_between][np.argmin(yall[mask_between])]

mp1 = (xall > 1e-3) & (xall <= t_local_min)
xp1, yp1, yep1 = xall[mp1], yall[mp1], yeall[mp1]

mp2 = (xall > t_local_min) & (xall <= 0.1) & (yall > 0)
xp2, yp2, yep2 = xall[mp2], yall[mp2], yeall[mp2]

# ---------------------------------------------------------------
# Wind FS model — joint optical + radio fit
# ---------------------------------------------------------------
mask_anchor = (x > 0.055) & (x < 0.065) & (flux_jy > 0)
t0_anchor   = np.median(x[mask_anchor])

mask_fs    = (x > 0.055) & (x < 0.1) & (flux_jy > 0)
t_fs, F_fs = x[mask_fs], flux_jy[mask_fs]
t_all_fs   = np.concatenate([t_fs,  t_radio])
F_all_fs   = np.concatenate([F_fs,  F_radio])
nu_all_fs  = np.concatenate([np.full(len(t_fs), nu_TESS), nu_radio])

def fs_model(F0, p, t, nu):
    return F0 * (nu / nu_TESS)**((1 - p) / 2) * (t / t0_anchor)**((1 - 3*p) / 4)

res_fs = least_squares(
    lambda pv: np.log10(fs_model(pv[0], pv[1], t_all_fs, nu_all_fs)) - np.log10(F_all_fs),
    x0=[6e-5, 2.3])
F0_fs, p_fs = res_fs.x
alpha_fs  = (3 * p_fs - 1) / 4
alpha_ism = 3 * (p_fs - 1) / 4
beta      = (p_fs - 1) / 2

# ---------------------------------------------------------------
# Morphological fits
# ---------------------------------------------------------------
def sbpl(t, F0, t_b, a1, a2):
    S = S1
    return F0 * (t/t_b)**(-a1) * (0.5*(1 + (t/t_b)**(1/S)))**(-(a2-a1)*S)

def dbpl(t, F0, tb1, tb2, a1, a2, a3):
    S = S2
    def bkn(t, tb, da): return (0.5*(1 + (t/tb)**(1/S)))**(-da*S)
    return F0 * (t/tb1)**(-a1) * bkn(t, tb1, a2-a1) * bkn(t, tb2, a3-a2)

r1, pcov1 = curve_fit(sbpl, xp1, yp1,
    p0=[5e-5, 0.020, -5.0, 1.8],
    bounds=([0, 0.010, -15.0, -5.0], [1e-3, 0.034, 0.0, 15.0]),
    sigma=yep1, absolute_sigma=True, maxfev=50000)
e1  = np.sqrt(np.diag(pcov1))
rc1 = np.sum(((yp1 - sbpl(xp1, *r1)) / yep1)**2) / (len(xp1) - 4)

r2, pcov2 = curve_fit(dbpl, xp2, yp2,
    p0=[5e-5, 0.047, 0.060, -5.0, -0.1, 1.8],
    bounds=([0, 0.040, 0.053, -12, -2.0, 0.5], [1e-3, 0.056, 0.072, 0.0, 1.0, 4.0]),
    sigma=yep2, absolute_sigma=True, maxfev=100000)
e2  = np.sqrt(np.diag(pcov2))
rc2 = np.sum(((yp2 - dbpl(xp2, *r2)) / yep2)**2) / (len(xp2) - 6)

# ---------------------------------------------------------------
# Print results
# ---------------------------------------------------------------
print(f"Trigger TJD:  {GRB_TJD:.6f}")
print(f"t0_anchor:    {t0_anchor*1440:.2f} min")
print(f"Local min:    {t_local_min*1440:.1f} min")
print(f"\nMASTER: t={t_master*1440:.2f} min, F~{F_master*1e6:.0f} uJy (approx, unfiltered)")
print(f"        Note: mag system unspecified; could be prompt tail")
print(f"\nWind FS:  p={p_fs:.3f}, alpha_Wind={alpha_fs:.3f}, alpha_ISM={alpha_ism:.3f}, "
      f"beta={beta:.3f}, F0={F0_fs*1e6:.2f} uJy")
print(f"\nPeak 1 (SBPL, s={S1}):")
print(f"  t_b    = {r1[1]*1440:.2f} +/- {e1[1]*1440:.2f} min")
print(f"  alpha1 = {r1[2]:.3f} +/- {e1[2]:.3f}  (rise)")
print(f"  alpha2 = {r1[3]:.3f} +/- {e1[3]:.3f}  (decay)")
print(f"  chi2_r = {rc1:.3f}  (N={len(xp1)}, dof={len(xp1)-4})")
print(f"\nPeak 2 (DBPL, s={S2}):")
print(f"  tb1    = {r2[1]*1440:.2f} +/- {e2[1]*1440:.2f} min  (rise->plateau)")
print(f"  tb2    = {r2[2]*1440:.2f} +/- {e2[2]*1440:.2f} min  (plateau->decay)")
print(f"  alpha1 = {r2[3]:.3f} +/- {e2[3]:.3f}  (rise)")
print(f"  alpha2 = {r2[4]:.3f} +/- {e2[4]:.3f}  (plateau)")
print(f"  alpha3 = {r2[5]:.3f} +/- {e2[5]:.3f}  (decay)")
print(f"  chi2_r = {rc2:.3f}  (N={len(xp2)}, dof={len(xp2)-6})")

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
t_f1      = np.logspace(np.log10(xp1[0]*0.9),       np.log10(t_local_min*1.02), 500)
t_f2      = np.logspace(np.log10(t_local_min*0.98),  np.log10(0.105),            500)
t_fs_full = np.logspace(-3,                           np.log10(0.105),            500)

def make_tess_panel(ax, xlim, title):
    ax.axhspan(1e-20, 3*sigma1, color='lightgray', alpha=0.5, zorder=0,
               label=r'$<3\sigma$')
    ax.errorbar(xall, yall, yerr=yeall, fmt='.', color='steelblue',
                markersize=3, elinewidth=0.6, alpha=0.55, capsize=0, zorder=2,
                label='TESS data')
    ax.errorbar(xp1, yp1, yerr=yep1, fmt='o', color='gold', markersize=6,
                elinewidth=0.9, capsize=3, zorder=3, alpha=0.9,
                label='Peak 1 fit pts')
    ax.errorbar(xp2, yp2, yerr=yep2, fmt='s', color='tomato', markersize=5,
                elinewidth=0.9, capsize=3, zorder=3, alpha=0.8,
                label='Peak 2 fit pts')
    ax.plot(t_f1, sbpl(t_f1, *r1), '-', color='goldenrod', lw=2.2, zorder=5,
            label=f'P1 SBPL (s={S1})')
    ax.plot(t_f2, dbpl(t_f2, *r2), '-', color='tomato', lw=2.2, zorder=5,
            label=f'P2 DBPL (s={S2})')
    ax.plot(t_fs_full, fs_model(F0_fs, p_fs, t_fs_full, nu_TESS), 'k--',
            lw=1.8, zorder=6, label='Wind FS (optical+radio)')
    # MASTER reference point — not fit
    # Note: unfiltered bandpass, magnitude system unspecified,
    #       could be prompt emission tail at this early epoch
    ax.errorbar(t_master, F_master, yerr=eF_master, fmt='D',
                color='mediumseagreen', markersize=8, elinewidth=1.2,
                capsize=3, zorder=7, markeredgecolor='k', markeredgewidth=0.5,
                label=f'MASTER unfiltered T+{t_master*1440:.1f} min\n'
                       r'(ref. only; mag system unknown;''\n'
                       r' may include prompt emission)')
    ax.axvline(r1[1],       color='goldenrod', lw=0.9, ls=':', alpha=0.7)
    ax.axvline(r2[1],       color='tomato',    lw=0.9, ls=':', alpha=0.5)
    ax.axvline(r2[2],       color='tomato',    lw=0.9, ls=':', alpha=0.5)
    ax.axvline(t_local_min, color='gray',      lw=1.0, ls='--', alpha=0.5)
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
# Page 1: two TESS panels
# ---------------------------------------------------------------
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 9.5))
fig1.subplots_adjust(hspace=0.42)
make_tess_panel(ax1, (1e-3, 0.1), 'TESS — Full view')
make_tess_panel(ax2, (8e-3, 0.1), 'TESS — Peak region')
fig1.suptitle('GRB260207A', fontsize=15, y=1.005)
buf1 = io.BytesIO()
fig1.savefig(buf1, format='png', dpi=180, bbox_inches='tight')
buf1.seek(0); plt.close()

# ---------------------------------------------------------------
# Page 2: radio panel
# ---------------------------------------------------------------
fig2, ax3 = plt.subplots(1, 1, figsize=(5.5, 4.5))
ax3.errorbar(t_6, F_6, yerr=Ferr_6, fmt='o', color='black', markersize=8,
             elinewidth=1.4, capsize=4, zorder=5, label='VLA 6 GHz data')
ax3.plot(t_radio_range, F_wind_6(t_radio_range), '-', color=col_wind, lw=2.2, zorder=4,
         label=f'Wind ($\\alpha_{{6}}$={alpha_fs:.2f})')
ax3.plot(t_radio_range, F_ism_6(t_radio_range),  '--', color=col_ism, lw=2.2, zorder=4,
         label=f'ISM ($\\alpha_{{6}}$={alpha_ism:.2f})')
ax3.set_xscale('log'); ax3.set_yscale('log')
ax3.set_xlim(5, 35)
ax3.set_ylim(min(F_6.min(), F_wind_6(30)) * 0.3, max(F_6.max(), F_ism_6(5)) * 3)
ax3.set_xlabel('Days post-burst', fontsize=12)
ax3.set_ylabel('Flux density (Jy)', fontsize=12)
ax3.set_title('VLA 6 GHz — ISM vs Wind extrapolation', fontsize=12)
ax3.legend(fontsize=10, loc='upper right')
ax3.grid(True, which='both', alpha=0.2, lw=0.5)
ax3.tick_params(labelsize=10)
fig2.suptitle('GRB260207A', fontsize=15, y=1.01)
buf2 = io.BytesIO()
fig2.savefig(buf2, format='png', dpi=180, bbox_inches='tight')
buf2.seek(0); plt.close()

# ---------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------
pdf_path = 'GRB260207A_lightcurve.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                        leftMargin=1.0*inch, rightMargin=1.0*inch,
                        topMargin=0.5*inch,  bottomMargin=0.5*inch)
styles = getSampleStyleSheet()
story  = []

story.append(RLImage(buf1, width=5.5*inch, height=9.5*inch))
story.append(PageBreak())
story.append(RLImage(buf2, width=5.5*inch, height=4.6*inch))
story.append(PageBreak())

cap = ParagraphStyle('cap', parent=styles['Normal'], fontSize=9, spaceAfter=8, leading=12)
story.append(Paragraph(
    '<b>Table 1.</b> Best-fit parameters for GRB260207A TESS light curve. '
    'Uncertainties are 1<i>&#963;</i>. '
    'Peak&#160;1 uses smoothness s&#160;=&#160;0.1 (sparse data); '
    'Peak&#160;2 uses s&#160;=&#160;0.02 (well-sampled). '
    'Wind FS model fit jointly to TESS (0.055&#8211;0.1&#160;d) and VLA radio. '
    'Radio panel shows ISM and Wind extrapolations (Wong 2011 colorblind-safe palette). '
    'MASTER unfiltered point at T+3.6&#160;min shown for reference only (green diamond); '
    'the GCN does not specify the magnitude system, the bandpass is broad '
    '(V+R+I), and the flux conversion carries&#160;&#8764;30% systematic uncertainty. '
    'At T+3.6&#160;min the emission may include a contribution from the prompt emission tail.',
    cap))
story.append(Spacer(1, 0.15*inch))

hs = ParagraphStyle('th', parent=styles['Normal'], fontSize=9,
                    fontName='Helvetica-Bold', leading=12)
cs = ParagraphStyle('td', parent=styles['Normal'], fontSize=9, leading=12)
H  = lambda s: Paragraph(s, hs)
C  = lambda s: Paragraph(s, cs)
pm = lambda v, e, f='.3f': f'{v:{f}} &#177; {e:{f}}'

table_data = [
    [H('Component'), H('Parameter'), H('Value'), H('Units'), H('Notes')],

    [C('Peak 1 (SBPL)'), C('s'),
     C(f'{S1}'), C('&#8212;'), C('Smoothness parameter')],
    [C(''), C('t<sub>b</sub>'),
     C(pm(r1[1]*1440, e1[1]*1440, '.2f')), C('min'), C('Break time')],
    [C(''), C('&#945;<sub>1</sub>'),
     C(pm(r1[2], e1[2])), C('&#8212;'), C('Rise index')],
    [C(''), C('&#945;<sub>2</sub>'),
     C(pm(r1[3], e1[3])), C('&#8212;'), C('Decay index')],
    [C(''), C('&#967;<sup>2</sup><sub>r</sub>'),
     C(f'{rc1:.2f}'), C('&#8212;'), C(f'N={len(xp1)}, dof={len(xp1)-4}')],

    [C('Peak 2 (DBPL)'), C('s'),
     C(f'{S2}'), C('&#8212;'), C('Smoothness parameter')],
    [C(''), C('t<sub>b1</sub>'),
     C(pm(r2[1]*1440, e2[1]*1440, '.2f')), C('min'), C('Rise &#8594; plateau')],
    [C(''), C('t<sub>b2</sub>'),
     C(pm(r2[2]*1440, e2[2]*1440, '.2f')), C('min'), C('Plateau &#8594; decay')],
    [C(''), C('&#945;<sub>1</sub>'),
     C(pm(r2[3], e2[3])), C('&#8212;'), C('Rise index')],
    [C(''), C('&#945;<sub>2</sub>'),
     C(pm(r2[4], e2[4])), C('&#8212;'), C('Plateau index')],
    [C(''), C('&#945;<sub>3</sub>'),
     C(pm(r2[5], e2[5])), C('&#8212;'), C('Decay index')],
    [C(''), C('&#967;<sup>2</sup><sub>r</sub>'),
     C(f'{rc2:.2f}'), C('&#8212;'), C(f'N={len(xp2)}, dof={len(xp2)-6}')],

    [C('Wind FS\n(optical+radio)'), C('p'),
     C(f'{p_fs:.3f}'), C('&#8212;'), C('Electron spectral index')],
    [C(''), C('&#945;<sub>Wind</sub>'),
     C(f'{alpha_fs:.3f}'), C('&#8212;'), C('Wind temporal decay index')],
    [C(''), C('&#945;<sub>ISM</sub>'),
     C(f'{alpha_ism:.3f}'), C('&#8212;'), C('ISM temporal decay index (same p)')],
    [C(''), C('&#946;'),
     C(f'{beta:.3f}'), C('&#8212;'), C('Spectral index = (p&#8722;1)/2')],
    [C(''), C('F<sub>0</sub>'),
     C(f'{F0_fs*1e6:.2f}'), C('&#956;Jy'),
     C(f'At t={t0_anchor*1440:.1f} min, &#957;<sub>TESS</sub>')],

    [C('MASTER (ref.)'), C('t'),
     C(f'{t_master*1440:.2f}'), C('min'),
     C('Not fit; unfiltered; mag system unknown')],
    [C(''), C('F'),
     C(f'{F_master*1e6:.0f}'), C('&#956;Jy'),
     C('17.3 mag, approx. conversion; may include prompt tail')],
]

col_widths = [1.4*inch, 1.2*inch, 1.5*inch, 0.85*inch, 2.05*inch]
tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
tbl.setStyle(TableStyle([
    ('BACKGROUND',    (0,0),  (-1,-1), colors.white),
    ('TEXTCOLOR',     (0,0),  (-1,-1), colors.black),
    ('FONTNAME',      (0,0),  (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE',      (0,0),  (-1,-1), 9),
    ('GRID',          (0,0),  (-1,-1), 0.4, colors.HexColor('#aaaaaa')),
    ('LINEBELOW',     (0,0),  (-1, 0), 1.2, colors.black),
    ('LINEBELOW',     (0,5),  (-1, 5), 0.8, colors.HexColor('#666666')),
    ('LINEBELOW',     (0,11), (-1,11), 0.8, colors.HexColor('#666666')),
    ('LINEBELOW',     (0,16), (-1,16), 0.8, colors.HexColor('#666666')),
    ('BACKGROUND',    (0,1),  (-1, 5), colors.HexColor('#f5f5f5')),
    ('BACKGROUND',    (0,6),  (-1,11), colors.white),
    ('BACKGROUND',    (0,12), (-1,16), colors.HexColor('#f5f5f5')),
    ('BACKGROUND',    (0,17), (-1,-1), colors.HexColor('#f0fff0')),
    ('TOPPADDING',    (0,0),  (-1,-1), 4),
    ('BOTTOMPADDING', (0,0),  (-1,-1), 4),
    ('LEFTPADDING',   (0,0),  (-1,-1), 6),
    ('RIGHTPADDING',  (0,0),  (-1,-1), 6),
    ('VALIGN',        (0,0),  (-1,-1), 'MIDDLE'),
]))
story.append(tbl)
doc.build(story)
print(f"\nSaved: {pdf_path}")
