import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from scipy.optimize import curve_fit
binstart = 0.055
S1 = 0.1    # Peak 1 smoothness (sparse data)
S2 = 0.02   # Peak 2 smoothness (well-sampled)
# --- Trigger time (days since JD 2457000)
trigger = Time("2026-02-07 05:40:16.947").jd - 2457000

# --- Load data
rawdata = np.genfromtxt("lc_GRB260207A_cand41148")

# --- Time since burst (days)
x = rawdata[:, 0] - trigger

# --- Convert to counts/sec (correcting inverted counts)
y = -rawdata[:, 1] / (200 * 0.8 * 0.99)
yerr = rawdata[:, 2] / (200 * 0.8 * 0.99)

# --- Restrict to 1e-3 to 10 days
mask_time = (x >= 1e-3) & (x <= 10)
x = x[mask_time]
y = y[mask_time]
yerr = yerr[mask_time]

# --- Restrict ONLY in time (not flux sign!)
mask_bin = (x >= binstart) & (x <= 10)

x_bin = x[mask_bin]
flux_bin = y[mask_bin]
flux_err_bin = yerr[mask_bin]

# --- Define log-spaced bins
nbins = 15
bins = np.logspace(np.log10(binstart), np.log10(10), nbins + 1)

x_binned = []
flux_binned = []
flux_err_binned = []

for i in range(nbins):
    in_bin = (x_bin >= bins[i]) & (x_bin < bins[i+1])
    
    if np.sum(in_bin) > 0:
        xb = x_bin[in_bin]
        fb = flux_bin[in_bin]
        feb = flux_err_bin[in_bin]
        
        w = 1.0 / (feb**2)
        
        f_avg = np.sum(w * fb) / np.sum(w)
        f_err = np.sqrt(1.0 / np.sum(w))
        
        x_avg = np.exp(np.mean(np.log(xb)))
        
        x_binned.append(x_avg)
        flux_binned.append(f_avg)
        flux_err_binned.append(f_err)

x_binned = np.array(x_binned)
flux_binned = np.array(flux_binned)
flux_err_binned = np.array(flux_err_binned)

# Non binned data
mask_early = x < binstart

x_early = x[mask_early]
flux_early = y[mask_early]
flux_err_early = yerr[mask_early]


# combine binned with non-binned in the appropriate time ranges
x_combined = np.concatenate([x_early, x_binned])
flux_combined = np.concatenate([flux_early, flux_binned])
flux_err_combined = np.concatenate([flux_err_early, flux_err_binned])

# Ensure proper time sorting
sort_idx = np.argsort(x_combined)

x_combined = x_combined[sort_idx]
flux_combined = flux_combined[sort_idx]
flux_err_combined = flux_err_combined[sort_idx]

# --- Remove any non-positive flux values (required for log scale)
mask_flux = (flux_combined > 0)
x = x_combined[mask_flux]
y = flux_combined[mask_flux]
yerr = flux_err_combined[mask_flux]

# --- Convert to magnitudes
mag = -2.5 * np.log10(y) + 20.44
mag_err = (2.5 / np.log(10)) * (yerr / y)

# --- Convert to flux (Jy)
flux = 2416 * 10**(-0.4 * mag)

# --- Propagate error (fractional error preserved)
flux_err = flux * (yerr / y)



# -------------------------------
# --- Model definitions
# -------------------------------

def sbpl(t, F0, t_b, a1, a2):
    """Single smoothly broken power law, s=S1 (Peak 1)."""
    S = S1
    return F0 * (t/t_b)**(-a1) * (0.5*(1 + (t/t_b)**(1/S)))**(-(a2-a1)*S)

def dsbpl(t, F0, tb1, tb2, a1, a2, a3):
    """Double smoothly broken power law, s=S2 (Peak 2)."""
    S = S2
    def bkn(t, tb, da): return (0.5*(1 + (t/tb)**(1/S)))**(-da*S)
    return F0 * (t/tb1)**(-a1) * bkn(t, tb1, a2-a1) * bkn(t, tb2, a3-a2)

def total_model(t,
                F1, tb1, a1_1, a2_1,
                F2, tb2_1, tb2_2, a1_2, a2_2, a3_2):
    
    comp1 = sbpl(t, F1, tb1, a1_1, a2_1)
    
    comp2 = dsbpl(t, F2, tb2_1, tb2_2,
                  a1_2, a2_2, a3_2,
                  )
    
    return comp1 + comp2


# -------------------------------
# --- Initial guesses (your values)
# -------------------------------

p0 = [
    5e-4, 1e-2, -1, 2,          # SBPL
    5e-4, 0.04, 0.055, -3.2, 2, 1  # DSBPL
]

# -------------------------------
# --- Bounds (your values)
# -------------------------------

bounds_lo = [
    1e-6,    7e-3,   -3.0, 0.1,
    1e-6,    0.035,  0.045, -12,  -3.0,  0.1
]

bounds_hi = [
    1e-3, 2e-2,   0.0,  5.0,
    1e-3, 0.045,  0.072,  0.0,  3.0,  4.0
]

# -------------------------------
# --- Perform fit
# -------------------------------

popt, pcov = curve_fit(
    total_model,
    x,
    flux,
    sigma=flux_err,
    p0=p0,
    bounds=(bounds_lo, bounds_hi),
    absolute_sigma=True,
    maxfev=20000
)

# -------------------------------
# --- Generate smooth model curve
# -------------------------------

t_fit = np.logspace(np.log10(min(x)),
                    np.log10(max(x)), 500)

f_fit = total_model(t_fit, *popt)

# Individual components (VERY useful)
comp1 = sbpl(t_fit, *popt[:4])
comp2 = dsbpl(t_fit, *popt[4:])




# --- Plot (log-log)
plt.figure(figsize=(7, 5))

plt.errorbar(
    x, flux,
    yerr=flux_err,
    fmt='o',
    markersize=4,
    capsize=2,
    label='GRB afterglow'
)
plt.plot(t_fit, f_fit, label='Total Fit', linewidth=2)

plt.plot(t_fit, comp1, '--', label='SBPL (Peak 1)')
plt.plot(t_fit, comp2, ':', label='DSBPL (Peak 2)')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Time since burst (days)')
plt.ylabel('Flux (Jy)')
plt.title('GRB Afterglow Light Curve')

# --- Grid for log scales
plt.grid(True, which="both", linestyle="--", alpha=0.5)
ax = plt.gca()
ax.set_ylim(1e-6,1e-2)
plt.legend()
plt.tight_layout()
plt.show()
