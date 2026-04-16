import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Load data
df = pd.read_csv("lc_GRB260207A_cand47734_cleaned", sep=r'\s+', comment='#', header=None)
t_btjd = df[0].values
cts = df[2].values

# Assume error column exists
cts_err = df[3].values if df.shape[1] > 3 else np.zeros_like(cts)

# Time conversion
t0 = pd.Timestamp("2026-02-07 05:42:41.348")
GRB_BTJD = t0.to_julian_date() - 2457000
t_days = t_btjd - GRB_BTJD

# Baseline subtraction
mask_bg = (t_days > -1) & (t_days < -(2/24))
baseline = np.median(cts[mask_bg])
cts_corr = cts - baseline

# Keep positive
mask = (cts_corr > 0)
t = t_days[mask]
cts_corr = cts_corr[mask]
cts_err = cts_err[mask]

# Convert to Jy
Tmag = -2.5 * np.log10(cts_corr) + 20.44
F = 2416 * 10**(-0.4 * Tmag)

# Error propagation
Ferr = F * (cts_err / cts_corr)

# --- Split data ---
mask_fit = (t > 0.06) & (t < 0.1)
mask_early = (t > 1e-3) & (t < 0.055)

t_tess = t[mask_fit]
F_tess = F[mask_fit]
Ferr_tess = Ferr[mask_fit]

t_early = t[mask_early]
F_early = F[mask_early]
Ferr_early = Ferr[mask_early]

# Radio data
t_radio = np.array([14, 23, 23])
nu_radio = np.array([6e9, 6e9, 15e9])
F_radio = np.array([24e-6, 18e-6, 17e-6])
Ferr_radio_stat = np.array([4e-6, 3e-6, 3e-6])

# Add 5% systematic
Ferr_radio = np.sqrt(Ferr_radio_stat**2 + (0.05 * F_radio)**2)

# Combine for fitting (UNCHANGED)
t_all = np.concatenate([t_tess, t_radio])
F_all = np.concatenate([F_tess, F_radio])
nu_all = np.concatenate([np.full_like(t_tess, 5e14), nu_radio])

# Model
def model(params, t, nu):
    F0, p = params
    t0 = 0.06
    nu0 = 5e14
    return F0 * (nu/nu0)**((1-p)/2) * (t/t0)**((1-3*p)/4)

def residuals(params):
    return np.log10(model(params, t_all, nu_all)) - np.log10(F_all)

# Fit (UNCHANGED)
res = least_squares(residuals, x0=[6e-5, 2.2])
F0_fit, p_fit = res.x

# Model curves
t_model = np.logspace(-3, 2, 400)

def model_curve(nu):
    return model([F0_fit, p_fit], t_model, nu)

Fopt = model_curve(5e14)
F6 = model_curve(6e9)
F15 = model_curve(15e9)

# Plot
fig, axs = plt.subplots(3, 1, figsize=(6, 12))

# Optical
axs[0].errorbar(t_early, F_early, yerr=Ferr_early,
                fmt='s', color='black', markersize=3, alpha=0.7)
axs[0].errorbar(t_tess, F_tess, yerr=Ferr_tess,
                fmt='.', markersize=3)
axs[0].loglog(t_model, Fopt)
axs[0].set_xlim(1e-3, 0.1)
axs[0].set_ylim(1e-5, 2e-4)
axs[0].grid(True, alpha=0.2)
axs[0].set_title("Optical")

# 6 GHz
axs[1].loglog(t_model, F6)
axs[1].errorbar([14, 23], [24e-6, 18e-6],
                yerr=Ferr_radio[:2], fmt='o')
axs[1].set_xlim(1, 30)
axs[1].set_ylim(1e-5, 2e-4)
axs[1].grid(True, alpha=0.2)
axs[1].set_title("6 GHz")

# 15 GHz
axs[2].loglog(t_model, F15)
axs[2].errorbar([23], [17e-6],
                yerr=[Ferr_radio[2]], fmt='o')
axs[2].set_xlim(1, 30)
axs[2].set_ylim(1e-5, 2e-4)
axs[2].grid(True, alpha=0.2)
axs[2].set_title("15 GHz")

plt.tight_layout()
plt.show()

# Print results
print("F0_fit =", F0_fit)
print("p_fit  =", p_fit)
