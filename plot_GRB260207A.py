import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from astropy.modeling import models
from astropy.time import Time

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_FILE = "lc_GRB260207A_cand47734_cleaned"
T_BURST_BTJD = Time("2026-02-07 05:40:16.947").jd -2457000   # BTJD of T_burst

df = pd.read_csv(
    DATA_FILE,
    comment="#",
    sep=r"\s+",
    names=["BTJD", "TJD", "cts_per_s", "e_cts_per_s",
           "mag", "e_mag", "bkg", "bkg_model", "bkg2", "e_bkg2"],
)

df["t_shifted"] = df["TJD"] - T_BURST_BTJD
df["flux1"] = df["cts_per_s"]

# ── Re-zero using median from 12 h to 2 h pre-burst ──────────────────────────
baseline_mask = (df["t_shifted"] >= -1) & (df["t_shifted"] <= -(2/24))
df["flux1"] -= df.loc[baseline_mask, "flux1"].median()

# ── Trim to window: 1 hour before to 10 days after T_burst ────────────────────
df = df[(df["t_shifted"] >= -1/24) & (df["t_shifted"] <= 10.0)].copy()
print(df)
# ── Binning helper: weighted mean in groups of n ──────────────────────────────
def bin_data(sub, col, n):
    sub = sub.copy().reset_index(drop=True)
    sub["bin"] = sub.index // n
    sub["w"]   = 1.0 / sub["e_cts_per_s"]**2
    grp = sub.groupby("bin")
    t = grp.apply(lambda g: np.average(g["t_shifted"], weights=g["w"])).values
    y = grp.apply(lambda g: np.average(g[col],         weights=g["w"])).values
    e = grp.apply(lambda g: 1.0 / np.sqrt(g["w"].sum())).values
    bkg = grp.apply(lambda g: np.average(g["bkg"],         weights=g["w"])).values
    return t, y, e, bkg

# ── Three segments ────────────────────────────────────────────────────────────
early   = df[df["t_shifted"] <= 0.1]
mid_raw = df[(df["t_shifted"] > 0.1) & (df["t_shifted"] <= 0.2)]
late_raw= df[df["t_shifted"] > 0.2]

early_t = early["t_shifted"].values
early_y = early["flux1"].values
early_e = early["e_cts_per_s"].values
early_bkg = early["bkg"].values

mid_t,  mid_y,  mid_e, mid_bkg = bin_data(mid_raw,  "flux1", 4)
late_t, late_y, late_e, late_bkg = bin_data(late_raw, "flux1", 128)
# ── Power-law fit: data between 0.05 and 2 days ─────────────────────────────
p = 2.2
def ismdecay(t, F0):
    alpha=3*(1-p)/4
    return F0 * t**(alpha) 

def winddecay(t, F0):
    alpha=(1-3*p)/4
    return F0 * t**(alpha) 
# Combine all segments then filter to fit window
all_t = np.concatenate([early_t, mid_t, late_t])
all_y = np.concatenate([early_y, mid_y, late_y])
all_e = np.concatenate([early_e, mid_e, late_e])

fit_mask = (all_t >= 0.05) & (all_t <= 4.0)
fit_t = all_t[fit_mask]
fit_y = all_y[fit_mask]
fit_e = all_e[fit_mask]

try:
    poptISM, pcovISM = curve_fit(
        ismdecay, fit_t, fit_y,
        p0=[1.0], sigma=fit_e, absolute_sigma=True,
        bounds=([0], [np.inf]),
        maxfev=10000,
    )
    perr = np.sqrt(np.diag(pcovISM))
    fit_okISM = True
    print(f"Power-law fit:  F0={poptISM[0]:.4f}")
except RuntimeError as exc:
    print(f"Fit did not converge: {exc}")
    fit_okISM = False
try:
    poptwind, pcovwind = curve_fit(
        winddecay, fit_t, fit_y,
        p0=[1.0], sigma=fit_e, absolute_sigma=True,
        bounds=([0], [np.inf]),
        maxfev=10000,
    )
    perrwind = np.sqrt(np.diag(pcovwind))
    fit_okwind = True
    print(f"Power-law fit:  F0={poptwind[0]:.4f}")
except RuntimeError as exc:
    print(f"Fit did not converge: {exc}")
    fit_okwind = False
# ── REVERSE SBPL fit: data between -0.002 and 0.02 days ─────────────────────────────
def dsbpl(x, A, xb1, xb2,alpha1,alpha2,alpha3):
    """
    Multiplicative smoothly broken power law
    """
    # Thick shell, slow cooling  (maybe?)
    # alpha1=-1/3
    # alpha2=(p-1)/2
    # alpha3=p/2
    # Thin shell, crossing (doesn't work here)
    # alpha1=-2
    # alpha2=-1/3
    # alpha3=((p-1)/2)
    # Fast cooling
    alpha1=-alpha1
    alpha2=-alpha2
    alpha3=-alpha3
    s=0.02
    s1=s
    s2=s
    x = np.asarray(x)

    term1 = (x / xb1) ** (-alpha1)

    smooth1 = (1 + (x / xb1) ** (1.0 / s1)) ** ((alpha1 - alpha2) * s1)
    smooth2 = (1 + (x / xb2) ** (1.0 / s2)) ** ((alpha2 - alpha3) * s2)

    return A * term1 * smooth1 * smooth2

# # Combine all segments then filter to fit window
all_t = np.concatenate([early_t, mid_t, late_t])
all_y = np.concatenate([early_y, mid_y, late_y])
all_e = np.concatenate([early_e, mid_e, late_e])

fit_mask = (all_t >= -0.002) & (all_t <= 0.02)
fit_t = all_t[fit_mask]
fit_y = all_y[fit_mask]
fit_e = all_e[fit_mask]

try:
    poptrev, pcovrev = curve_fit(
        dsbpl, fit_t, fit_y,
        p0=[5.0,7e-3,0.012,0.33,-0.5,-1.1], sigma=fit_e, absolute_sigma=True,
        bounds=([0,6e-3,1e-2,0.1,-1,-2], [100,9e-3,0.015,0.5,0,-0.1]),
        maxfev=10000,
    )
    perr = np.sqrt(np.diag(pcovrev))
    fit_okrev = True
    print(f"Power-law fit:  F0={poptrev[0]:.4f} xb1={poptrev[1]:.4f} xb2={poptrev[2]:.4f} a1={poptrev[3]:.4f} a2={poptrev[4]:.4f} a3={poptrev[5]:.4f}")
except RuntimeError as exc:
    print(f"Fit did not converge: {exc}")
    fit_okrev = False
# ── FS SBPL fit: data between -0.002 and 0.02 days ─────────────────────────────

if fit_okrev:
    def FS(x, A, xb1, xb2, alpha1, alpha2):
        """
        Multiplicative smoothly broken power law
        """
        alpha1=-alpha1
        alpha2=-alpha2
        rev = dsbpl(x, *poptrev)
        alpha3=-(1-3*p)/4
        s=0.02
        s1=s
        s2=s
        x = np.asarray(x)
    
        term1 = (x / xb1) ** (-alpha1)
    
        smooth1 = (1 + (x / xb1) ** (1.0 / s1)) ** ((alpha1 - alpha2) * s1)
        smooth2 = (1 + (x / xb2) ** (1.0 / s2)) ** ((alpha2 - alpha3) * s2)
    
        return (A * term1 * smooth1 * smooth2) + rev
    
    # # Combine all segments then filter to fit window
    all_t = np.concatenate([early_t, mid_t, late_t])
    all_y = np.concatenate([early_y, mid_y, late_y])
    all_e = np.concatenate([early_e, mid_e, late_e])
    
    fit_mask = (all_t >= 0.02) & (all_t <= 2)
    fit_t = all_t[fit_mask]
    fit_y = all_y[fit_mask]
    fit_e = all_e[fit_mask]
    
    try:
        poptfwd, pcovfwd = curve_fit(
            FS, fit_t, fit_y,
            p0=[5.0,0.0325,0.049,1,0], sigma=fit_e, absolute_sigma=True,
            bounds=([0,0.0,0.042,0.1,-1], [100,0.035,0.051,10,1]),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcovfwd))
        fit_okfwd = True
        print(f"Power-law fit:  F0={poptfwd[0]:.4f} xb1={poptfwd[1]:.4f} xb2={poptfwd[2]:.4f} a1={poptfwd[3]:.4f} a2={poptfwd[4]:.4f} a3={(1-3*p)/4} (not fit)")
    except RuntimeError as exc:
        print(f"Fit did not converge: {exc}")
        fit_okfwd = False
else:
    fit_okfwd = False


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

# Connecting line through all data points
t_all = np.concatenate([early_t, mid_t, late_t])
y_all = np.concatenate([early_y, mid_y, late_y])
ax.plot(t_all, y_all, color="black", lw=0.6, alpha=0.4, zorder=2)

# Scatter + errorbars — semi-transparent
for t, y, e in [(early_t, early_y, early_e),
                (mid_t,   mid_y,   mid_e),
                (late_t,  late_y,  late_e)]:
    ax.errorbar(t, y, yerr=e,
                fmt="o", ms=4, color="black", alpha=0.4,
                ecolor="black", elinewidth=0.8, capsize=2, capthick=0.8,
                linewidth=0, zorder=3)

# Power-law: plotted from 0.01 to 3 days
if fit_okISM:
    t_pl = np.logspace(np.log10(0.01), np.log10(3.0), 2000)
    ax.plot(t_pl, ismdecay(t_pl, *poptISM), color="tomato", lw=2, zorder=5,ls=':',
            label=rf"ISM")
if fit_okwind:
    t_pl = np.logspace(np.log10(0.01), np.log10(3.0), 2000)
    ax.plot(t_pl, winddecay(t_pl, *poptwind), color="navy", lw=2, zorder=5,ls='-.',
            label=rf"Wind")

if fit_okrev:
    t_pl = np.logspace(np.log10(1e-6), np.log10(3e-2), 2000)
    ax.plot(t_pl, dsbpl(t_pl, *poptrev), color="black", lw=2, zorder=5,ls='-',
            label=rf"SBPL")
if fit_okfwd:
    t_pl = np.logspace(np.log10(1e-6), np.log10(10), 2000)
    ax.plot(t_pl, FS(t_pl, *poptfwd), color="black", lw=2, zorder=5,ls='-',
            label=rf"FS+REV")
# T_burst line
ax.axvline(0, color="black",alpha=0.5, lw=1.5, ls="-", zorder=4, label=r"$T_{\rm burst}$")

ax.set_xlabel(r"$t - T_{\rm burst}$  (days)", fontsize=13)
ax.set_ylabel(r"cts s$^{-1}$ (re-zeroed)", fontsize=13)
ax.set_title("GRB 260207A  —  TESS light curve", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, which="both", ls=":", alpha=0.4)

MIN_T = 0.01
# ax.set_xscale("symlog", linthresh=MIN_T)
ax.set_xscale("log")# , linthresh=MIN_T)
ax.set_yscale("log")
ax.set_xlim(0, 10.0)
ax.set_ylim(1e-2,7)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
plt.savefig("GRB260207A_lightcurve.png", dpi=150, bbox_inches="tight")
print("Saved GRB260207A_lightcurve.png")
plt.show()
# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

# Connecting line through all data points
t_all = np.concatenate([early_t, mid_t, late_t])
y_all = np.concatenate([early_bkg, mid_bkg, late_bkg])
ax.plot(t_all, y_all, color="black", lw=0.6, alpha=0.5, zorder=2,marker='o')
t_all = np.concatenate([early_t, mid_t, late_t])
y_all = np.concatenate([early_y, mid_y, late_y])
ax.plot(t_all, y_all, color="red", lw=0.6, alpha=0.5, zorder=2,marker='o')

# T_burst line
ax.axvline(0, color="black",alpha=0.5, lw=1.5, ls="-", zorder=4, label=r"$T_{\rm burst}$")

ax.set_xlabel(r"$t - T_{\rm burst}$  (days)", fontsize=13)
ax.set_ylabel(r"BKG cts s$^{-1}$ (re-zeroed)", fontsize=13)
ax.set_title("GRB 260207A  —  TESS Background", fontsize=14, fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.4)

MIN_T = 0.01
# ax.set_xscale("symlog", linthresh=MIN_T)
ax.set_xscale("log")# , linthresh=MIN_T)
ax.set_yscale("log")
ax.set_xlim(0, 10.0)
ax.set_ylim(1e-2,7)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
plt.savefig("bkg_lightcurve.png", dpi=150, bbox_inches="tight")
print("Saved bkg_lightcurve.png")
plt.close()

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Use ONLY t > 0 for log x-axis
mask = df["t_shifted"] > 0
dfp = df[mask].copy()

# ── Panel 1: Flux (linear y because of negatives!) ──
axes[0].scatter(dfp["t_shifted"], dfp["flux1"], marker='o', alpha=0.6)
axes[0].set_ylabel("Flux (cts/s)")
axes[0].set_title("Flux (re-zeroed)")
axes[0].set_xscale("log")
axes[0].set_yscale("symlog", linthresh=1e-3)
axes[0].grid(True, which="both", ls=":", alpha=0.4)

# ── Panel 2: BKG ──
axes[1].scatter(dfp["t_shifted"], dfp["bkg"], marker='o', color='red', alpha=0.6)
axes[1].set_ylabel("BKG")
axes[1].set_title("Background (bkg)")
axes[1].set_xscale("log")
axes[1].set_yscale("symlog", linthresh=1e-3)
axes[1].set_ylim(-1e-1,10)
axes[1].grid(True, which="both", ls=":", alpha=0.4)

# ── Panel 3: flux over bkg ──
axes[2].scatter(dfp["t_shifted"], dfp["flux1"]/dfp["bkg"], marker='o', color='purple', alpha=0.6)
axes[2].set_ylabel("Flux/BKG")
axes[2].set_title("Flux / Background")
axes[2].set_xscale("log")
axes[2].set_yscale("symlog", linthresh=MIN_T)
axes[2].grid(True, which="both", ls=":", alpha=0.4)

axes[2].set_xlabel(r"$t - T_{\rm burst}$ (days)")

plt.tight_layout()
# plt.show()
plt.savefig("bkgratio.png", dpi=150, bbox_inches="tight")
plt.close()
