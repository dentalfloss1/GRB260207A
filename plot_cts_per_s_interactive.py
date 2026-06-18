"""Interactive counts/sec light-curve plot for GRB260207A."""

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time


trigger_tjd = Time("2026-02-07 05:40:16.947").jd - 2457000

rawdata = np.loadtxt("lc_GRB260207A_cand41148_geo")
dt_burst = rawdata[:, 0] - trigger_tjd
cts_per_s = -rawdata[:, 1] / (200 * 0.8 * 0.99)
ects_per_s = rawdata[:, 2] / (200 * 0.8 * 0.99)

mask = (dt_burst >= -1.0) & (dt_burst <= 20.0)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.errorbar(
    dt_burst[mask],
    cts_per_s[mask],
    yerr=ects_per_s[mask],
    fmt=".",
    color="black",
    markersize=3,
    elinewidth=0.4,
    alpha=0.75,
    capsize=0,
)

ax.axvline(0, color="gray", ls="--", lw=1.0, alpha=0.8)
ax.set_xlim(-1.0, 20.0)
ax.set_xlabel(r"$\Delta t_{\rm burst}$ (days)")
ax.set_ylabel(r"Counts s$^{-1}$")
ax.set_title("GRB 260207A TESS light curve")
ax.grid(True, alpha=0.25, lw=0.6)
fig.tight_layout()
plt.show()
