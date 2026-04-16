import matplotlib.pyplot as plt

from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.binning.unbinned import bin_by_time
from gdt.core.plot.lightcurve import Lightcurve

# -------------------------------
# 1. Load multiple detectors
# -------------------------------
import glob
tte_files = glob.glob("glg_tte_n?_bn*_v00.fit")

ttes = [GbmTte.open(f) for f in tte_files]

# -------------------------------
# 2. Merge detectors (boost S/N)
# -------------------------------
tte_merged = GbmTte.merge(ttes)

# -------------------------------
# 3. Select energy range (optional but recommended)
# -------------------------------
# Typical GRB band
tte_merged = tte_merged.slice_energy((50.0, 300.0))

# -------------------------------
# 4. Bin into PHAII (coarse bins)
# -------------------------------
phaii = tte_merged.to_phaii(
    bin_by_time,
    5.0,                      # <-- coarse binning (try 5–20 s)
    time_range=(-50, 2000)    # <-- EXTENDED RANGE (critical!)
)

# -------------------------------
# 5. Convert to light curve
# -------------------------------
lc = phaii.to_lightcurve()

# -------------------------------
# 6. Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

lcplot = Lightcurve(data=lc, ax=ax)

ax.set_title("GBM Light Curve (Merged Detectors, 50–300 keV)")
ax.set_xlabel("Time since trigger (s)")
ax.set_ylabel("Counts/s")

ax.set_xlim(-50, 2000)

plt.tight_layout()
plt.show()
