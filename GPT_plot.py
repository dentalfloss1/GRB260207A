import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "lc_GRB260207A_cand47734_cleaned"
df = pd.read_csv(file_path, comment='#', sep=r'\s+')

BTJD = df.iloc[:,0]
TJD  = df.iloc[:,1]
cts  = df.iloc[:,2]

t0 = 4078.736307256855

def process_and_bin(time, cts):
    x = time - t0
    
    # Background subtraction
    mask_bg = (x >= -1) & (x <= -0.5)
    median_val = np.median(cts[mask_bg])
    cts_corr = cts - median_val

    def bin_data(x, y, binsize):
        idx = np.argsort(x)
        x_sorted = x.iloc[idx].values
        y_sorted = y.iloc[idx].values
        
        xb, yb = [], []
        for i in range(0, len(x_sorted), binsize):
            xb.append(np.mean(x_sorted[i:i+binsize]))
            yb.append(np.mean(y_sorted[i:i+binsize]))
        return np.array(xb), np.array(yb)

    # Binning regions
    mask1 = (x >= 0.1) & (x < 1)
    mask2 = (x >= 1) & (x <= 10)

    x1, y1 = bin_data(x[mask1], cts_corr[mask1], 4)
    x2, y2 = bin_data(x[mask2], cts_corr[mask2], 32)

    xb = np.concatenate([x1, x2])
    yb = np.concatenate([y1, y2])

    # Keep raw outside
    raw_mask = ~((x >= 0.1) & (x <= 10))
    xr = x[raw_mask].values
    yr = cts_corr[raw_mask].values

    x_all = np.concatenate([xr, xb])
    y_all = np.concatenate([yr, yb])

    # Only require positive flux (symlog allows negative x)
    valid = (y_all > 0)
    x_all = x_all[valid]
    y_all = y_all[valid]

    # Convert
    Tmag = -2.5 * np.log10(y_all) + 20.44
    Jy = 2416 * 10**(-0.4 * Tmag)

    return x_all, Jy

# Process both
x_btjd, Jy_btjd = process_and_bin(BTJD, cts)
x_tjd,  Jy_tjd  = process_and_bin(TJD, cts)

# Time offset
dt = np.median(BTJD - TJD)
print(f"Median (BTJD - TJD) = {dt:.6f} days (~{dt*86400:.1f} seconds)")
dt_all = BTJD - TJD
print(np.min(dt_all)*24*60*60, np.max(dt_all)*24*60*60, np.std(dt_all)*24*60*60)
# Plot
fig, axes = plt.subplots(2, 1, figsize=(6, 8))

linthresh_x = 200/60/60/24

for ax, xvals, Jy, title in zip(
    axes,
    [x_btjd, x_tjd],
    [Jy_btjd, Jy_tjd],
    ["BTJD-based", "TJD-based"]
):
    ax.plot(xvals, Jy, 'o')
    ax.set_xscale('symlog', linthresh=linthresh_x)
    ax.set_yscale('log')
    ax.set_xlim(-1e-1, 10)
    ax.set_ylim(np.min(Jy)*0.8, np.max(Jy)*1.2)
    ax.axvline(0, linestyle='--')
    ax.set_xlabel("days post burst")
    ax.set_ylabel("Flux (Jy)")
    ax.set_title(title)

plt.tight_layout()
plt.show()
