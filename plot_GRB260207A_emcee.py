"""
GRB260207A — emcee fit of P1-SBPL + P2-DSBPL model.

Two fits, same model (P1: SBPL; P2: double SBPL), window-integrated forward model.
  Fit A: t in [5 min, 1 day]   (excludes straddle and very early points)
  Fit B: t in (0, 0.1 d]       (includes straddle, excludes noisy late times)

Priors:
  log_uniform on F0_1, F0_2           (wide range across decades)
  log_uniform on tb_1, tb2a_2, tb2b_2 (covers all reasonable break times)
  uniform on alpha indices
  ordering: tb2a_2 < tb2b_2 enforced

Sampled parameters:
  theta = [log10(F0_1), log10(tb_1), a1_1, a2_1,
           log10(F0_2), log10(tb2a_2), log10(tb2b_2), a1_2, a2_2, a3_2]
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import curve_fit
from astropy.time import Time

trigger_tjd = Time("2026-02-07 05:40:16.947").jd - 2457000

master_obs = Time("2026-02-07 05:46:04.3")
t_master   = master_obs.jd - 2457000 - trigger_tjd
F_master   = 3276.0 * 10**(-0.4 * 17.3)
eF_master  = F_master * 0.30

rawdata  = np.loadtxt('lc_GRB260207A_cand41148_geo')
x_all    = rawdata[:, 0] - trigger_tjd
y_all    = -rawdata[:, 1] / (200 * 0.8 * 0.99)
yerr_all =  rawdata[:, 2] / (200 * 0.8 * 0.99)
zp        = 2416 * 10**(-0.4 * 20.44)
flux_all  = y_all    * zp
eflux_all = yerr_all * zp

CADENCE = 200 / 86400
HALF_C  = CADENCE / 2

mask_plot = (x_all > -HALF_C) & (x_all <= 1.5)
x_plot  = x_all[mask_plot]
y_plot  = flux_all[mask_plot]
ye_plot = eflux_all[mask_plot]

# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
S = 0.02

def sbpl(t, F0, tb, a1, a2, S=S):
    return F0 * (t/tb)**(-a1) * (0.5*(1 + (t/tb)**(1/S)))**(-(a2-a1)*S)

def dsbpl(t, F0, tb1, tb2, a1, a2, a3, S=S):
    """Double SBPL: breaks at tb1 (a1->a2) and tb2 (a2->a3)."""
    f1 = (0.5 * (1 + (t/tb1)**(1/S)))**(-(a2-a1)*S)
    f2 = (0.5 * (1 + (t/tb2)**(1/S)))**(-(a3-a2)*S)
    return F0 * (t/tb1)**(-a1) * f1 * f2

def model_2sbpl(t, F0_1, tb_1, a1_1, a2_1,
                   F0_2, tb2a_2, tb2b_2, a1_2, a2_2, a3_2):
    p1 = sbpl(t, F0_1, tb_1, a1_1, a2_1)
    p2 = dsbpl(t, F0_2, tb2a_2, tb2b_2, a1_2, a2_2, a3_2)
    return np.where(p1 > 0, p1, 0) + np.where(p2 > 0, p2, 0)

HALF_WIN_DAY = CADENCE / 2.0
FULL_WIN_DAY = CADENCE
WINDOW_THRESH_DAY = 5.0 / 1440  # 5 min

def model_2sbpl_windowed(t_arr, *params):
    """Vectorized: late points evaluated instantaneously; early points (t<5 min)
    integrated over their 200-s window with 11 sub-samples."""
    early_mask = t_arr <= WINDOW_THRESH_DAY
    out = np.empty_like(t_arr)
    # Late points: vectorized instantaneous eval
    if (~early_mask).any():
        out[~early_mask] = model_2sbpl(t_arr[~early_mask], *params)
    # Early points: per-point trapezoid (typically only 1-3 points)
    if early_mask.any():
        for i in np.where(early_mask)[0]:
            tmid = t_arr[i]
            ts = np.linspace(tmid - HALF_WIN_DAY, tmid + HALF_WIN_DAY, 11)
            valid = ts > 0
            if not valid.any():
                out[i] = 0.0
                continue
            f = np.zeros_like(ts)
            f[valid] = model_2sbpl(ts[valid], *params)
            out[i] = np.trapezoid(f, ts) / FULL_WIN_DAY
    return out

# ---------------------------------------------------------------
# Priors
# theta = [logF1, logTb1, a1_1, a2_1,
#          logF2, logTb2a, logTb2b, a1_2, a2_2, a3_2]
# ---------------------------------------------------------------
# Log-uniform priors:
#   F0 in [1e-6, 5e-3] Jy   -> log10 in [-6, -2.3]
#   tb in [1e-4, 1e-1] d    -> log10 in [-4, -1]   (0.14 min to 144 min)
# Uniform priors:
#   a1 (rise) in [-50, 0]
#   a2_1, a3_2 (decay) in [0.2, 5]
#   a2_2 (P2 middle, approx flat) in [-2, 3]
# Ordering: tb2a_2 < tb2b_2 enforced in log_prior

PRIOR_RANGES = {
    'logF1':   (-6.0, -2.3),
    'logTb1':  (-4.0, -1.0),
    'a1_1':    (-50.0, 0.0),
    'a2_1':    (0.2, 5.0),
    'logF2':   (-6.0, -2.3),
    'logTb2a': (-4.0, -1.0),
    'logTb2b': (-4.0, -1.0),
    'a1_2':    (-50.0, 0.0),
    'a2_2':    (-2.0, 3.0),
    'a3_2':    (0.2, 5.0),
}
PARAM_NAMES = list(PRIOR_RANGES.keys())

def log_prior(theta):
    for v, (lo, hi) in zip(theta, PRIOR_RANGES.values()):
        if not (lo <= v <= hi):
            return -np.inf
    # Require first P2 break before second
    if theta[6] <= theta[5]:
        return -np.inf
    return 0.0

def theta_to_params(theta):
    """Convert sampled theta (log10 for F and tb) to physical parameters."""
    logF1, logTb1, a1_1, a2_1, logF2, logTb2a, logTb2b, a1_2, a2_2, a3_2 = theta
    return (10**logF1, 10**logTb1, a1_1, a2_1,
            10**logF2, 10**logTb2a, 10**logTb2b, a1_2, a2_2, a3_2)

def log_likelihood(theta, x, y, yerr):
    params = theta_to_params(theta)
    try:
        ymod = model_2sbpl_windowed(x, *params)
    except Exception:
        return -np.inf
    if not np.all(np.isfinite(ymod)):
        return -np.inf
    return -0.5 * np.sum(((y - ymod) / yerr)**2)

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, x, y, yerr)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# ---------------------------------------------------------------
# Run emcee for one dataset
# ---------------------------------------------------------------
def run_emcee(x, y, yerr, theta0, label, nwalkers=32, nburn=500, nprod=10_000):
    ndim = len(theta0)
    # Initialize walkers in tight ball around theta0
    pos = theta0 + 1e-3 * np.random.randn(nwalkers, ndim)
    # Make sure walkers start in prior
    for i in range(nwalkers):
        for j, (lo, hi) in enumerate(PRIOR_RANGES.values()):
            pos[i, j] = np.clip(pos[i, j], lo + 1e-4, hi - 1e-4)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                     args=(x, y, yerr))
    print(f"[{label}] burn-in ({nburn} steps, {nwalkers} walkers)...")
    state = sampler.run_mcmc(pos, nburn, progress=False)
    sampler.reset()
    print(f"[{label}] production ({nprod} steps)...")
    sampler.run_mcmc(state, nprod, progress=False)

    # Acceptance fraction & autocorrelation
    af = np.mean(sampler.acceptance_fraction)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        tau_max = np.max(tau)
    except Exception:
        tau_max = None

    print(f"[{label}] mean acceptance: {af:.3f},  max tau: {tau_max}")

    return sampler

# ---------------------------------------------------------------
# Set up datasets
# ---------------------------------------------------------------
# Fit A: 0 min to 1 day
mask_A = ((x_plot >= 0/1440) & (x_plot <= 1.0) & (ye_plot > 0))
xA = x_plot[mask_A]; yA = y_plot[mask_A]; yeA = ye_plot[mask_A]

# Fit B: 0 to 0.1 d
mask_B = ((x_plot > 0) & (x_plot <= 0.1) & (ye_plot > 0))
xB = x_plot[mask_B]; yB = y_plot[mask_B]; yeB = ye_plot[mask_B]

print(f"Fit A: N = {len(xA)},  t in [{xA.min()*1440:.2f}, {xA.max():.3f} d]")
print(f"Fit B: N = {len(xB)},  t in [{xB.min()*1440:.2f}, {xB.max():.3f} d]")

# Initial point informed by previous fits:
#   Peak 1: F0 ~ 700 uJy, tb ~ 10 min, a1 ~ -2, a2 ~ 1
#   Peak 2: F0 ~ 300 uJy, tb2a ~ 45 min, tb2b ~ 75 min, a1 ~ -10, a2 ~ 0, a3 ~ 2
theta0 = [
    np.log10(7e-4),  np.log10(10/1440), -2.0, 1.0,
    np.log10(3e-4),  np.log10(45/1440), np.log10(75/1440), -10.0, 0.0, 2.0,
]

np.random.seed(42)

samplerA = run_emcee(xA, yA, yeA, theta0, 'Fit A')
samplerB = run_emcee(xB, yB, yeB, theta0, 'Fit B')

# ---------------------------------------------------------------
# Extract results
# ---------------------------------------------------------------
def summarize(sampler, label):
    samples = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)
    # Best-fit (max log-prob)
    best_idx = np.argmax(log_prob)
    theta_best = samples[best_idx]
    params_best = theta_to_params(theta_best)
    # Quantiles per parameter (in physical space, after converting where needed)
    quantiles = []
    for j, name in enumerate(PARAM_NAMES):
        if name.startswith('logF') or name.startswith('logTb'):
            phys = 10**samples[:, j]
        else:
            phys = samples[:, j]
        q16, q50, q84 = np.percentile(phys, [16, 50, 84])
        quantiles.append((name, q16, q50, q84))
    return theta_best, params_best, quantiles, samples, log_prob

bestA, paramsA, qA, samplesA, logprobA = summarize(samplerA, 'Fit A')
bestB, paramsB, qB, samplesB, logprobB = summarize(samplerB, 'Fit B')

# chi^2
chi2A = -2 * np.max(logprobA)
chi2_rA = chi2A / (len(xA) - 10)
chi2B = -2 * np.max(logprobB)
chi2_rB = chi2B / (len(xB) - 10)

# Display
print(f"\n=== FIT A SUMMARY (t >= 0 min, t <= 1 d, N={len(xA)}) ===")
print(f"  Best chi2_r = {chi2_rA:.3f}")
labels_phys = ['F0_1 (uJy)', 'tb_1 (min)', 'a1_1', 'a2_1',
               'F0_2 (uJy)', 'tb2a_2 (min)', 'tb2b_2 (min)', 'a1_2', 'a2_2', 'a3_2']
scale = [1e6, 1440, 1, 1, 1e6, 1440, 1440, 1, 1, 1]
for (name, q16, q50, q84), lbl, sc, p in zip(qA, labels_phys, scale, paramsA):
    print(f"  {lbl:>14s}:  median={q50*sc:8.2f}  +{(q84-q50)*sc:6.2f} -{(q50-q16)*sc:6.2f}  best={p*sc:8.2f}")

print(f"\n=== FIT B SUMMARY (t > 0, t <= 0.1 d, N={len(xB)}) ===")
print(f"  Best chi2_r = {chi2_rB:.3f}")
for (name, q16, q50, q84), lbl, sc, p in zip(qB, labels_phys, scale, paramsB):
    print(f"  {lbl:>14s}:  median={q50*sc:8.2f}  +{(q84-q50)*sc:6.2f} -{(q50-q16)*sc:6.2f}  best={p*sc:8.2f}")

# ---------------------------------------------------------------
# Plot model + data with posterior bands
# ---------------------------------------------------------------
def plot_fit(ax, sampler, mask_used, label, chi2_r, n_draws=200):
    samples = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)
    best_idx = np.argmax(log_prob)
    params_best = theta_to_params(samples[best_idx])

    pos = y_plot > 0
    fitted_pos = mask_used & pos
    excluded_pos = ~mask_used & pos & (x_plot < 1.5)

    if (~pos).any():
        ax.errorbar(x_plot[~pos], np.abs(y_plot[~pos]), yerr=ye_plot[~pos],
                    fmt='.', color='lightgray', markersize=2, elinewidth=0.4,
                    alpha=0.3, capsize=0, zorder=1)
    if excluded_pos.any():
        ax.errorbar(x_plot[excluded_pos], y_plot[excluded_pos],
                    yerr=ye_plot[excluded_pos], fmt='.', color='gray',
                    markersize=2, elinewidth=0.3, alpha=0.4, capsize=0,
                    zorder=1.5, label='Excluded')
    ax.errorbar(x_plot[fitted_pos], y_plot[fitted_pos],
                yerr=ye_plot[fitted_pos], fmt='.', color='black',
                markersize=3, elinewidth=0.4, alpha=0.8, capsize=0,
                zorder=2, label='Fitted')
    ax.errorbar(t_master, F_master, yerr=eF_master, fmt='D',
                color='mediumseagreen', markersize=8, elinewidth=1.4,
                capsize=4, markeredgecolor='k', markeredgewidth=0.6,
                label=f'MASTER T+{t_master*1440:.1f} min')

    # Posterior draws
    t_model = np.logspace(np.log10(2e-4), np.log10(1.5), 1500)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(samples), n_draws, replace=False)
    for i in idx:
        params = theta_to_params(samples[i])
        ymod = model_2sbpl(t_model, *params)
        ax.plot(t_model, ymod, '-', color='crimson', lw=0.3, alpha=0.04, zorder=4)

    # Best fit
    ymod_best = model_2sbpl(t_model, *params_best)
    p1_best = sbpl(t_model, params_best[0], params_best[1],
                   params_best[2], params_best[3])
    p2_best = dsbpl(t_model, params_best[4], params_best[5], params_best[6],
                    params_best[7], params_best[8], params_best[9])
    ax.plot(t_model, ymod_best, '-', color='crimson', lw=2.0, zorder=6,
            label=f'Best fit (χ²_r={chi2_r:.2f})')
    ax.plot(t_model, np.where(p1_best>0, p1_best, np.nan), '--',
            color='goldenrod', lw=1.4, zorder=5,
            label=f'P1 tb={params_best[1]*1440:.1f} min'
                  f'  α=({params_best[2]:.2f},{params_best[3]:.2f})')
    ax.plot(t_model, np.where(p2_best>0, p2_best, np.nan), ':',
            color='steelblue', lw=1.6, zorder=5,
            label=f'P2 tb=({params_best[5]*1440:.1f},{params_best[6]*1440:.1f}) min'
                  f'  α=({params_best[7]:.2f},{params_best[8]:.2f},{params_best[9]:.2f})')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-4, 1.5)
    vis = (x_plot >= 1e-4) & (x_plot <= 1.5) & (y_plot > 0)
    ax.set_ylim(ye_plot[vis].min() * 0.3, max(y_plot[vis].max(), F_master) * 3)
    ax.set_xlabel('Days post-burst', fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, which='both', alpha=0.2, lw=0.5)

    ax_min = ax.twiny()
    ax_min.set_xscale('log'); ax_min.set_xlim(ax.get_xlim())
    min_ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ax_min.set_xticks([m/1440. for m in min_ticks])
    ax_min.set_xticklabels([str(m) for m in min_ticks], fontsize=8)
    ax_min.set_xlabel('Minutes post-burst', fontsize=10)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
plot_fit(axes[0], samplerA, mask_A,
         f'Fit A: t in [5 min, 1 d]   (N={len(xA)})', chi2_rA)
plot_fit(axes[1], samplerB, mask_B,
         f'Fit B: t in (0, 0.1 d]   (N={len(xB)})', chi2_rB)
axes[0].set_ylabel('Flux density (Jy)', fontsize=11)

plt.suptitle('GRB260207A — emcee 2-SBPL fits (log priors on F, tb)',
             fontsize=12, y=0.98)
plt.tight_layout()
plt.savefig('GRB260207A_emcee_compare.png', dpi=130, bbox_inches='tight')
plt.close()
print("\nSaved: GRB260207A_emcee_compare.png")

# ---------------------------------------------------------------
# Single-panel Fit B only plot
# ---------------------------------------------------------------
fig_b, ax_b = plt.subplots(figsize=(8, 6.5))
plot_fit(ax_b, samplerB, mask_B,
         f'Fit B: t in (0, 0.1 d]   (N={len(xB)})', chi2_rB)
ax_b.set_ylabel('Flux density (Jy)', fontsize=11)
plt.suptitle('GRB 260207A', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('GRB260207A_emcee_fitB.png', dpi=130, bbox_inches='tight')
plt.close()
print("Saved: GRB260207A_emcee_fitB.png")

# ---------------------------------------------------------------
# Save corner plot for Fit B (the more interesting one)
# ---------------------------------------------------------------
try:
    import corner
    samples_phys_B = samplesB.copy()
    # Convert log10 columns to physical (F0 in uJy, tb in min)
    samples_phys_B[:, 0] = 10**samples_phys_B[:, 0] * 1e6   # F0_1 uJy
    samples_phys_B[:, 1] = 10**samples_phys_B[:, 1] * 1440  # tb_1 min
    samples_phys_B[:, 4] = 10**samples_phys_B[:, 4] * 1e6   # F0_2 uJy
    samples_phys_B[:, 5] = 10**samples_phys_B[:, 5] * 1440  # tb2a_2 min
    samples_phys_B[:, 6] = 10**samples_phys_B[:, 6] * 1440  # tb2b_2 min
    with plt.rc_context({'text.usetex': True}):
        fig_c = corner.corner(
            samples_phys_B,
            labels=[r'$F_{0,1}$ ($\mu$Jy)', r'$t_{\rm b,1}$ (min)',
                    r'$\alpha_{1,1}$', r'$\alpha_{2,1}$',
                    r'$F_{0,2}$ ($\mu$Jy)', r'$t_{\rm b1,2}$ (min)', r'$t_{\rm b2,2}$ (min)',
                    r'$\alpha_{1,2}$', r'$\alpha_{2,2}$', r'$\alpha_{3,2}$'],
            quantiles=[0.16, 0.5, 0.84], show_titles=True,
            title_fmt='.2f', label_kwargs={'fontsize': 10})
        fig_c.savefig('GRB260207A_corner_B.png', dpi=110)
        plt.close(fig_c)
    print("Saved: GRB260207A_corner_B.png")
except ImportError:
    print("corner not available, skipping corner plot")
