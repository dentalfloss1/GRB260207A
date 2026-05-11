"""
GRB260207A — emcee comparison of three models.

Fit: t in (0, 2.0 d], unbinned.

Model A  SBPL + SBPL, S=0.1                  [8 free params]
Model B  SBPL(×M(t)) + SBPL, S=0.1          [11 free params]
Model C  SBPL + TSBPL, S=0.02               [12 free params]

All models fit P1 rise (a1_1) and decay (a2_1) slopes freely.

Parallelised: Models A, B, C run concurrently via ProcessPoolExecutor.
              log_prob functions are module-level so they are picklable.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import emcee
from concurrent.futures import ProcessPoolExecutor
from astropy.time import Time

# ---------------------------------------------------------------
# Module-level constants  (available to subprocesses on spawn)
# ---------------------------------------------------------------
trigger_tjd = Time("2026-02-07 05:40:16.947").jd - 2457000
master_obs  = Time("2026-02-07 05:46:04.3")
t_master    = master_obs.jd - 2457000 - trigger_tjd
F_master    = 3276.0 * 10**(-0.4 * 17.3)
eF_master   = F_master * 0.30

CADENCE    = 200 / 86400
HALF_C     = CADENCE / 2
S_AB       = 0.1
S_C        = 0.02
HALF_WIN   = CADENCE / 2.0
WIN_THRESH = 5.0 / 1440      # 5 min in days

# ---------------------------------------------------------------
# Base model functions (S passed explicitly)
# ---------------------------------------------------------------
def sbpl(t, F0, tb, a1, a2, S):
    return F0 * (t/tb)**(-a1) * (0.5*(1 + (t/tb)**(1/S)))**(-(a2-a1)*S)

def tsbpl(t, F0, tb1, tb2, tb3, a1, a2, a3, a4, S):
    f1 = (0.5*(1+(t/tb1)**(1/S)))**(-(a2-a1)*S)
    f2 = (0.5*(1+(t/tb2)**(1/S)))**(-(a3-a2)*S)
    f3 = (0.5*(1+(t/tb3)**(1/S)))**(-(a4-a3)*S)
    return F0 * (t/tb1)**(-a1) * f1 * f2 * f3

def windowed_eval(model_fn, t_arr, params):
    """Integrate early (t < 5 min) points over 200-s cadence window."""
    early = t_arr <= WIN_THRESH
    out   = np.empty_like(t_arr)
    if (~early).any():
        out[~early] = model_fn(t_arr[~early], *params)
    if early.any():
        for i in np.where(early)[0]:
            tmid  = t_arr[i]
            ts    = np.linspace(tmid - HALF_WIN, tmid + HALF_WIN, 11)
            valid = ts > 0
            if not valid.any():
                out[i] = 0.0
                continue
            f = np.zeros_like(ts)
            f[valid] = model_fn(ts[valid], *params)
            out[i] = np.trapezoid(f, ts) / CADENCE
    return out

# ===============================================================
# MODEL A — SBPL + SBPL, S=0.1
# theta: [logF1, logTb1, a1_1, a2_1, logF2, logTb2, a1_2, a2_2]
# ===============================================================
def model_A(t, F0_1, tb_1, a1_1, a2_1, F0_2, tb_2, a1_2, a2_2):
    p1 = sbpl(t, F0_1, tb_1, a1_1, a2_1, S_AB)
    p2 = sbpl(t, F0_2, tb_2, a1_2, a2_2, S_AB)
    return np.where(p1 > 0, p1, 0) + np.where(p2 > 0, p2, 0)

PRIOR_A = {
    'logF1':  (-6.0, -2.3),   'logTb1': (-2.523, -1.699),
    'a1_1':   (-10.0,  0.0),  'a2_1':   (  0.2,   5.0),
    'logF2':  (-6.0, -2.3),   'logTb2': (-1.699,  -1.0),
    'a1_2':   (-50.0,  0.0),  'a2_2':   (  0.2,   5.0),
}
NAMES_A = list(PRIOR_A.keys())

def log_prior_A(theta):
    for v, (lo, hi) in zip(theta, PRIOR_A.values()):
        if not (lo <= v <= hi): return -np.inf
    return 0.0

def theta_to_params_A(theta):
    logF1, logTb1, a1_1, a2_1, logF2, logTb2, a1_2, a2_2 = theta
    return (10**logF1, 10**logTb1, a1_1, a2_1,
            10**logF2, 10**logTb2, a1_2, a2_2)

def log_prob_A(theta, x, y, yerr):
    lp = log_prior_A(theta)
    if not np.isfinite(lp): return -np.inf
    params = theta_to_params_A(theta)
    try:    ymod = windowed_eval(model_A, x, params)
    except: return -np.inf
    if not np.all(np.isfinite(ymod)): return -np.inf
    return lp - 0.5 * np.sum(((y - ymod) / yerr)**2)

def get_components_A(t, p):
    # p: F0_1, tb_1, a1_1, a2_1, F0_2, tb_2, a1_2, a2_2
    c1 = sbpl(t, p[0], p[1], p[2], p[3], S_AB)
    c2 = sbpl(t, p[4], p[5], p[6], p[7], S_AB)
    return [
        ('P1', c1, '--', 'goldenrod',
         f"tb={p[1]*1440:.1f} min  α=({p[2]:.2f},{p[3]:.2f})"),
        ('P2', c2, ':',  'steelblue',
         f"tb={p[5]*1440:.1f} min  α=({p[6]:.2f},{p[7]:.2f})"),
    ]

theta0_A = np.array([np.log10(7e-4), np.log10(10/1440),
                     -0.5, 1.0,
                     np.log10(3e-4),  np.log10(75/1440), -10.0, 2.0])

# ===============================================================
# MODEL B — P1=SBPL×M(t) + P2=SBPL, S=0.1
# theta: [logF1, logTb1, a1_1, a2_1, logTinj, logA, logK,
#         logF2, logTb2, a1_2, a2_2]
# ===============================================================
def model_B(t, F0_1, tb_1, a1_1, a2_1, t_inj, A, k, F0_2, tb_2, a1_2, a2_2):
    sigmoid = np.where(t > 0, 1.0 / (1.0 + (t / t_inj)**(-k)), 0.0)
    p1 = sbpl(t, F0_1, tb_1, a1_1, a2_1, S_AB) * (1.0 + (A - 1.0) * sigmoid)
    p2 = sbpl(t, F0_2, tb_2, a1_2, a2_2, S_AB)
    return np.where(p1 > 0, p1, 0) + np.where(p2 > 0, p2, 0)

PRIOR_B = {
    'logF1':   (-6.0, -2.3),    'logTb1':  (-2.523, -1.699),
    'a1_1':    (-10.0,  0.0),   'a2_1':    (  0.2,   5.0),
    'logTinj': (-1.699, -0.699), 'logA':   (  0.0,   3.0),
    'logK':    (  0.0,  2.0),
    'logF2':   (-6.0, -2.3),    'logTb2':  (-1.699,  -1.0),
    'a1_2':    (-50.0,  0.0),   'a2_2':    (  0.2,   5.0),
}
NAMES_B = list(PRIOR_B.keys())

def log_prior_B(theta):
    for v, (lo, hi) in zip(theta, PRIOR_B.values()):
        if not (lo <= v <= hi): return -np.inf
    return 0.0

def theta_to_params_B(theta):
    logF1, logTb1, a1_1, a2_1, logTinj, logA, logK, logF2, logTb2, a1_2, a2_2 = theta
    return (10**logF1, 10**logTb1, a1_1, a2_1,
            10**logTinj, 10**logA, 10**logK,
            10**logF2, 10**logTb2, a1_2, a2_2)

def log_prob_B(theta, x, y, yerr):
    lp = log_prior_B(theta)
    if not np.isfinite(lp): return -np.inf
    params = theta_to_params_B(theta)
    try:    ymod = windowed_eval(model_B, x, params)
    except: return -np.inf
    if not np.all(np.isfinite(ymod)): return -np.inf
    return lp - 0.5 * np.sum(((y - ymod) / yerr)**2)

def get_components_B(t, p):
    # p: F0_1, tb_1, a1_1, a2_1, t_inj, A, k, F0_2, tb_2, a1_2, a2_2
    sigmoid = np.where(t > 0, 1.0 / (1.0 + (t / p[4])**(-p[6])), 0.0)
    c1 = sbpl(t, p[0], p[1], p[2], p[3], S_AB) * (1.0 + (p[5] - 1.0) * sigmoid)
    c2 = sbpl(t, p[7], p[8], p[9], p[10], S_AB)
    return [
        ('P1xM(t)', c1, '--', 'goldenrod',
         f"tb={p[1]*1440:.1f} min  α=({p[2]:.2f},{p[3]:.2f})"
         f"  t_inj={p[4]*1440:.1f} min  A={p[5]:.2f}  k={p[6]:.1f}"),
        ('P2',      c2, ':',  'steelblue',
         f"tb={p[8]*1440:.1f} min  α=({p[9]:.2f},{p[10]:.2f})"),
    ]

theta0_B = np.array([np.log10(7e-4), np.log10(10/1440),
                     -0.5, 1.0,
                     np.log10(0.05), np.log10(2.0), 1.0,
                     np.log10(3e-4),  np.log10(75/1440), -10.0, 2.0])

# ===============================================================
# MODEL C — SBPL + TSBPL, S=0.02
# theta: [logF1, logTb1, a1_1, a2_1,
#         logF2, logTb2a, logTb2b, logTb2c,
#         a1_2, a2_2, a3_2, a4_2]
# ===============================================================
def model_C(t, F0_1, tb_1, a1_1, a2_1, F0_2, tb2a, tb2b, tb2c, a1_2, a2_2, a3_2, a4_2):
    p1 = sbpl(t,  F0_1, tb_1, a1_1, a2_1, S_C)
    p2 = tsbpl(t, F0_2, tb2a, tb2b, tb2c, a1_2, a2_2, a3_2, a4_2, S_C)
    return np.where(p1 > 0, p1, 0) + np.where(p2 > 0, p2, 0)

PRIOR_C = {
    'logF1':   (-6.0, -2.3),    'logTb1':  (-2.523, -1.699),
    'a1_1':    (-10.0,  0.0),   'a2_1':    (  0.2,   5.0),
    'logF2':   (-6.0, -2.3),    'logTb2a': (-4.0,    0.0),
    'logTb2b': (-1.523, -1.155),  'logTb2c': (-4.0,   -0.699),
    'a1_2':    (-50.0,  0.0),   'a2_2':    ( -2.0,   3.0),
    'a3_2':    (  0.2,  5.0),   'a4_2':    (  0.5,  30.0),
}
NAMES_C = list(PRIOR_C.keys())

def log_prior_C(theta):
    for v, (lo, hi) in zip(theta, PRIOR_C.values()):
        if not (lo <= v <= hi): return -np.inf
    # logTb2a < logTb2b < logTb2c  (indices 5, 6, 7 in new ordering)
    if theta[6] <= theta[5] or theta[7] <= theta[6]: return -np.inf
    return 0.0

def theta_to_params_C(theta):
    logF1, logTb1, a1_1, a2_1, logF2, logTb2a, logTb2b, logTb2c, a1_2, a2_2, a3_2, a4_2 = theta
    return (10**logF1, 10**logTb1, a1_1, a2_1,
            10**logF2, 10**logTb2a, 10**logTb2b, 10**logTb2c,
            a1_2, a2_2, a3_2, a4_2)

def log_prob_C(theta, x, y, yerr):
    lp = log_prior_C(theta)
    if not np.isfinite(lp): return -np.inf
    params = theta_to_params_C(theta)
    try:    ymod = windowed_eval(model_C, x, params)
    except: return -np.inf
    if not np.all(np.isfinite(ymod)): return -np.inf
    return lp - 0.5 * np.sum(((y - ymod) / yerr)**2)

def get_components_C(t, p):
    # p: F0_1, tb_1, a1_1, a2_1, F0_2, tb2a, tb2b, tb2c,
    #    a1_2, a2_2, a3_2, a4_2
    c1 = sbpl(t,  p[0], p[1], p[2], p[3], S_C)
    c2 = tsbpl(t, p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], S_C)
    return [
        ('P1',         c1, '--', 'goldenrod',
         f"tb={p[1]*1440:.1f} min  α=({p[2]:.2f},{p[3]:.2f})"),
        ('P2 (TSBPL)', c2, ':',  'steelblue',
         f"tb=({p[5]*1440:.1f},{p[6]*1440:.1f},{p[7]*1440:.1f}) min  "
         f"α=({p[8]:.2f},{p[9]:.2f},{p[10]:.2f},{p[11]:.2f})"),
    ]

theta0_C = np.array([np.log10(7e-4), np.log10(10/1440),
                     -0.5, 1.0,
                     np.log10(3e-4),
                     np.log10(45/1440), np.log10(0.05), np.log10(115/1440),
                     -10.0, 0.0, 2.0, 10.0])

# ---------------------------------------------------------------
# Generic emcee runner
# ---------------------------------------------------------------
def run_emcee(x, y, yerr, theta0, prior_ranges, log_prob_fn, label,
              nwalkers=32, nburn=1000, nprod=100_000):
    ndim = len(theta0)
    pos  = theta0 + 1e-3 * np.random.randn(nwalkers, ndim)
    for i in range(nwalkers):
        for j, (lo, hi) in enumerate(prior_ranges.values()):
            pos[i, j] = np.clip(pos[i, j], lo + 1e-4, hi - 1e-4)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, args=(x, y, yerr))
    print(f"[{label}] burn-in ({nburn} steps, {nwalkers} walkers)...")
    state = sampler.run_mcmc(pos, nburn, progress=False)
    sampler.reset()
    print(f"[{label}] production ({nprod} steps)...")
    sampler.run_mcmc(state, nprod, progress=False)

    af = np.mean(sampler.acceptance_fraction)
    try:    tau_max = np.max(sampler.get_autocorr_time(quiet=True))
    except: tau_max = None
    print(f"[{label}] acceptance={af:.3f}  max_tau={tau_max}")
    return sampler.get_chain(flat=True), sampler.get_log_prob(flat=True)

# ---------------------------------------------------------------
# Parallel worker functions  (module-level = picklable)
# ---------------------------------------------------------------
def _fit_A(xD, yD, yeD):
    np.random.seed(42)
    return run_emcee(xD, yD, yeD, theta0_A, PRIOR_A, log_prob_A, 'Model A')

def _fit_B(xD, yD, yeD):
    np.random.seed(43)
    return run_emcee(xD, yD, yeD, theta0_B, PRIOR_B, log_prob_B, 'Model B')

def _fit_C(xD, yD, yeD):
    np.random.seed(44)
    return run_emcee(xD, yD, yeD, theta0_C, PRIOR_C, log_prob_C, 'Model C')

# ---------------------------------------------------------------
# Summarize flat chains
# ---------------------------------------------------------------
def summarize(flat_chain, flat_lp, theta_to_params_fn, param_names):
    best_idx    = np.argmax(flat_lp)
    theta_best  = flat_chain[best_idx]
    params_best = theta_to_params_fn(theta_best)
    quantiles   = []
    for j, name in enumerate(param_names):
        phys = 10**flat_chain[:, j] if name.startswith('log') else flat_chain[:, j]
        q16, q50, q84 = np.percentile(phys, [16, 50, 84])
        quantiles.append((name, q16, q50, q84))
    return theta_best, params_best, quantiles

# ---------------------------------------------------------------
# Print parameter table
# ---------------------------------------------------------------
def print_params(tag, theta_best, param_names, quantiles,
                 labels_phys, scale, chi2_r, bic_val):
    print(f"\n=== MODEL {tag} ===  chi2_r={chi2_r:.3f}   BIC={bic_val:.1f}")
    print(f"  {'Parameter':>24s}   {'Best-fit':>9s}   {'Median':>9s}"
          f"   {'+1sigma':>8s}   {'-1sigma':>8s}")
    print(f"  {'-'*72}")
    for j, ((name, q16, q50, q84), lbl, sc) in enumerate(zip(quantiles, labels_phys, scale)):
        best = 10**theta_best[j] * sc if name.startswith('log') else theta_best[j] * sc
        print(f"  {lbl:>24s}: {best:9.3f}   {q50*sc:9.3f}"
              f"  +{(q84-q50)*sc:7.3f}  -{(q50-q16)*sc:7.3f}")

# ---------------------------------------------------------------
# Corner plots
# ---------------------------------------------------------------
_CORNER_CFG = {
    'A': dict(
        names=NAMES_A,
        scale=[1e6, 1440, 1, 1, 1e6, 1440, 1, 1],
        labels=[r'$F_{0,1}\ (\mu\mathrm{Jy})$', r'$t_{b,1}\ (\mathrm{min})$',
                r'$\alpha_{1,1}$', r'$\alpha_{2,1}$',
                r'$F_{0,2}\ (\mu\mathrm{Jy})$', r'$t_{b,2}\ (\mathrm{min})$',
                r'$\alpha_{1,2}$', r'$\alpha_{2,2}$'],
    ),
    'B': dict(
        names=NAMES_B,
        scale=[1e6, 1440, 1, 1, 1440, 1, 1, 1e6, 1440, 1, 1],
        labels=[r'$F_{0,1}\ (\mu\mathrm{Jy})$', r'$t_{b,1}\ (\mathrm{min})$',
                r'$\alpha_{1,1}$', r'$\alpha_{2,1}$',
                r'$t_{\rm inj}\ (\mathrm{min})$', r'$A$', r'$k$',
                r'$F_{0,2}\ (\mu\mathrm{Jy})$', r'$t_{b,2}\ (\mathrm{min})$',
                r'$\alpha_{1,2}$', r'$\alpha_{2,2}$'],
    ),
    'C': dict(
        names=NAMES_C,
        scale=[1e6, 1440, 1, 1, 1e6, 1440, 1440, 1440, 1, 1, 1, 1],
        labels=[r'$F_{0,1}\ (\mu\mathrm{Jy})$', r'$t_{b,1}\ (\mathrm{min})$',
                r'$\alpha_{1,1}$', r'$\alpha_{2,1}$',
                r'$F_{0,2}\ (\mu\mathrm{Jy})$',
                r'$t_{b1,2}\ (\mathrm{min})$', r'$t_{b2,2}\ (\mathrm{min})$',
                r'$t_{b3,2}\ (\mathrm{min})$',
                r'$\alpha_{1,2}$', r'$\alpha_{2,2}$',
                r'$\alpha_{3,2}$', r'$\alpha_{4,2}$'],
    ),
}

def save_corner(tag, flat_chain):
    try:
        import corner
    except ImportError:
        print("corner not installed — skipping corner plots")
        return
    cfg  = _CORNER_CFG[tag]
    samp = flat_chain.copy()
    for j, (name, sc) in enumerate(zip(cfg['names'], cfg['scale'])):
        samp[:, j] = 10**samp[:, j] * sc if name.startswith('log') else samp[:, j] * sc
    try:
        with plt.rc_context({'text.usetex': True}):
            fig_c = corner.corner(samp, labels=cfg['labels'],
                                  quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                  title_fmt='.2f', label_kwargs={'fontsize': 9})
    except Exception:
        fig_c = corner.corner(samp, labels=cfg['labels'],
                              quantiles=[0.16, 0.5, 0.84], show_titles=True,
                              title_fmt='.2f', label_kwargs={'fontsize': 9})
    fig_c.savefig(f'GRB260207A_corner_{tag}.png', dpi=110, bbox_inches='tight')
    plt.close(fig_c)
    print(f"Saved: GRB260207A_corner_{tag}.png")

# ---------------------------------------------------------------
# Plotting helpers  (reference x_plot/y_plot/ye_plot as module globals
#                    set inside __main__ before these are ever called)
# ---------------------------------------------------------------
MODEL_COLOR = {'A': 'crimson', 'B': 'royalblue', 'C': 'darkorange'}

def add_data_to_ax(ax, mask_used, alpha_excl=0.4):
    pos          = y_plot > 0
    fitted_pos   = mask_used & pos
    excluded_pos = ~mask_used & pos & (x_plot < 10)
    if (~pos).any():
        ax.errorbar(x_plot[~pos], np.abs(y_plot[~pos]), yerr=ye_plot[~pos],
                    fmt='.', color='lightgray', markersize=2, elinewidth=0.4,
                    alpha=0.3, capsize=0, zorder=1)
    if excluded_pos.any():
        ax.errorbar(x_plot[excluded_pos], y_plot[excluded_pos],
                    yerr=ye_plot[excluded_pos], fmt='.', color='gray',
                    markersize=2, elinewidth=0.3, alpha=alpha_excl,
                    capsize=0, zorder=1.5, label='Excluded')
    ax.errorbar(x_plot[fitted_pos], y_plot[fitted_pos],
                yerr=ye_plot[fitted_pos], fmt='.', color='black',
                markersize=3, elinewidth=0.4, alpha=0.8, capsize=0,
                zorder=2, label='Fitted')
    # ax.errorbar(t_master, F_master, yerr=eF_master, fmt='D',
    #             color='mediumseagreen', markersize=8, elinewidth=1.4,
    #             capsize=4, markeredgecolor='k', markeredgewidth=0.6,
    #             label=f'MASTER T+{t_master*1440:.1f} min')

def format_main_ax(ax):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-4, 10)
    vis = (x_plot >= 1e-4) & (x_plot <= 10) & (y_plot > 0)
    ax.set_ylim(ye_plot[vis].min() * 0.3, max(y_plot[vis].max(), F_master) * 3)
    ax.grid(True, which='both', alpha=0.2, lw=0.5)

def add_minutes_axis(ax):
    ax_min = ax.twiny()
    ax_min.set_xscale('log'); ax_min.set_xlim(ax.get_xlim())
    min_ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ax_min.set_xticks([m/1440. for m in min_ticks])
    ax_min.set_xticklabels([str(m) for m in min_ticks], fontsize=8)
    ax_min.set_xlabel('Minutes post-burst', fontsize=10)

def plot_residuals(ax_res, mask_used, model_fn, params_best, line_color):
    pos_fit   = mask_used & (y_plot > 0)
    ymod_data = model_fn(x_plot[pos_fit], *params_best)
    resid     = (y_plot[pos_fit] - ymod_data) / ye_plot[pos_fit]
    ax_res.axhline(0,  color=line_color, lw=1.2, zorder=5)
    ax_res.axhline( 3, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax_res.axhline(-3, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax_res.errorbar(x_plot[pos_fit], resid, yerr=np.ones_like(resid),
                    fmt='.', color='black', markersize=3, elinewidth=0.4,
                    alpha=0.8, capsize=0, zorder=2)
    ax_res.set_xscale('log'); ax_res.set_xlim(1e-4, 10); ax_res.set_ylim(-7, 7)
    ax_res.set_xlabel('Days post-burst', fontsize=11)
    ax_res.set_ylabel('Residuals (σ)', fontsize=10)
    ax_res.grid(True, which='both', alpha=0.2, lw=0.5)

# ===============================================================
# Main
# ===============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nomodel', action='store_true', help='Plot data only')
    args = parser.parse_args()

    rawdata   = np.loadtxt('lc_GRB260207A_cand41148_geo')
    x_all     = rawdata[:, 0] - trigger_tjd
    y_all     = -rawdata[:, 1] / (200 * 0.8 * 0.99)
    yerr_all  =  rawdata[:, 2] / (200 * 0.8 * 0.99)
    zp        = 2416 * 10**(-0.4 * 20.44)
    flux_all  = y_all    * zp
    eflux_all = yerr_all * zp

    mask_plot = (x_all > -HALF_C) & (x_all <= 10)
    x_plot  = x_all[mask_plot]
    y_plot  = flux_all[mask_plot]
    ye_plot = eflux_all[mask_plot]

    if args.nomodel:
        # --- Running RMS of background (col 5), window=10, sliding
        bkg_col = rawdata[:, 5]
        N_raw   = len(bkg_col)
        WIN     = 10
        flagged = np.zeros(N_raw, dtype=bool)
        n_bad_windows = 0
        for i in range(N_raw - WIN + 1):
            rms_i = np.sqrt(np.mean(bkg_col[i:i+WIN]**2))
            if rms_i > 1.0:
                flagged[i:i+WIN] = True
                n_bad_windows += 1
        print(f"Running RMS (window={WIN}): {n_bad_windows} windows exceed RMS>1, "
              f"{flagged.sum()} data points flagged")

        # --- Extended mask: first cadence onward through end of data
        mask_ext = (x_all > -HALF_C) & (x_all <= x_all.max())
        xp   = x_all[mask_ext]
        yp   = y_all[mask_ext]
        yep  = yerr_all[mask_ext]
        fp   = flagged[mask_ext]

        # --- Pre-burst baseline: weighted mean of unflagged t in [-6 h, 0]
        pre = (x_all >= -0.25) & (x_all <= 0) & ~flagged
        wt      = 1.0 / yerr_all[pre]**2
        bkg_val = np.sum(wt * y_all[pre]) / np.sum(wt)
        bkg_err = np.sqrt(1.0 / np.sum(wt))
        print(f"Pre-burst baseline: {bkg_val:.4f} +/- {bkg_err:.4f} cps  (N={pre.sum()})")

        # --- Log-bin unflagged data from 0.07 d onwards
        BIN_START = 0.07
        nbins     = 20
        sel       = (xp >= BIN_START) & ~fp
        x_s, y_s, ye_s = xp[sel], yp[sel], yep[sel]
        x_binned, y_binned, ye_binned = [], [], []
        if len(x_s) > 0:
            bins = np.logspace(np.log10(BIN_START), np.log10(xp.max()), nbins + 1)
            for i in range(nbins):
                in_bin = (x_s >= bins[i]) & (x_s < bins[i+1])
                if in_bin.sum() == 0:
                    continue
                w    = 1.0 / ye_s[in_bin]**2
                y_b  = np.sum(w * y_s[in_bin]) / np.sum(w)
                ye_b = np.sqrt(1.0 / np.sum(w))
                x_b  = np.exp(np.mean(np.log(x_s[in_bin])))
                x_binned.append(x_b); y_binned.append(y_b); ye_binned.append(ye_b)
        x_binned  = np.array(x_binned)
        y_binned  = np.array(y_binned)
        ye_binned = np.array(ye_binned)
        print(f"Log bins (>={BIN_START} d): {len(x_binned)} non-empty bins")

        # --- Plot
        fig, ax = plt.subplots(figsize=(8, 6.5))

        # Unflagged data
        good_pos = ~fp & (yp >  0)
        good_neg = ~fp & (yp <= 0)
        bad_pos  =  fp & (yp >  0)
        bad_neg  =  fp & (yp <= 0)

        if good_neg.any():
            ax.errorbar(xp[good_neg], np.abs(yp[good_neg]), yerr=yep[good_neg],
                        fmt='.', color='lightgray', markersize=2, elinewidth=0.4,
                        alpha=0.3, capsize=0, zorder=1)
        ax.errorbar(xp[good_pos], yp[good_pos], yerr=yep[good_pos],
                    fmt='.', color='black', markersize=3, elinewidth=0.4,
                    alpha=0.8, capsize=0, zorder=2, label='TESS (unflagged)')

        # Flagged data
        if bad_neg.any():
            ax.errorbar(xp[bad_neg], np.abs(yp[bad_neg]), yerr=yep[bad_neg],
                        fmt='.', color='lightsalmon', markersize=2, elinewidth=0.4,
                        alpha=0.3, capsize=0, zorder=1.5)
        if bad_pos.any():
            ax.errorbar(xp[bad_pos], yp[bad_pos], yerr=yep[bad_pos],
                        fmt='.', color='tomato', markersize=3, elinewidth=0.4,
                        alpha=0.6, capsize=0, zorder=2.5,
                        label=f'Flagged (RMS$_{{10}}>1$, N={fp.sum()})')

        # ax.errorbar(t_master, F_master, yerr=eF_master, fmt='D',
        #             color='mediumseagreen', markersize=8, elinewidth=1.4,
        #             capsize=4, markeredgecolor='k', markeredgewidth=0.6,
        #             label=f'MASTER T+{t_master*1440:.1f} min')

        # Log-binned data overlay
        if len(x_binned) > 0:
            bin_pos = y_binned > 0
            if bin_pos.any():
                ax.errorbar(x_binned[bin_pos], y_binned[bin_pos],
                            yerr=ye_binned[bin_pos],
                            fmt='o', color='crimson', markersize=6,
                            elinewidth=1.2, capsize=3, zorder=6,
                            label=f'Log-binned ($\\geq${BIN_START} d)')

        # Pre-burst baseline overplot
        t_lo   = xp[xp > 0].min()
        t_hi   = xp.max()
        t_fill = np.logspace(np.log10(t_lo), np.log10(t_hi), 300)
        ax.axhline(bkg_val, color='royalblue', lw=1.5, ls='--', zorder=5,
                   label=f'Pre-burst baseline: {bkg_val:.3f} $\\pm$ {bkg_err:.3f} cps')
        ax.fill_between(t_fill, bkg_val - bkg_err, bkg_val + bkg_err,
                        color='royalblue', alpha=0.2, zorder=4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-4, t_hi * 1.05)
        vis = (xp > 1e-4) & (yp > 0) & ~fp
        ax.set_ylim(yep[vis].min() * 0.3, yp[vis].max() * 3)
        ax.grid(True, which='both', alpha=0.2, lw=0.5)
        ax.set_xlabel('Days post-burst', fontsize=11)
        ax.set_ylabel('Counts s$^{-1}$', fontsize=11)
        ax.legend(fontsize=8, loc='lower left')
        add_minutes_axis(ax)
        plt.suptitle('GRB 260207A', fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig('GRB260207A_data_only.png', dpi=130, bbox_inches='tight')
        plt.close()
        print("Saved: GRB260207A_data_only.png")
        raise SystemExit(0)

    mask_fit = (x_plot > 0) & (x_plot <= 2.0) & (y_plot > 0)
    xD = x_plot[mask_fit]; yD = y_plot[mask_fit]; yeD = ye_plot[mask_fit]
    print(f"N_fit = {len(xD)},  t in [{xD.min()*1440:.2f} min, {xD.max():.3f} d]")

    # -----------------------------------------------------------
    # Run all three models in parallel
    # -----------------------------------------------------------
    print("\nLaunching Models A, B, C in parallel (3 worker processes)...")
    with ProcessPoolExecutor(max_workers=3) as ex:
        fut_A = ex.submit(_fit_A, xD, yD, yeD)
        fut_B = ex.submit(_fit_B, xD, yD, yeD)
        fut_C = ex.submit(_fit_C, xD, yD, yeD)
        chainA, lpA = fut_A.result()
        chainB, lpB = fut_B.result()
        chainC, lpC = fut_C.result()

    # -----------------------------------------------------------
    # Summarize
    # -----------------------------------------------------------
    thetaA, pA, qA = summarize(chainA, lpA, theta_to_params_A, NAMES_A)
    thetaB, pB, qB = summarize(chainB, lpB, theta_to_params_B, NAMES_B)
    thetaC, pC, qC = summarize(chainC, lpC, theta_to_params_C, NAMES_C)

    N = len(xD)
    chi2_A  = -2*np.max(lpA);  chi2_rA = chi2_A / (N - 8);   bic_A = chi2_A +  8*np.log(N)
    chi2_B  = -2*np.max(lpB);  chi2_rB = chi2_B / (N - 11);  bic_B = chi2_B + 11*np.log(N)
    chi2_C  = -2*np.max(lpC);  chi2_rC = chi2_C / (N - 12);  bic_C = chi2_C + 12*np.log(N)

    print_params('A', thetaA, NAMES_A, qA,
                 ['F0_1 (uJy)', 'tb_1 (min)', 'a1_1', 'a2_1',
                  'F0_2 (uJy)', 'tb_2 (min)', 'a1_2', 'a2_2'],
                 [1e6, 1440, 1, 1, 1e6, 1440, 1, 1],
                 chi2_rA, bic_A)

    print_params('B', thetaB, NAMES_B, qB,
                 ['F0_1 (uJy)', 'tb_1 (min)', 'a1_1', 'a2_1',
                  't_inj (min)', 'A', 'k',
                  'F0_2 (uJy)', 'tb_2 (min)', 'a1_2', 'a2_2'],
                 [1e6, 1440, 1, 1, 1440, 1, 1, 1e6, 1440, 1, 1],
                 chi2_rB, bic_B)

    print_params('C', thetaC, NAMES_C, qC,
                 ['F0_1 (uJy)', 'tb_1 (min)', 'a1_1', 'a2_1',
                  'F0_2 (uJy)', 'tb2a (min)', 'tb2b (min)', 'tb2c (min)',
                  'a1_2', 'a2_2', 'a3_2', 'a4_2'],
                 [1e6, 1440, 1, 1, 1e6, 1440, 1440, 1440, 1, 1, 1, 1],
                 chi2_rC, bic_C)

    print(f"\n=== MODEL COMPARISON  (N={N}, t <= 2 d) ===")
    print(f"  Model A   8 params   chi2_r={chi2_rA:.3f}   BIC={bic_A:.1f}")
    print(f"  Model B  11 params   chi2_r={chi2_rB:.3f}   BIC={bic_B:.1f}")
    print(f"  Model C  12 params   chi2_r={chi2_rC:.3f}   BIC={bic_C:.1f}")
    print(f"  Delta_BIC(B-A)={bic_B-bic_A:.1f}   Delta_BIC(C-A)={bic_C-bic_A:.1f}")

    # -----------------------------------------------------------
    # Corner plots
    # -----------------------------------------------------------
    save_corner('A', chainA)
    save_corner('B', chainB)
    save_corner('C', chainC)

    # -----------------------------------------------------------
    # Individual model figures (main panel + residuals)
    # -----------------------------------------------------------
    t_model = np.logspace(np.log10(2e-4), np.log10(10), 1500)

    MODELS = [
        ('A', chainA, lpA, theta_to_params_A, model_A, get_components_A,
         chi2_rA,  8, 'Model A: SBPL+SBPL, S=0.1'),
        ('B', chainB, lpB, theta_to_params_B, model_B, get_components_B,
         chi2_rB, 11, 'Model B: SBPL(xM(t))+SBPL, S=0.1'),
        ('C', chainC, lpC, theta_to_params_C, model_C, get_components_C,
         chi2_rC, 12, 'Model C: SBPL+TSBPL, S=0.02'),
    ]

    for (tag, chain, lp, theta_fn, mfn, comp_fn, chi2_r, ndof, title) in MODELS:
        p_best = theta_fn(chain[np.argmax(lp)])
        color  = MODEL_COLOR[tag]

        fig, (ax, ax_res) = plt.subplots(
            2, 1, figsize=(8, 9),
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

        add_data_to_ax(ax, mask_fit)

        rng = np.random.default_rng(42)
        for i in rng.choice(len(chain), 200, replace=False):
            ax.plot(t_model, mfn(t_model, *theta_fn(chain[i])),
                    '-', color=color, lw=0.3, alpha=0.04, zorder=4)

        ax.plot(t_model, mfn(t_model, *p_best), '-', color=color, lw=2.0, zorder=6,
                label=f'Best fit  chi2_r={chi2_r:.2f}')
        for (clabel, cy, ls, ccolor, desc) in comp_fn(t_model, p_best):
            ax.plot(t_model, np.where(cy > 0, cy, np.nan),
                    ls, color=ccolor, lw=1.4, zorder=5, label=f'{clabel}: {desc}')

        format_main_ax(ax)
        ax.tick_params(labelbottom=False)
        ax.set_ylabel('Flux density (Jy)', fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6.5, loc='lower left')
        add_minutes_axis(ax)
        plot_residuals(ax_res, mask_fit, mfn, p_best, color)

        plt.suptitle('GRB 260207A', fontsize=14, y=0.995)
        plt.savefig(f'GRB260207A_emcee_{tag}.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f"Saved: GRB260207A_emcee_{tag}.png")

    # -----------------------------------------------------------
    # Comparison figure
    # -----------------------------------------------------------
    bic_by_tag = {'A': bic_A, 'B': bic_B, 'C': bic_C}

    fig_cmp = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig_cmp,
                           height_ratios=[3, 1], hspace=0.08, wspace=0.08)
    ax_main = fig_cmp.add_subplot(gs[0, :])
    ax_rA   = fig_cmp.add_subplot(gs[1, 0])
    ax_rB   = fig_cmp.add_subplot(gs[1, 1])
    ax_rC   = fig_cmp.add_subplot(gs[1, 2])

    add_data_to_ax(ax_main, mask_fit, alpha_excl=0.25)
    for (tag, chain, lp, theta_fn, mfn, _, chi2_r, _, _) in MODELS:
        p_best = theta_fn(chain[np.argmax(lp)])
        ax_main.plot(t_model, mfn(t_model, *p_best), '-',
                     color=MODEL_COLOR[tag], lw=2.0, zorder=6,
                     label=f'Model {tag}  chi2_r={chi2_r:.2f}  BIC={bic_by_tag[tag]:.0f}')

    format_main_ax(ax_main)
    ax_main.tick_params(labelbottom=False)
    ax_main.set_ylabel('Flux density (Jy)', fontsize=11)
    ax_main.legend(fontsize=8, loc='lower left')
    add_minutes_axis(ax_main)

    for ax_res, (tag, chain, lp, theta_fn, mfn, _, chi2_r, _, _) in zip(
            [ax_rA, ax_rB, ax_rC], MODELS):
        p_best = theta_fn(chain[np.argmax(lp)])
        plot_residuals(ax_res, mask_fit, mfn, p_best, MODEL_COLOR[tag])
        ax_res.set_title(f'Model {tag}', fontsize=9)
        if tag != 'A':
            ax_res.set_ylabel('')
            ax_res.tick_params(labelleft=False)

    plt.suptitle('GRB 260207A — Model Comparison', fontsize=13, y=0.995)
    plt.savefig('GRB260207A_emcee_compare.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: GRB260207A_emcee_compare.png")
