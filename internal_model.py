"""GRB260207A — emcee fit of a forward shock plus internal flare.

The early forward shock is fit first.  Its posterior-median electron index,
rise slope, and smoothness are then fixed in a full trigger-frame fit:

    F_model(t) = F_FS(t) + F_flare(t) + C_bg

The flare has a logistic turn-on/plateau and a curvature tail.  The curvature
relation uses beta=1 and t0=t_rise, so its fixed exponent is -3.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import emcee
from astropy.time import Time

# ---------------------------------------------------------------
# Module-level constants  (available to subprocesses on spawn)
# ---------------------------------------------------------------
trigger_tjd = Time("2026-02-07 05:42:33.65").jd - 2457000
master_obs  = Time("2026-02-07 05:46:04.3")
t_master    = master_obs.jd - 2457000 - trigger_tjd
F_master    = 3276.0 * 10**(-0.4 * 17.3)
eF_master   = F_master * 0.30

CADENCE    = 200 / 86400
HALF_C     = CADENCE / 2
S_AB       = 0.02
HALF_WIN   = CADENCE / 2.0
WIN_THRESH = 5.0 / 1440      # 5 min in days
FS_FIT_MAX = 0.02            # days post-burst
INTERNAL_FIT_MAX = 6.5       # days post-burst for FS+flare fit
PREBURST_MIN = -1.0          # one day before trigger
PREBURST_MAX = 0.0           # trigger
INTERNAL_MODEL_VERSION = 2
BETA_INTERNAL = 1.0

# ---------------------------------------------------------------
# Base model functions (S passed explicitly)
# ---------------------------------------------------------------
def sbpl(t, F0, tb, a1, a2, S):
    return F0 * (t/tb)**(-a1) * (0.5*(1 + (t/tb)**(1/S)))**(-(a2-a1)*S)

def windowed_eval(model_fn, t_arr, params):
    """Integrate near-trigger (t < 5 min) points over 200-s cadence window.

    Pre-burst points whose entire window is at t<=0 are vectorised into a
    single model call (returns constant background only — no power-law loop).
    Only the handful of points near t=0 whose window straddles the trigger
    go through the per-point integration loop.
    """
    early = t_arr <= WIN_THRESH          # includes all t < 0
    out   = np.empty_like(t_arr)

    # --- late points: direct vectorised evaluation
    if (~early).any():
        out[~early] = model_fn(t_arr[~early], *params)

    # --- pre-burst: window entirely ≤ 0 → vectorised model call (bg only)
    pre_burst = early & (t_arr < -HALF_WIN)
    if pre_burst.any():
        out[pre_burst] = model_fn(t_arr[pre_burst], *params)

    # --- near-trigger: window straddles or is entirely in t > 0 → integrate
    near = early & ~pre_burst
    for i in np.where(near)[0]:
        tmid  = t_arr[i]
        ts    = np.linspace(tmid - HALF_WIN, tmid + HALF_WIN, 11)
        valid = ts > 0
        if not valid.any():
            out[i] = model_fn(np.array([tmid]), *params)[0]
            continue
        f = np.zeros_like(ts)
        f[valid] = model_fn(ts[valid], *params)
        out[i] = np.trapezoid(f, ts) / CADENCE
    return out

def fit_constant_background(flux, flux_err):
    """Weighted constant fit; returns mean, mean error, chi2, and dof."""
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    valid = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Need at least two finite points to fit a constant background")
    flux = flux[valid]
    flux_err = flux_err[valid]
    weight = 1.0 / flux_err**2
    mean = np.sum(weight * flux) / np.sum(weight)
    mean_err = np.sqrt(1.0 / np.sum(weight))
    chi2 = np.sum(((flux - mean) / flux_err)**2)
    return mean, mean_err, chi2, len(flux) - 1

def estimate_additive_jitter(flux, flux_err):
    """Find a constant error floor that gives a constant fit chi2/dof = 1.

    The background mean is re-fit for every trial jitter. The returned jitter
    has the same units as ``flux`` and is zero when the formal errors already
    give chi2/dof <= 1.
    """
    _, _, chi2, dof = fit_constant_background(flux, flux_err)
    if chi2 <= dof:
        return 0.0

    flux_err = np.asarray(flux_err, dtype=float)
    finite_err = flux_err[np.isfinite(flux_err) & (flux_err > 0)]

    def excess_chi2(jitter):
        effective_err = np.sqrt(flux_err**2 + jitter**2)
        return fit_constant_background(flux, effective_err)[2] - dof

    lower = 0.0
    upper = float(np.median(finite_err))
    while excess_chi2(upper) > 0:
        upper *= 2.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if excess_chi2(midpoint) > 0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)

# ===============================================================
# Forward-shock model — SBPL + TESS-bg constant
# theta: [logF0, logTb, logC_bg, logS_FS, p, alpha_rise_FS]
# ===============================================================
ALPHA_RISE_INITIAL = 0.5
ALPHA_RISE_INTERNAL = ALPHA_RISE_INITIAL
P_INITIAL = 2.3
P_INTERNAL = P_INITIAL

def decay_alpha_from_p(p):
    return 3.0 * (p - 1.0) / 4.0

def forward_shock_source_flux(t, F0, tb, p, S=None, alpha_rise=None):
    """Source-only forward-shock flux; zero before the trigger."""
    t = np.asarray(t, dtype=float)
    pos = t > 0
    ts = np.where(pos, t, 1e-10)
    a2 = decay_alpha_from_p(p)
    smooth = S_AB if S is None else S
    rise = ALPHA_RISE_INTERNAL if alpha_rise is None else alpha_rise
    return np.where(pos, sbpl(ts, F0, tb, -rise, a2, smooth), 0.0)

def model_FS(t, F0, tb, C_bg, S_FS, p, alpha_rise_FS):
    return forward_shock_source_flux(t, F0, tb, p, S_FS, alpha_rise_FS) + C_bg

PRIOR_FS = {
    'logF0':   (-6.0, -2.3),
    'logTb':   (-2.523, -1.699),  # 3–30 min, matching the original P1 range
    'logC_bg': (-10.0, -4.0),
    'logS_FS': (np.log10(0.005), np.log10(0.5)),
    'p':       (1.5, 3.0),
    'alpha_rise_FS': (0.0, 2.0),
}
NAMES_FS = list(PRIOR_FS.keys())

def log_prior_FS(theta):
    for v, (lo, hi) in zip(theta, PRIOR_FS.values()):
        if not (lo <= v <= hi): return -np.inf
    return 0.0

def theta_to_params_FS(theta):
    logF0, logTb, logC_bg, logS_FS, p, alpha_rise_FS = theta
    return (10**logF0, 10**logTb, 10**logC_bg, 10**logS_FS,
            p, alpha_rise_FS)

def log_prob_FS(theta, x, y, yerr):
    lp = log_prior_FS(theta)
    if not np.isfinite(lp): return -np.inf
    params = theta_to_params_FS(theta)
    try:    ymod = windowed_eval(model_FS, x, params)
    except: return -np.inf
    if not np.all(np.isfinite(ymod)): return -np.inf
    return lp - 0.5 * np.sum(((y - ymod) / yerr)**2)

def get_components_FS(t, p):
    a2  = decay_alpha_from_p(p[4])
    fs  = forward_shock_source_flux(t, p[0], p[1], p[4], p[3], p[5])
    cbg = np.full_like(t, p[2])
    return [
        ('Forward shock', fs, '--', 'goldenrod',
         f"tb={p[1]*1440:.1f} min  S={p[3]:.3f}  "
         f"rise=+{p[5]:.2f}  decay=-{a2:.2f}  p={p[4]:.2f}"),
        ('TESS bg', cbg, '-.', 'mediumseagreen',
         f"C_bg={p[2]*1e6:.2f} µJy"),
    ]

theta0_FS = np.array([
    np.log10(7e-4), np.log10(10/1440), -4.5, np.log10(S_AB),
    P_INITIAL, ALPHA_RISE_INITIAL,
])

def fs_smoothness_posterior_median(flat_chain):
    return float(np.median(10**np.asarray(flat_chain)[:, NAMES_FS.index('logS_FS')]))

def fs_rise_posterior_median(flat_chain):
    return float(np.median(np.asarray(flat_chain)[:, NAMES_FS.index('alpha_rise_FS')]))

def fs_p_posterior_median(flat_chain):
    return float(np.median(np.asarray(flat_chain)[:, NAMES_FS.index('p')]))

# ===============================================================
# Internal model — unchanged FS plus piecewise flare and TESS background
# ===============================================================
NAMES_INTERNAL = [
    'logF0_FS', 'logTb_FS', 'logC_bg', 't_rise', 'tau_rise', 't_off',
    'logF_peak',
]
PRIOR_INTERNAL = {
    'logF0_FS':  (-6.0, -2.3),
    'logTb_FS':  (-2.523, -1.699),
    'logC_bg':   (-10.0, -4.0),
    't_rise':    (0.012, 0.050),
    'tau_rise':  (0.0, 0.020),
    't_off':     (0.035, 0.080),
    'logF_peak': (-8.0, -2.0),
}

def internal_flare_flux(t, F_peak, t_rise, tau_rise, t_off,
                        beta=BETA_INTERNAL):
    """Logistic turn-on followed by a curvature tail with t0=t_rise.

    The logistic is normalized at ``t_off`` so the two branches meet exactly
    at ``F_peak``.  ``tau_rise=0`` is the sharp, unresolved-rise limit.
    """
    t = np.asarray(t, dtype=float)
    if not (0.0 < t_rise < t_off):
        raise ValueError("Require 0 < t_rise < t_off")
    if F_peak <= 0.0 or tau_rise < 0.0 or beta <= -2.0:
        raise ValueError("Invalid flare amplitude or slope")
    active_rise = (t > 0.0) & (t <= t_off)
    tail = t > t_off
    flare = np.zeros_like(t)
    if tau_rise == 0.0:
        flare[active_rise] = np.where(
            t[active_rise] < t_rise, 0.0, F_peak)
    else:
        def logistic(z):
            out = np.empty_like(z, dtype=float)
            nonnegative = z >= 0.0
            out[nonnegative] = 1.0 / (1.0 + np.exp(-z[nonnegative]))
            exp_z = np.exp(z[~nonnegative])
            out[~nonnegative] = exp_z / (1.0 + exp_z)
            return out

        scale_off = logistic(np.array([(t_off - t_rise) / tau_rise]))[0]
        flare[active_rise] = F_peak * logistic(
            (t[active_rise] - t_rise) / tau_rise) / scale_off
    flare[tail] = F_peak * ((t[tail] - t_rise) /
                            (t_off - t_rise))**(-(2.0 + beta))
    return flare

def model_internal(t, F0_FS, tb_FS, C_bg, t_rise, tau_rise, t_off,
                   F_peak):
    fs = forward_shock_source_flux(t, F0_FS, tb_FS, P_INTERNAL)
    flare = internal_flare_flux(
        t, F_peak, t_rise, tau_rise, t_off)
    return fs + flare + C_bg

def log_prior_internal(theta):
    for value, (lo, hi) in zip(theta, PRIOR_INTERNAL.values()):
        if not (lo <= value <= hi):
            return -np.inf
    _, logTb_FS, _, t_rise, _, t_off, _ = theta
    if not (10**logTb_FS < t_rise < t_off < 0.10):
        return -np.inf
    return 0.0

def theta_to_params_internal(theta):
    (logF0_FS, logTb_FS, logC_bg, t_rise, tau_rise, t_off,
     logF_peak) = theta
    return (10**logF0_FS, 10**logTb_FS, 10**logC_bg,
            t_rise, tau_rise, t_off, 10**logF_peak)

def log_prob_internal(theta, x, y, yerr):
    lp = log_prior_internal(theta)
    if not np.isfinite(lp):
        return -np.inf
    try:
        ymod = windowed_eval(model_internal, x, theta_to_params_internal(theta))
    except Exception:
        return -np.inf
    if not np.all(np.isfinite(ymod)):
        return -np.inf
    return lp - 0.5 * np.sum(((y - ymod) / yerr)**2)

def theta0_internal_from_fs(theta_FS_best):
    theta0 = np.array([
        theta_FS_best[0], theta_FS_best[1], theta_FS_best[2],
        0.030, 0.002, 0.046, np.log10(3.5e-4),
    ])
    for j, (lo, hi) in enumerate(PRIOR_INTERNAL.values()):
        theta0[j] = np.clip(theta0[j], lo + 1e-4, hi - 1e-4)
    return theta0

def get_components_internal(t, p):
    (F0_FS, tb_FS, C_bg, t_rise, tau_rise, t_off, F_peak) = p
    fs = forward_shock_source_flux(t, F0_FS, tb_FS, P_INTERNAL)
    flare = internal_flare_flux(t, F_peak, t_rise, tau_rise, t_off)
    cbg = np.full_like(t, C_bg)
    return [
        ('Forward shock', fs, '--', 'goldenrod',
         f"tb={tb_FS*1440:.1f} min  decay=-{decay_alpha_from_p(P_INTERNAL):.2f}  "
         f"p={P_INTERNAL:.2f}"),
        ('Internal flare', flare, ':', 'crimson',
         f"F_p={F_peak*1e6:.2f} uJy  tau_r={tau_rise*1440:.2f} min  "
         f"tail=-{2+BETA_INTERNAL:.1f}"),
        ('TESS bg', cbg, '-.', 'mediumseagreen',
         f"C_bg={C_bg*1e6:.2f} uJy"),
    ]

# ---------------------------------------------------------------
# Generic emcee runner
# ---------------------------------------------------------------
_QUICK_NBURN  = 300
_QUICK_NPROD  = 2_000
_FULL_NBURN   = 1_000
_FULL_MAX     = 100_000   # hard cap; adaptive stopping usually exits much earlier
_CONV_CHECK   = 1_000     # check convergence every N production steps
_CONV_RATIO   = 50        # need chain_length > CONV_RATIO * tau_max
_CONV_DTAU    = 0.02      # and |Δτ/τ| < this

def run_emcee(x, y, yerr, theta0, prior_ranges, log_prob_fn, label,
              nwalkers=32, quick=False):
    ndim = len(theta0)
    pos  = theta0 + 1e-3 * np.random.randn(nwalkers, ndim)
    for i in range(nwalkers):
        for j, (lo, hi) in enumerate(prior_ranges.values()):
            pos[i, j] = np.clip(pos[i, j], lo + 1e-4, hi - 1e-4)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, args=(x, y, yerr))

    nburn = _QUICK_NBURN if quick else _FULL_NBURN
    print(f"[{label}] burn-in ({nburn} steps, {nwalkers} walkers)...")
    state = sampler.run_mcmc(pos, nburn, progress=False)
    sampler.reset()

    if quick:
        print(f"[{label}] production ({_QUICK_NPROD} steps, quick mode)...")
        sampler.run_mcmc(state, _QUICK_NPROD, progress=False)
    else:
        # Adaptive: stop when chain is long enough relative to autocorr time
        print(f"[{label}] production (adaptive, max {_FULL_MAX} steps)...")
        old_tau = np.inf
        for _ in sampler.sample(state, iterations=_FULL_MAX, progress=False):
            if sampler.iteration % _CONV_CHECK:
                continue
            try:
                tau  = sampler.get_autocorr_time(tol=0)
                tmax = float(np.max(tau))
                long_enough = sampler.iteration > _CONV_RATIO * tmax
                stable      = abs(old_tau - tmax) / tmax < _CONV_DTAU
                print(f"[{label}]   step {sampler.iteration:6d}  "
                      f"tau_max={tmax:.1f}  {'converged' if long_enough and stable else '...'}")
                if long_enough and stable:
                    break
                old_tau = tmax
            except Exception:
                pass

    af = np.mean(sampler.acceptance_fraction)
    try:    tau_rep = np.max(sampler.get_autocorr_time(quiet=True))
    except: tau_rep = None
    print(f"[{label}] done: steps={sampler.iteration}  "
          f"accept={af:.3f}  tau_max={tau_rep}")
    return sampler.get_chain(flat=True), sampler.get_log_prob(flat=True)

# ---------------------------------------------------------------
# Fit wrapper
# ---------------------------------------------------------------
def _fit_FS(xD, yD, yeD, quick=False):
    np.random.seed(42)
    return run_emcee(xD, yD, yeD, theta0_FS, PRIOR_FS, log_prob_FS,
                     'Forward shock', quick=quick)

def _fit_internal(xD, yD, yeD, theta0_internal, quick=False):
    np.random.seed(52)
    return run_emcee(xD, yD, yeD, theta0_internal, PRIOR_INTERNAL,
                     log_prob_internal, 'FS + internal flare', quick=quick)

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
def print_params(title, theta_best, param_names, quantiles,
                 labels_phys, scale, chi2_r):
    print(f"\n=== {title} ===  chi2_r={chi2_r:.3f}")
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
    'FS': dict(
        names=NAMES_FS,
        scale=[1e6, 1440, 1e6, 1, 1, 1],
        labels=[r'$F_0\ (\mu\mathrm{Jy})$', r'$t_b\ (\mathrm{min})$',
                r'$C_{\rm bg}\ (\mu\mathrm{Jy})$', r'$S_{\rm FS}$', r'$p$',
                r'$\alpha_{\rm FS,rise}$'],
    ),
    'INTERNAL': dict(
        names=NAMES_INTERNAL,
        scale=[1e6, 1440, 1e6, 1, 1440, 1, 1e6],
        labels=[r'$F_{0,\rm FS}\ (\mu\mathrm{Jy})$', r'$t_{b,\rm FS}\ (\mathrm{min})$',
                r'$C_{\rm bg}\ (\mu\mathrm{Jy})$', r'$t_r\ (\mathrm{d})$',
                r'$\tau_r\ (\mathrm{min})$', r'$t_{\rm off}\ (\mathrm{d})$',
                r'$F_p\ (\mu\mathrm{Jy})$'],
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
    corner_kwargs = dict(
        labels=cfg['labels'],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.2f',
        label_kwargs={'fontsize': 14, 'labelpad': 14},
        title_kwargs={'fontsize': 13},
    )
    try:
        with plt.rc_context({'text.usetex': True}):
            fig_c = corner.corner(samp, **corner_kwargs)
    except Exception:
        fig_c = corner.corner(samp, **corner_kwargs)
    for ax in fig_c.axes:
        ax.tick_params(axis='both', labelsize=9, pad=3)
        ax.xaxis.labelpad = 14
        ax.yaxis.labelpad = 14
    fig_c.savefig(f'GRB260207A_corner_{tag}.png', dpi=160, bbox_inches='tight')
    plt.close(fig_c)
    print(f"Saved: GRB260207A_corner_{tag}.png")

# ---------------------------------------------------------------
# Plotting helpers  (reference x_plot/y_plot/ye_plot as module globals
#                    set inside __main__ before these are ever called)
# ---------------------------------------------------------------
MODEL_COLOR = {'FS': 'royalblue', 'INTERNAL': 'navy'}

def add_data_to_ax(ax, mask_used, alpha_excl=0.4):
    pos          = y_plot > 0
    fitted_pos   = mask_used & pos
    excluded_pos = ~mask_used & pos & (x_plot < 13)
    fitted_nonpos = mask_used & ~pos
    excluded_nonpos = ~mask_used & ~pos & (x_plot < 13)
    if excluded_pos.any():
        ax.errorbar(x_plot[excluded_pos], y_plot[excluded_pos],
                    yerr=ye_plot[excluded_pos], fmt='.', color='gray',
                    markersize=2, elinewidth=0.3, alpha=alpha_excl,
                    capsize=0, zorder=1.5, label='Excluded')
    ax.errorbar(x_plot[fitted_pos], y_plot[fitted_pos],
                yerr=ye_plot[fitted_pos], fmt='.', color='black',
                markersize=3, elinewidth=0.4, alpha=0.8, capsize=0,
                zorder=2, label='Fitted')
    floor_trans = blended_transform_factory(ax.transData, ax.transAxes)
    if excluded_nonpos.any():
        ax.scatter(x_plot[excluded_nonpos], np.full(excluded_nonpos.sum(), 0.025),
                   marker='v', s=18, color='gray', alpha=alpha_excl,
                   transform=floor_trans, clip_on=False, zorder=3,
                   label='Excluded <= 0')
    if fitted_nonpos.any():
        ax.scatter(x_plot[fitted_nonpos], np.full(fitted_nonpos.sum(), 0.025),
                   marker='v', s=22, color='black', alpha=0.8,
                   transform=floor_trans, clip_on=False, zorder=4,
                   label='Fitted <= 0')
    ax.errorbar(t_master, F_master, yerr=eF_master, fmt='D',
                color='mediumseagreen', markersize=8, elinewidth=1.4,
                capsize=4, markeredgecolor='k', markeredgewidth=0.6,
                label=f'MASTER T+{t_master*1440:.1f} min')

def format_main_ax(ax):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-4, 13)
    vis = (x_plot >= 1e-4) & (x_plot <= 13) & (y_plot > 0)
    ax.set_ylim(ye_plot[vis].min() * 0.3, max(y_plot[vis].max(), F_master) * 3)
    ax.grid(True, which='both', alpha=0.2, lw=0.5)

def add_minutes_axis(ax):
    ax_min = ax.twiny()
    ax_min.set_xscale('log'); ax_min.set_xlim(ax.get_xlim())
    min_ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 17000]
    ax_min.set_xticks([m/1440. for m in min_ticks])
    ax_min.set_xticklabels([str(m) for m in min_ticks], fontsize=8)
    ax_min.set_xlabel('Minutes post-burst', fontsize=10)

def plot_residuals(ax_res, mask_used, model_fn, params_best, line_color):
    visible  = (x_plot >= 1e-4) & (x_plot <= 13)
    pos_fit  = mask_used & visible
    pos_excl = ~mask_used & visible

    ax_res.axhline(0,  color=line_color, lw=1.2, zorder=5)
    ax_res.axhline( 3, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax_res.axhline(-3, color='gray', lw=0.8, ls='--', alpha=0.5)

    groups = []
    all_resid = []
    for mask, color, markersize, alpha, zorder, label in [
        (pos_excl, 'gray', 2, 0.45, 1.5, 'Excluded'),
        (pos_fit, 'black', 3, 0.8, 2, 'Fitted'),
    ]:
        if not mask.any():
            continue
        ymod = model_fn(x_plot[mask], *params_best)
        resid = (y_plot[mask] - ymod) / ye_plot[mask]
        finite = np.isfinite(resid)
        if not finite.any():
            continue
        x_resid = x_plot[mask][finite]
        resid = resid[finite]
        all_resid.append(resid)
        groups.append((x_resid, resid, color, markersize, alpha, zorder, label))

    if all_resid:
        resid_abs = np.abs(np.concatenate(all_resid))
        finite = resid_abs[np.isfinite(resid_abs)]
        ymax = max(7.0, min(30.0, 1.2 * np.percentile(finite, 99))) if len(finite) else 7.0
    else:
        ymax = 7.0

    for x_resid, resid, color, markersize, alpha, zorder, label in groups:
        in_range = (resid >= -ymax) & (resid <= ymax)
        high = resid > ymax
        low = resid < -ymax
        if in_range.any():
            ax_res.errorbar(x_resid[in_range], resid[in_range],
                            yerr=np.ones_like(resid[in_range]),
                            fmt='.', color=color, markersize=markersize,
                            elinewidth=0.3, alpha=alpha, capsize=0,
                            zorder=zorder, label=label)
        if high.any():
            ax_res.scatter(x_resid[high], np.full(high.sum(), 0.96 * ymax),
                           marker='^', s=22 if markersize > 2 else 18,
                           color=color, alpha=alpha, zorder=zorder + 0.2)
        if low.any():
            ax_res.scatter(x_resid[low], np.full(low.sum(), -0.96 * ymax),
                           marker='v', s=22 if markersize > 2 else 18,
                           color=color, alpha=alpha, zorder=zorder + 0.2)
    ax_res.set_xscale('log'); ax_res.set_xlim(1e-4, 13); ax_res.set_ylim(-ymax, ymax)
    ax_res.set_xlabel('Days post-burst', fontsize=11)
    ax_res.set_ylabel('Residuals (σ)', fontsize=10)
    ax_res.legend(fontsize=7, loc='upper right')
    ax_res.grid(True, which='both', alpha=0.2, lw=0.5)

def save_internal_plot(mask_internal_fit, chain, lp, chi2_r):
    p_best = theta_to_params_internal(chain[np.argmax(lp)])
    t_model = np.logspace(np.log10(2e-4), np.log10(13), 1500)
    color = MODEL_COLOR['INTERNAL']

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(8, 9),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    add_data_to_ax(ax, mask_internal_fit)

    rng = np.random.default_rng(52)
    for i in rng.choice(len(chain), min(200, len(chain)), replace=False):
        y_samp = model_internal(t_model, *theta_to_params_internal(chain[i]))
        ax.plot(t_model, np.where(y_samp > 0, y_samp, np.nan),
                '-', color=color, lw=0.3, alpha=0.04, zorder=4)

    y_best = model_internal(t_model, *p_best)
    ax.plot(t_model, np.where(y_best > 0, y_best, np.nan),
            '-', color=color, lw=2.0, zorder=6,
            label=f'Best fit  chi2_r={chi2_r:.2f}')
    for (clabel, cy, ls, ccolor, desc) in get_components_internal(t_model, p_best):
        ax.plot(t_model, np.where(cy > 0, cy, np.nan),
                color=ccolor, ls=ls, lw=1.4, zorder=5, label=f'{clabel}: {desc}')

    for tx, label in [(p_best[3], r'$t_r$'),
                      (p_best[5], r'$t_{\rm off}$')]:
        ax.axvline(tx, color='dimgray', ls=':', lw=0.9, alpha=0.75,
                   label=f'{label}={tx:.4f} d')
    format_main_ax(ax)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel('Flux density (Jy)', fontsize=11)
    ax.set_title('Forward shock plus piecewise internal flare', fontsize=10)
    ax.legend(fontsize=6.2, loc='best')
    add_minutes_axis(ax)
    plot_residuals(ax_res, mask_internal_fit, model_internal, p_best, color)

    plt.suptitle('GRB 260207A', fontsize=14, y=0.995)
    plt.savefig('GRB260207A_emcee_internal_flare.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: GRB260207A_emcee_internal_flare.png")

def save_background_plot(bg_models):
    """Dedicated plot of the TESS background constant.

    bg_models: list of (tag, chain, lp, theta_fn, c_bg_index).
    Uses module globals x_plot / y_plot / ye_plot (full mask_plot range).
    Shows linear x-axis so the pre-burst region and the full background
    level are visible. y-axis uses symlog so both the bright afterglow and
    the low-level background are legible simultaneously.
    """
    t_bg = np.linspace(x_plot.min() - 0.02, x_plot.max() + 0.3, 2000)

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- full dataset (linear x, symlog y) -----------------------
    pos = y_plot > 0
    if (~pos).any():
        ax.errorbar(x_plot[~pos], y_plot[~pos], yerr=ye_plot[~pos],
                    fmt='.', color='lightgray', markersize=2, elinewidth=0.3,
                    alpha=0.3, capsize=0, zorder=1)
    ax.errorbar(x_plot[pos], y_plot[pos], yerr=ye_plot[pos],
                fmt='.', color='black', markersize=2, elinewidth=0.3,
                alpha=0.5, capsize=0, zorder=2, label='Data')

    # --- posterior samples + best-fit constant background --------
    rng = np.random.default_rng(0)
    for tag, chain, lp, theta_fn, c_bg_index in bg_models:
        color = MODEL_COLOR.get(tag, MODEL_COLOR['FS'])
        idx   = rng.choice(len(chain), min(300, len(chain)), replace=False)
        for i in idx:
            p   = theta_fn(chain[i])
            ax.axhline(p[c_bg_index], color=color, lw=0.3, alpha=0.04, zorder=3)
        p_best = theta_fn(chain[np.argmax(lp)])
        C_bg   = p_best[c_bg_index]
        ax.axhline(C_bg, color=color, lw=2, zorder=5,
                   label=f'{tag}  C_bg={C_bg*1e6:.2f} µJy')

    # --- cosmetic -------------------------------------------------
    ax.axvline(0, color='gray', ls='--', lw=0.8, alpha=0.6, zorder=4)
    ax.axvspan(x_plot.min() - 0.02, 0, color='steelblue', alpha=0.04, zorder=0)
    ax.text(0.003, 0.97, 'trigger', transform=ax.transAxes,
            va='top', fontsize=8, color='gray')

    # linthresh at rough noise floor so background detail is in the linear region
    linthresh = max(float(np.median(ye_plot)), 1e-7)
    ax.set_yscale('symlog', linthresh=linthresh)
    ax.set_xlim(x_plot.min() - 0.02, x_plot.max() + 0.3)
    ax.set_xlabel('Days post-burst', fontsize=11)
    ax.set_ylabel('Flux density (Jy)', fontsize=11)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, which='both', alpha=0.2, lw=0.5)
    plt.suptitle('GRB 260207A — TESS Background (constant)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('GRB260207A_tess_bg_internal.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: GRB260207A_tess_bg_internal.png")


# ===============================================================
# Main
# ===============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nomodel', action='store_true', help='Plot data only')
    parser.add_argument('--quick',   action='store_true',
                        help=f'Fast test run (nburn={_QUICK_NBURN}, nprod={_QUICK_NPROD}); '
                             f'normal mode uses adaptive stopping')
    args = parser.parse_args()

    rawdata   = np.loadtxt('lc_GRB260207A_cand41148_geo')
    # The light-curve timestamps are cadence starts; model and plot at midtimes.
    x_all     = rawdata[:, 0] + HALF_C - trigger_tjd
    y_all     = -rawdata[:, 1] / (200 * 0.8 * 0.99)
    yerr_all  =  rawdata[:, 2] / (200 * 0.8 * 0.99)
    zp        = 2416 * 10**(-0.4 * 20.44)
    flux_all  = y_all    * zp
    eflux_all = yerr_all * zp

    # Calibrate an additive photometric noise floor from the full one-day
    # pre-burst interval, where the source model is a constant background.
    # This captures cadence-level confusion and subtraction scatter associated
    # with unresolved sources and background structure in the large TESS pixels.
    pre = (x_all >= PREBURST_MIN) & (x_all <= PREBURST_MAX)
    _, _, chi2_formal, dof_bg = fit_constant_background(
        flux_all[pre], eflux_all[pre])
    jitter_flux = estimate_additive_jitter(flux_all[pre], eflux_all[pre])
    jitter_cps = jitter_flux / zp
    yerr_eff_all = np.sqrt(yerr_all**2 + jitter_cps**2)
    eflux_eff_all = yerr_eff_all * zp
    bg_eff, bg_eff_err, chi2_eff, _ = fit_constant_background(
        flux_all[pre], eflux_eff_all[pre])
    print("Pre-burst additive-jitter calibration:")
    print(f"  N={pre.sum()}, interval=[{PREBURST_MIN:.1f}, {PREBURST_MAX:.1f}] d")
    print(f"  constant background={bg_eff*1e6:.3f} +/- {bg_eff_err*1e6:.3f} uJy")
    print(f"  formal chi2/dof={chi2_formal:.2f}/{dof_bg}={chi2_formal/dof_bg:.3f}")
    print(f"  jitter={jitter_flux*1e6:.3f} uJy ({jitter_cps:.3f} counts/s)")
    print(f"  effective chi2/dof={chi2_eff:.2f}/{dof_bg}={chi2_eff/dof_bg:.3f}")

    # Include 1 d (24 h) pre-burst for the background constraint, out to 12 d post-burst.
    mask_plot = (x_all >= PREBURST_MIN) & (x_all <= 12.0)
    x_plot  = x_all[mask_plot]
    y_plot  = flux_all[mask_plot]
    ye_plot = eflux_eff_all[mask_plot]

    if args.nomodel:
        # --- Extended mask: first cadence onward through end of data
        mask_ext = (x_all > -HALF_C) & (x_all <= x_all.max())
        xp   = x_all[mask_ext]
        yp   = y_all[mask_ext]
        yep  = yerr_eff_all[mask_ext]

        # --- Pre-burst baseline: weighted mean over the full interval t in [-1 d, 0]
        wt      = 1.0 / yerr_eff_all[pre]**2
        bkg_val = np.sum(wt * y_all[pre]) / np.sum(wt)
        bkg_err = np.sqrt(1.0 / np.sum(wt))
        print(f"Pre-burst baseline: {bkg_val:.4f} +/- {bkg_err:.4f} cps  (N={pre.sum()})")

        # --- Log-bin all data from 0.07 d onwards
        BIN_START = 0.07
        nbins     = 20
        sel       = xp >= BIN_START
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

        pos = yp > 0
        nonpos = ~pos

        if pos.any():
            ax.errorbar(xp[pos], yp[pos], yerr=yep[pos],
                    fmt='.', color='black', markersize=3, elinewidth=0.4,
                    alpha=0.8, capsize=0, zorder=2, label='TESS')
        floor_trans = blended_transform_factory(ax.transData, ax.transAxes)
        if nonpos.any():
            ax.scatter(xp[nonpos], np.full(nonpos.sum(), 0.025),
                       marker='v', s=18, color='black', alpha=0.5,
                       transform=floor_trans, clip_on=False, zorder=3,
                       label='TESS <= 0')

        ax.errorbar(t_master, F_master, yerr=eF_master, fmt='D',
                    color='mediumseagreen', markersize=8, elinewidth=1.4,
                    capsize=4, markeredgecolor='k', markeredgewidth=0.6,
                    label=f'MASTER T+{t_master*1440:.1f} min')

        # Log-binned data overlay
        if len(x_binned) > 0:
            bin_pos = y_binned > 0
            bin_nonpos = ~bin_pos
            if bin_pos.any():
                ax.errorbar(x_binned[bin_pos], y_binned[bin_pos],
                            yerr=ye_binned[bin_pos],
                            fmt='o', color='crimson', markersize=6,
                            elinewidth=1.2, capsize=3, zorder=6,
                            label=f'Log-binned ($\\geq${BIN_START} d)')
            if bin_nonpos.any():
                ax.scatter(x_binned[bin_nonpos], np.full(bin_nonpos.sum(), 0.055),
                           marker='v', s=38, color='crimson', alpha=0.85,
                           transform=floor_trans, clip_on=False, zorder=7,
                           label=f'Log-binned <= 0 ($\\geq${BIN_START} d)')

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
        vis = (xp > 1e-4) & (yp > 0)
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

    # Fit only through 0.02 d post-burst, keeping pre-burst data for background.
    mask_fit = (x_plot >= PREBURST_MIN) & (x_plot <= FS_FIT_MAX)
    xD = x_plot[mask_fit]; yD = y_plot[mask_fit]; yeD = ye_plot[mask_fit]
    print(f"N_fit = {len(xD)},  t in [{xD.min()*1440:.2f} min, {xD.max():.3f} d]")

    mode_str = "quick" if args.quick else "adaptive (convergence-based)"
    print(f"\nLaunching forward-shock fit ({mode_str})...")
    chainFS, lpFS = _fit_FS(xD, yD, yeD, args.quick)

    # -----------------------------------------------------------
    # Summarize
    # -----------------------------------------------------------
    thetaFS, pFS, qFS = summarize(chainFS, lpFS, theta_to_params_FS, NAMES_FS)

    N = len(xD)
    npar = len(NAMES_FS)
    chi2_FS  = -2*np.max(lpFS)
    chi2_rFS = chi2_FS / (N - npar)
    decay_best = decay_alpha_from_p(pFS[4])

    print_params('FORWARD SHOCK MODEL', thetaFS, NAMES_FS, qFS,
                 ['F0 (uJy)', 'tb (min)', 'C_bg (uJy)', 'S_FS', 'p',
                  'alpha_rise_FS'],
                 [1e6, 1440, 1e6, 1, 1, 1],
                 chi2_rFS)
    print(f"  {'rise slope':>24s}:    {pFS[5]: .3f}   (fitted)")
    print(f"  {'decay slope':>24s}:    {-decay_best: .3f}   "
          f"(fixed by best-fit p as -3(p-1)/4)")
    ALPHA_RISE_INTERNAL = fs_rise_posterior_median(chainFS)
    S_AB = fs_smoothness_posterior_median(chainFS)
    P_INTERNAL = fs_p_posterior_median(chainFS)
    print(f"  {'rise for internal':>24s}:     {ALPHA_RISE_INTERNAL:.3f}   "
          "(fixed at FS posterior median)")
    print(f"  {'S_FS for internal':>24s}:     {S_AB:.3f}   "
          "(fixed at FS posterior median)")
    print(f"  {'p for internal':>24s}:     {P_INTERNAL:.3f}   "
          "(fixed at FS posterior median)")

    # -----------------------------------------------------------
    # Corner plot
    # -----------------------------------------------------------
    save_corner('FS', chainFS)

    # -----------------------------------------------------------
    # Model figure (main panel + residuals)
    # -----------------------------------------------------------
    t_model = np.logspace(np.log10(2e-4), np.log10(13), 1500)
    p_best = theta_to_params_FS(chainFS[np.argmax(lpFS)])
    color  = MODEL_COLOR['FS']

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(8, 9),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    add_data_to_ax(ax, mask_fit)

    rng = np.random.default_rng(42)
    for i in rng.choice(len(chainFS), min(200, len(chainFS)), replace=False):
        ax.plot(t_model, model_FS(t_model, *theta_to_params_FS(chainFS[i])),
                '-', color=color, lw=0.3, alpha=0.04, zorder=4)

    ax.plot(t_model, model_FS(t_model, *p_best), '-', color=color, lw=2.0, zorder=6,
            label=f'Best fit  chi2_r={chi2_rFS:.2f}')
    for (clabel, cy, ls, ccolor, desc) in get_components_FS(t_model, p_best):
        ax.plot(t_model, np.where(cy > 0, cy, np.nan),
                color=ccolor, ls=ls, lw=1.4, zorder=5, label=f'{clabel}: {desc}')

    format_main_ax(ax)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel('Flux density (Jy)', fontsize=11)
    ax.set_title(f'Forward shock only: rise t$^{{{p_best[5]:.2f}}}$, '
                 'decay t$^{-3(p-1)/4}$', fontsize=10)
    ax.legend(fontsize=6.5, loc='lower left')
    add_minutes_axis(ax)
    plot_residuals(ax_res, mask_fit, model_FS, p_best, color)

    plt.suptitle('GRB 260207A', fontsize=14, y=0.995)
    plt.savefig('GRB260207A_emcee_forward_shock.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: GRB260207A_emcee_forward_shock.png")

    # Fit the full trigger-frame forward shock plus piecewise flare.
    theta0_internal = theta0_internal_from_fs(thetaFS)
    mask_internal_fit = ((x_plot >= PREBURST_MIN) &
                         (x_plot <= INTERNAL_FIT_MAX))
    xI = x_plot[mask_internal_fit]
    yI = y_plot[mask_internal_fit]
    yeI = ye_plot[mask_internal_fit]
    print(f"\nN_internal_fit = {len(xI)},  t in "
          f"[{xI.min()*1440:.2f} min, {xI.max():.3f} d]")
    print(f"Launching FS + internal-flare fit ({mode_str})...")
    chainInternal, lpInternal = _fit_internal(
        xI, yI, yeI, theta0_internal, args.quick)

    thetaInternal, pInternal, qInternal = summarize(
        chainInternal, lpInternal, theta_to_params_internal, NAMES_INTERNAL)
    ymod_internal = windowed_eval(model_internal, xI, pInternal)
    chi2_internal = np.sum(((yI - ymod_internal) / yeI)**2)
    chi2_rInternal = chi2_internal / (len(xI) - len(NAMES_INTERNAL))
    print_params('FORWARD SHOCK + INTERNAL FLARE MODEL',
                 thetaInternal, NAMES_INTERNAL, qInternal,
                 ['F0_FS (uJy)', 'tb_FS (min)', 'C_bg (uJy)',
                  't_rise (d)', 'tau_rise (min)', 't_off (d)',
                  'F_peak (uJy)'],
                 [1e6, 1440, 1e6, 1, 1440, 1, 1e6],
                 chi2_rInternal)
    print(f"  {'p':>24s}: {P_INTERNAL:9.3f}   (fixed at FS posterior median)")
    print(f"  {'beta':>24s}: {BETA_INTERNAL:9.3f}   (fixed)")
    print(f"  {'curvature slope':>24s}: {-(2+BETA_INTERNAL):9.3f}")
    print(f"  {'rise timescale (min)':>24s}: {pInternal[4]*1440:9.3f}")
    print(f"  {'t_off-t_r (min)':>24s}: "
          f"{(pInternal[5]-pInternal[3])*1440:9.3f}")

    save_corner('INTERNAL', chainInternal)
    save_internal_plot(mask_internal_fit, chainInternal, lpInternal,
                       chi2_rInternal)

    # -----------------------------------------------------------
    # TESS background plot
    # -----------------------------------------------------------
    save_background_plot([
        ('Forward shock', chainFS, lpFS, theta_to_params_FS, 2),
        ('INTERNAL', chainInternal, lpInternal, theta_to_params_internal, 2),
    ])
