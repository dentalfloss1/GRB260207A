"""
GRB260207A — emcee fit of a refreshed forward-shock model.

The early forward shock is the same smoothly broken power law used by
internal_model.py. The full light curve is then fit with a single forward shock
whose source flux is multiplied by a monotonic energy-injection history:

    F_model(t) = F_FS,0(t) * [E(t) / E0]**((p + 3) / 4) + C_bg

The background is never multiplied by the energy factor.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import emcee
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
HALF_WIN   = CADENCE / 2.0
WIN_THRESH = 5.0 / 1440      # 5 min in days
FS_FIT_MAX = 0.02            # days post-burst
SHIFT_FIT_MAX = 0.08         # days after computed shifted origin for DSBPL fit
SHIFT_PLOT_MAX = 1.0         # days after computed shifted origin for DSBPL plot
COMBINED_FIT_MAX = 1.0       # days post-burst for combined FS+DSBPL fit

# ---------------------------------------------------------------
# Base model functions (S passed explicitly)
# ---------------------------------------------------------------
def sbpl(t, F0, tb, a1, a2, S):
    return F0 * (t/tb)**(-a1) * (0.5*(1 + (t/tb)**(1/S)))**(-(a2-a1)*S)

def dsbpl(t, F0, tb1, tb2, a1, a2, a3, S):
    f1 = (0.5*(1 + (t/tb1)**(1/S)))**(-(a2-a1)*S)
    f2 = (0.5*(1 + (t/tb2)**(1/S)))**(-(a3-a2)*S)
    return F0 * (t/tb1)**(-a1) * f1 * f2

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

# ===============================================================
# Forward-shock model — SBPL + TESS-bg constant, S=0.1
# theta: [logF0, logTb, p, logC_bg]
# ===============================================================
FS_RISE_ALPHA = -0.5

def decay_alpha_from_p(p):
    return 3.0 * (p - 1.0) / 4.0

def forward_shock_source_flux(t, F0, tb, p):
    """Source-only forward-shock flux; zero before the trigger."""
    t = np.asarray(t, dtype=float)
    pos = t > 0
    ts = np.where(pos, t, 1e-10)
    a2 = decay_alpha_from_p(p)
    return np.where(pos, sbpl(ts, F0, tb, FS_RISE_ALPHA, a2, S_AB), 0.0)

def model_FS(t, F0, tb, p, C_bg):
    return forward_shock_source_flux(t, F0, tb, p) + C_bg

PRIOR_FS = {
    'logF0':   (-6.0, -2.3),
    'logTb':   (-2.523, -1.699),  # 3–30 min, matching the original P1 range
    'p':       ( 2.1,  2.5),
    'logC_bg': (-10.0, -2.0),
}
NAMES_FS = list(PRIOR_FS.keys())

def log_prior_FS(theta):
    for v, (lo, hi) in zip(theta, PRIOR_FS.values()):
        if not (lo <= v <= hi): return -np.inf
    return 0.0

def theta_to_params_FS(theta):
    logF0, logTb, p, logC_bg = theta
    return (10**logF0, 10**logTb, p, 10**logC_bg)

def log_prob_FS(theta, x, y, yerr):
    lp = log_prior_FS(theta)
    if not np.isfinite(lp): return -np.inf
    params = theta_to_params_FS(theta)
    try:    ymod = windowed_eval(model_FS, x, params)
    except: return -np.inf
    if not np.all(np.isfinite(ymod)): return -np.inf
    return lp - 0.5 * np.sum(((y - ymod) / yerr)**2)

def get_components_FS(t, p):
    # p: F0, tb, electron-index p, C_bg
    a2  = decay_alpha_from_p(p[2])
    fs  = forward_shock_source_flux(t, p[0], p[1], p[2])
    cbg = np.full_like(t, p[3])
    return [
        ('Forward shock', fs, '--', 'goldenrod',
         f"tb={p[1]*1440:.1f} min  rise=+0.50  decay=-{a2:.2f}  p={p[2]:.2f}"),
        ('TESS bg', cbg, '-.', 'mediumseagreen',
         f"C_bg={p[3]*1e6:.2f} µJy"),
    ]

theta0_FS = np.array([np.log10(7e-4), np.log10(10/1440), 2.25, -4.5])

# ===============================================================
# Shifted excess model — DSBPL only
# The forward-shock subtraction already removes the fitted C_bg, so adding a
# second constant here double-counts the background.
# theta: [logF0, logTb1, logTb2, a1, a2, a3]
# ===============================================================
S_SHIFT = 0.02

def model_shifted_excess(t, F0, tb1, tb2, a1, a2, a3):
    pos = t > 0
    ts = np.where(pos, t, 1e-10)
    comp = np.where(pos, dsbpl(ts, F0, tb1, tb2, a1, a2, a3, S_SHIFT), 0.0)
    return comp

PRIOR_SHIFT = {
    'logF0':   (-6.5, -2.5),
    'logTb1':  (np.log10(0.01), np.log10(0.02)),
    'logTb2':  (np.log10(0.02), np.log10(0.05)),
    'a1':      (-8.0,  0.0),
    'a2':      (-2.0,  5.0),
    'a3':      ( 0.2, 12.0),
}
NAMES_SHIFT = list(PRIOR_SHIFT.keys())

def log_prior_shift(theta):
    for v, (lo, hi) in zip(theta, PRIOR_SHIFT.values()):
        if not (lo <= v <= hi): return -np.inf
    if theta[2] <= theta[1]: return -np.inf
    return 0.0

def theta_to_params_shift(theta):
    logF0, logTb1, logTb2, a1, a2, a3 = theta
    return (10**logF0, 10**logTb1, 10**logTb2, a1, a2, a3)

def log_prob_shift(theta, x, y, yerr):
    lp = log_prior_shift(theta)
    if not np.isfinite(lp): return -np.inf
    params = theta_to_params_shift(theta)
    ymod = model_shifted_excess(x, *params)
    if not np.all(np.isfinite(ymod)): return -np.inf
    return lp - 0.5 * np.sum(((y - ymod) / yerr)**2)

def get_components_shift(t, p):
    pos = t > 0
    ts = np.where(pos, t, 1e-10)
    comp = np.where(pos, dsbpl(ts, p[0], p[1], p[2], p[3], p[4], p[5], S_SHIFT), 0.0)
    return [
        ('DSBPL excess', comp, '--', 'darkorange',
         f"tb=({p[1]:.3f},{p[2]:.3f}) d  alpha=({p[3]:.2f},{p[4]:.2f},{p[5]:.2f})"),
    ]

theta0_SHIFT = np.array([np.log10(3e-4), np.log10(0.015), np.log10(0.035),
                         -3.0, 1.0, 3.0])

# ===============================================================
# Combined trigger-frame model — FS SBPL + DSBPL(t - t0_D) + C_bg
# theta: [logF0_FS, logTb_FS, p, logF0_D, t0_D, logTauB1_D,
#         logTauB2_D, a1_D, a2_D, a3_D, logC_bg]
# ===============================================================
COMBINED_T0_HALF_WIDTH = 0.02
PRIOR_COMBINED = {
    'logF0_FS':  (-6.0, -2.3),
    'logTb_FS':  (-2.523, -1.699),
    'p':         ( 2.1,  2.5),
    'logF0_D':   (-8.0, -2.0),
    't0_D':      ( 0.0,  0.08),
    'logTauB1_D': (np.log10(0.003), np.log10(0.08)),
    'logTauB2_D': (np.log10(0.006), np.log10(0.20)),
    'a1_D':      (-8.0,  0.0),
    'a2_D':      (-2.0,  5.0),
    'a3_D':      ( 0.2, 12.0),
    'logC_bg':   (-10.0, -2.0),
}
NAMES_COMBINED = list(PRIOR_COMBINED.keys())

def set_combined_t0_prior(t0_center):
    lo = max(0.0, float(t0_center) - COMBINED_T0_HALF_WIDTH)
    hi = float(t0_center) + COMBINED_T0_HALF_WIDTH
    PRIOR_COMBINED['t0_D'] = (lo, hi)

def model_combined(t, F0_FS, tb_FS, p_FS, F0_D, t0_D, tau_b1_D, tau_b2_D,
                   a1_D, a2_D, a3_D, C_bg):
    fs = model_FS(t, F0_FS, tb_FS, p_FS, C_bg)
    tau = t - t0_D
    pos = tau > 0
    tau_eval = np.where(pos, tau, 1e-10)
    ds = np.where(pos, dsbpl(tau_eval, F0_D, tau_b1_D, tau_b2_D,
                             a1_D, a2_D, a3_D, S_SHIFT), 0.0)
    return fs + ds

def log_prior_combined(theta):
    for v, (lo, hi) in zip(theta, PRIOR_COMBINED.values()):
        if not (lo <= v <= hi): return -np.inf
    if theta[6] <= theta[5]: return -np.inf
    return 0.0

def theta_to_params_combined(theta):
    (logF0_FS, logTb_FS, p_FS, logF0_D, t0_D, logTauB1_D,
     logTauB2_D, a1_D, a2_D, a3_D, logC_bg) = theta
    return (10**logF0_FS, 10**logTb_FS, p_FS,
            10**logF0_D, t0_D, 10**logTauB1_D, 10**logTauB2_D,
            a1_D, a2_D, a3_D, 10**logC_bg)

def log_prob_combined(theta, x, y, yerr):
    lp = log_prior_combined(theta)
    if not np.isfinite(lp): return -np.inf
    params = theta_to_params_combined(theta)
    try:    ymod = windowed_eval(model_combined, x, params)
    except: return -np.inf
    if not np.all(np.isfinite(ymod)): return -np.inf
    return lp - 0.5 * np.sum(((y - ymod) / yerr)**2)

def theta0_combined_from_fits(theta_FS_best, p_shift, t0_D):
    F0_D, tau_b1_D, tau_b2_D, a1_D, a2_D, a3_D = p_shift
    theta0 = np.array([
        theta_FS_best[0],
        theta_FS_best[1],
        theta_FS_best[2],
        np.log10(F0_D),
        t0_D,
        np.log10(tau_b1_D),
        np.log10(tau_b2_D),
        a1_D,
        a2_D,
        a3_D,
        theta_FS_best[3],
    ])
    for j, (lo, hi) in enumerate(PRIOR_COMBINED.values()):
        theta0[j] = np.clip(theta0[j], lo + 1e-4, hi - 1e-4)
    if theta0[6] <= theta0[5]:
        theta0[6] = min(PRIOR_COMBINED['logTauB2_D'][1] - 1e-4, theta0[5] + 1e-3)
    return theta0

def get_components_combined(t, p):
    F0_FS, tb_FS, p_FS, F0_D, t0_D, tau_b1_D, tau_b2_D, a1_D, a2_D, a3_D, C_bg = p
    pos_fs = t > 0
    ts = np.where(pos_fs, t, 1e-10)
    fs = np.where(pos_fs, sbpl(ts, F0_FS, tb_FS, FS_RISE_ALPHA,
                               decay_alpha_from_p(p_FS), S_AB), 0.0)
    tau = t - t0_D
    pos_d = tau > 0
    tau_eval = np.where(pos_d, tau, 1e-10)
    ds = np.where(pos_d, dsbpl(tau_eval, F0_D, tau_b1_D, tau_b2_D,
                             a1_D, a2_D, a3_D, S_SHIFT), 0.0)
    cbg = np.full_like(t, C_bg)
    return [
        ('Forward shock', fs, '--', 'goldenrod',
         f"tb={tb_FS*1440:.1f} min  rise=+0.50  decay=-{decay_alpha_from_p(p_FS):.2f}"),
        ('Second peak (combined)', ds, '--', 'darkorange',
         f"t0={t0_D:.4f} d  tau_b=({tau_b1_D:.3f},{tau_b2_D:.3f}) d  alpha=({a1_D:.2f},{a2_D:.2f},{a3_D:.2f})"),
        ('TESS bg', cbg, '-.', 'mediumseagreen',
         f"C_bg={C_bg*1e6:.2f} uJy"),
    ]

# ===============================================================
# Refreshed forward-shock model — FS source * [E(t)/E0]^g + C_bg
# theta: [logF0_FS, logTb_FS, p, logC_bg, t_start, t_break, t_end,
#         log_r1, log_r2]
# log_r1/log_r2 are natural logs; other log* parameters are log10.
# ===============================================================
NAMES_INJECTION = [
    'logF0_FS', 'logTb_FS', 'p', 'logC_bg',
    't_start', 't_break', 't_end', 'log_r1', 'log_r2',
]

PRIOR_INJECTION = {
    'logF0_FS': (-6.0, -2.3),
    'logTb_FS': (-2.523, -1.699),
    'p': (2.1, 2.5),
    'logC_bg': (-10.0, -2.0),
    't_start': (0.012, 0.025),
    't_break': (0.025, 0.040),
    't_end': (0.038, 0.060),
    'log_r1': (np.log(1.0001), np.log(5.0)),
    'log_r2': (np.log(1.0), np.log(2.0)),
}

def energy_flux_index(p):
    """ISM, slow cooling, nu_m < nu_TESS < nu_c."""
    return (p + 3.0) / 4.0

def energy_ratio_piecewise(t, t_start, t_break, t_end, log_r1, log_r2):
    """Return E(t) / E0 for a two-stage monotonic injection history."""
    t = np.asarray(t, dtype=float)
    if not (0.0 < t_start < t_break < t_end):
        raise ValueError("Require 0 < t_start < t_break < t_end")

    r1 = np.exp(log_r1)
    r2 = np.exp(log_r2)
    ratio = np.ones_like(t)

    mask1 = (t >= t_start) & (t < t_break)
    if np.any(mask1):
        u1 = np.log(t[mask1] / t_start) / np.log(t_break / t_start)
        ratio[mask1] = np.exp(log_r1 * u1)

    mask2 = (t >= t_break) & (t < t_end)
    if np.any(mask2):
        u2 = np.log(t[mask2] / t_break) / np.log(t_end / t_break)
        ratio[mask2] = r1 * np.exp(log_r2 * u2)

    ratio[t >= t_end] = r1 * r2
    return ratio

def model_injection(t, F0_FS, tb_FS, p_FS, C_bg,
                    t_start, t_break, t_end, log_r1, log_r2):
    f_fs0 = forward_shock_source_flux(t, F0_FS, tb_FS, p_FS)
    e_ratio = energy_ratio_piecewise(t, t_start, t_break, t_end, log_r1, log_r2)
    return f_fs0 * e_ratio**energy_flux_index(p_FS) + C_bg

def log_prior_injection(theta):
    for v, (lo, hi) in zip(theta, PRIOR_INJECTION.values()):
        if not (lo <= v <= hi):
            return -np.inf

    (logF0_FS, logTb_FS, p_FS, logC_bg,
     t_start, t_break, t_end, log_r1, log_r2) = theta
    tb = 10**logTb_FS

    if not (tb < t_start < t_break < t_end < 0.10):
        return -np.inf

    e1 = log_r1 / np.log(t_break / t_start)
    e2 = log_r2 / np.log(t_end / t_break)
    if not (0.0 <= e2 < e1 < 3.0):
        return -np.inf

    return 0.0

def theta_to_params_injection(theta):
    (logF0_FS, logTb_FS, p_FS, logC_bg,
     t_start, t_break, t_end, log_r1, log_r2) = theta
    return (10**logF0_FS, 10**logTb_FS, p_FS, 10**logC_bg,
            t_start, t_break, t_end, log_r1, log_r2)

def log_prob_injection(theta, x, y, yerr):
    lp = log_prior_injection(theta)
    if not np.isfinite(lp):
        return -np.inf
    params = theta_to_params_injection(theta)
    try:
        ymod = windowed_eval(model_injection, x, params)
    except Exception:
        return -np.inf
    if not np.all(np.isfinite(ymod)):
        return -np.inf
    return lp - 0.5 * np.sum(((y - ymod) / yerr)**2)

def theta0_injection_from_fs(theta_FS_best):
    theta0 = np.array([
        theta_FS_best[0],
        theta_FS_best[1],
        theta_FS_best[2],
        theta_FS_best[3],
        0.018,
        0.032,
        0.046,
        np.log(2.2),
        np.log(1.3),
    ])
    for j, (lo, hi) in enumerate(PRIOR_INJECTION.values()):
        theta0[j] = np.clip(theta0[j], lo + 1e-4, hi - 1e-4)
    if theta0[5] <= theta0[4]:
        theta0[5] = min(PRIOR_INJECTION['t_break'][1] - 1e-4, theta0[4] + 0.004)
    if theta0[6] <= theta0[5]:
        theta0[6] = min(PRIOR_INJECTION['t_end'][1] - 1e-4, theta0[5] + 0.006)
    return theta0

def get_components_injection(t, p):
    (F0_FS, tb_FS, p_FS, C_bg, t_start, t_break, t_end,
     log_r1, log_r2) = p
    fs0 = forward_shock_source_flux(t, F0_FS, tb_FS, p_FS)
    e_ratio = energy_ratio_piecewise(t, t_start, t_break, t_end, log_r1, log_r2)
    fs_refreshed = fs0 * e_ratio**energy_flux_index(p_FS)
    cbg = np.full_like(t, C_bg)
    return [
        ('Unrefreshed FS', fs0 + C_bg, '--', 'goldenrod',
         f"tb={tb_FS*1440:.1f} min  decay=-{decay_alpha_from_p(p_FS):.2f}"),
        ('Injected FS', fs_refreshed + C_bg, '-', 'navy',
         f"R1={np.exp(log_r1):.2f}  R2={np.exp(log_r2):.2f}"),
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

def _fit_shift(xD, yD, yeD, quick=False):
    np.random.seed(46)
    return run_emcee(xD, yD, yeD, theta0_SHIFT, PRIOR_SHIFT, log_prob_shift,
                     'Shifted excess DSBPL', quick=quick)

def _fit_combined(xD, yD, yeD, theta0_combined, quick=False):
    np.random.seed(52)
    return run_emcee(xD, yD, yeD, theta0_combined, PRIOR_COMBINED,
                     log_prob_combined, 'Combined FS+DSBPL', quick=quick)

def _fit_injection(xD, yD, yeD, theta0_injection, quick=False):
    np.random.seed(62)
    return run_emcee(xD, yD, yeD, theta0_injection, PRIOR_INJECTION,
                     log_prob_injection, 'FS energy injection', quick=quick)

# ---------------------------------------------------------------
# Summarize flat chains
# ---------------------------------------------------------------
NATURAL_LOG_PARAMS = {'log_r1', 'log_r2'}

def theta_column_to_phys(name, values):
    if name in NATURAL_LOG_PARAMS:
        return np.exp(values)
    if name.startswith('log'):
        return 10**values
    return values

def summarize(flat_chain, flat_lp, theta_to_params_fn, param_names):
    best_idx    = np.argmax(flat_lp)
    theta_best  = flat_chain[best_idx]
    params_best = theta_to_params_fn(theta_best)
    quantiles   = []
    for j, name in enumerate(param_names):
        phys = theta_column_to_phys(name, flat_chain[:, j])
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
        best = theta_column_to_phys(name, theta_best[j]) * sc
        print(f"  {lbl:>24s}: {best:9.3f}   {q50*sc:9.3f}"
              f"  +{(q84-q50)*sc:7.3f}  -{(q50-q16)*sc:7.3f}")

# ---------------------------------------------------------------
# Corner plots
# ---------------------------------------------------------------
_CORNER_CFG = {
    'FS': dict(
        names=NAMES_FS,
        scale=[1e6, 1440, 1, 1e6],
        labels=[r'$F_0\ (\mu\mathrm{Jy})$', r'$t_b\ (\mathrm{min})$',
                r'$p$', r'$C_{\rm bg}\ (\mu\mathrm{Jy})$'],
    ),
    'SHIFT': dict(
        names=NAMES_SHIFT,
        scale=[1e6, 1, 1, 1, 1, 1],
        labels=[r'$F_0\ (\mu\mathrm{Jy})$', r'$t_{b,1}\ (\mathrm{d})$',
                r'$t_{b,2}\ (\mathrm{d})$', r'$\alpha_1$', r'$\alpha_2$',
                r'$\alpha_3$'],
    ),
    'COMBINED': dict(
        names=NAMES_COMBINED,
        scale=[1e6, 1440, 1, 1e6, 1, 1, 1, 1, 1, 1, 1e6],
        labels=[r'$F_{0,\rm FS}\ (\mu\mathrm{Jy})$', r'$t_{b,\rm FS}\ (\mathrm{min})$',
                r'$p$', r'$F_{0,\rm D}\ (\mu\mathrm{Jy})$',
                r'$t_{0,\rm D}\ (\mathrm{d})$',
                r'$\tau_{b,1,\rm D}\ (\mathrm{d})$', r'$\tau_{b,2,\rm D}\ (\mathrm{d})$',
                r'$\alpha_{1,\rm D}$', r'$\alpha_{2,\rm D}$',
                r'$\alpha_{3,\rm D}$', r'$C_{\rm bg}\ (\mu\mathrm{Jy})$'],
    ),
    'INJECTION': dict(
        names=NAMES_INJECTION,
        scale=[1e6, 1440, 1, 1e6, 1, 1, 1, 1, 1],
        labels=[r'$F_{0,\rm FS}\ (\mu\mathrm{Jy})$', r'$t_{b,\rm FS}\ (\mathrm{min})$',
                r'$p$', r'$C_{\rm bg}\ (\mu\mathrm{Jy})$',
                r'$t_s\ (\mathrm{d})$', r'$t_1\ (\mathrm{d})$',
                r'$t_2\ (\mathrm{d})$', r'$R_1$', r'$R_2$'],
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
        samp[:, j] = theta_column_to_phys(name, samp[:, j]) * sc
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
MODEL_COLOR = {'FS': 'royalblue', 'COMBINED': 'navy', 'INJECTION': 'navy'}

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
    # ax.errorbar(t_master, F_master, yerr=eF_master, fmt='D',
    #             color='mediumseagreen', markersize=8, elinewidth=1.4,
    #             capsize=4, markeredgecolor='k', markeredgewidth=0.6,
    #             label=f'MASTER T+{t_master*1440:.1f} min')

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

def compute_effective_t90(x, excess, t_min, t_max):
    """Return T05 offset, effective T90, and T95 for positive excess fluence."""
    sel = (x >= t_min) & (x <= t_max) & np.isfinite(excess) & (excess > 0)
    tx = x[sel]
    fx = excess[sel]
    if len(tx) < 2:
        return t_min, 0.0, t_min

    order = np.argsort(tx)
    tx = tx[order]
    fx = fx[order]
    edges = np.empty(len(tx) + 1)
    edges[1:-1] = 0.5 * (tx[:-1] + tx[1:])
    edges[0] = max(t_min, tx[0] - 0.5 * (tx[1] - tx[0]))
    edges[-1] = min(t_max, tx[-1] + 0.5 * (tx[-1] - tx[-2]))
    dt = np.maximum(np.diff(edges), 0.0)
    cumulative = np.cumsum(fx * dt)
    total = cumulative[-1]
    if not np.isfinite(total) or total <= 0:
        return t_min, 0.0, t_min

    t05 = np.interp(0.05 * total, cumulative, tx)
    t95 = np.interp(0.95 * total, cumulative, tx)
    return t05, t95 - t05, t95

def save_shifted_excess_plot(model_fn, params_best, tburst_offset, t90_eff):
    """Plot positive Data - Model excess versus computed shifted time."""
    t_shift = x_plot - tburst_offset
    in_win  = (t_shift > 0) & (t_shift <= SHIFT_PLOT_MAX)
    ymod    = model_fn(x_plot[in_win], *params_best)
    excess  = y_plot[in_win] - ymod
    e_excess = ye_plot[in_win]
    x_excess = t_shift[in_win]

    pos = excess > 0
    n_omit = int((~pos).sum())

    fig, ax = plt.subplots(figsize=(8, 6))
    if pos.any():
        ax.errorbar(x_excess[pos], excess[pos], yerr=e_excess[pos],
                    fmt='.', color='black', markersize=3, elinewidth=0.4,
                    alpha=0.8, capsize=0, label='Data - model')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(max(x_excess[pos].min() * 0.8, 1e-5) if pos.any() else 1e-5, 1.0)
    if pos.any():
        ax.set_ylim(max(e_excess[pos].min() * 0.3, excess[pos].min() * 0.5),
                    excess[pos].max() * 2.0)
    ax.set_xlabel(f'Days after T$_0$ + {tburst_offset:.5f} d', fontsize=11)
    ax.set_ylabel('Data - Model (Jy)', fontsize=11)
    ax.grid(True, which='both', alpha=0.2, lw=0.5)
    ax.legend(fontsize=8, loc='best')
    ax.set_title(f'Positive excess after forward-shock subtraction  '
                 f'(T90={t90_eff:.3f} d; {n_omit} non-positive points omitted)', fontsize=10)
    plt.suptitle('GRB 260207A', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('GRB260207A_shifted_excess.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: GRB260207A_shifted_excess.png  "
          f"(positive points={int(pos.sum())}, omitted non-positive={n_omit})")

def save_shifted_dsbpl_plot(x_shift_plot, y_shift_plot, ye_shift_plot,
                            fit_mask_plot, chain, lp, chi2_r,
                            tburst_offset, t90_eff):
    p_best = theta_to_params_shift(chain[np.argmax(lp)])
    t_model = np.logspace(np.log10(max(x_shift_plot[x_shift_plot > 0].min() * 0.8, 1e-5)),
                          np.log10(SHIFT_PLOT_MAX), 1200)

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(8, 9),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    pos_data = y_shift_plot > 0
    fit_pos = fit_mask_plot & pos_data
    excl_pos = ~fit_mask_plot & pos_data
    fit_nonpos = fit_mask_plot & ~pos_data
    excl_nonpos = ~fit_mask_plot & ~pos_data
    if excl_pos.any():
        ax.errorbar(x_shift_plot[excl_pos], y_shift_plot[excl_pos],
                    yerr=ye_shift_plot[excl_pos], fmt='.', color='gray',
                    markersize=2, elinewidth=0.3, alpha=0.45, capsize=0,
                    label='Excluded')
    if fit_pos.any():
        ax.errorbar(x_shift_plot[fit_pos], y_shift_plot[fit_pos],
                    yerr=ye_shift_plot[fit_pos], fmt='.', color='black',
                    markersize=3, elinewidth=0.4, alpha=0.8, capsize=0,
                    label='Fitted')
    if (fit_nonpos | excl_nonpos).any():
        nonpos_x = np.concatenate([x_shift_plot[excl_nonpos], x_shift_plot[fit_nonpos]])
        nonpos_y = np.concatenate([np.abs(y_shift_plot[excl_nonpos]),
                                   np.abs(y_shift_plot[fit_nonpos])])
        nonpos_ye = np.concatenate([ye_shift_plot[excl_nonpos],
                                    ye_shift_plot[fit_nonpos]])
        if len(nonpos_x):
            ax.errorbar(nonpos_x, nonpos_y, yerr=nonpos_ye, fmt='.',
                        color='lightgray', markersize=2, elinewidth=0.3,
                        alpha=0.35, capsize=0, label='Non-positive residuals (abs)')
    ax.axvline(SHIFT_FIT_MAX, color='gray', ls='--', lw=0.9, alpha=0.6,
               label=f'Fit limit: {SHIFT_FIT_MAX:.2f} d')

    rng = np.random.default_rng(46)
    for i in rng.choice(len(chain), min(200, len(chain)), replace=False):
        y_samp = model_shifted_excess(t_model, *theta_to_params_shift(chain[i]))
        ax.plot(t_model, np.where(y_samp > 0, y_samp, np.nan),
                '-', color='darkorange', lw=0.3, alpha=0.04, zorder=4)

    y_best = model_shifted_excess(t_model, *p_best)
    ax.plot(t_model, np.where(y_best > 0, y_best, np.nan),
            '-', color='darkorange', lw=2.0, zorder=6,
            label=f'Best fit  chi2_r={chi2_r:.2f}')
    for (clabel, cy, ls, ccolor, desc) in get_components_shift(t_model, p_best):
        ax.plot(t_model, np.where(cy > 0, cy, np.nan),
                color=ccolor, ls=ls, lw=1.4, zorder=5, label=f'{clabel}: {desc}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(t_model.min(), SHIFT_PLOT_MAX)
    vis = pos_data & np.isfinite(y_shift_plot)
    if vis.any():
        ax.set_ylim(max(ye_shift_plot[vis].min() * 0.3, y_shift_plot[vis].min() * 0.5),
                    y_shift_plot[vis].max() * 2.0)
    ax.set_ylabel('Data - Forward Shock (Jy)', fontsize=11)
    ax.set_title(f'Shifted excess: DSBPL only  '
                 f'(T90={t90_eff:.3f} d)', fontsize=10)
    ax.legend(fontsize=6.5, loc='best')
    ax.grid(True, which='both', alpha=0.2, lw=0.5)
    ax.tick_params(labelbottom=False)

    ymod_data = model_shifted_excess(x_shift_plot, *p_best)
    resid = (y_shift_plot - ymod_data) / ye_shift_plot
    ax_res.axhline(0, color='darkorange', lw=1.2, zorder=5)
    ax_res.axhline( 3, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax_res.axhline(-3, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax_res.errorbar(x_shift_plot[~fit_mask_plot], resid[~fit_mask_plot],
                    yerr=np.ones_like(resid[~fit_mask_plot]),
                    fmt='.', color='gray', markersize=2, elinewidth=0.3,
                    alpha=0.45, capsize=0)
    ax_res.errorbar(x_shift_plot[fit_mask_plot], resid[fit_mask_plot],
                    yerr=np.ones_like(resid[fit_mask_plot]),
                    fmt='.', color='black', markersize=3, elinewidth=0.4,
                    alpha=0.8, capsize=0)
    ax_res.axvline(SHIFT_FIT_MAX, color='gray', ls='--', lw=0.9, alpha=0.6)
    finite = np.abs(resid[np.isfinite(resid)])
    ymax = max(7.0, min(30.0, 1.2 * np.percentile(finite, 99))) if len(finite) else 7.0
    ax_res.set_xscale('log')
    ax_res.set_xlim(t_model.min(), SHIFT_PLOT_MAX)
    ax_res.set_ylim(-ymax, ymax)
    ax_res.set_xlabel(f'Days after T$_0$ + {tburst_offset:.5f} d', fontsize=11)
    ax_res.set_ylabel('Residuals (sigma)', fontsize=10)
    ax_res.grid(True, which='both', alpha=0.2, lw=0.5)

    plt.suptitle('GRB 260207A', fontsize=14, y=0.995)
    plt.savefig('GRB260207A_shifted_excess_dsbpl.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: GRB260207A_shifted_excess_dsbpl.png")

def save_combined_plot(mask_combined_fit, chain, lp, chi2_r, tburst_offset):
    p_best = theta_to_params_combined(chain[np.argmax(lp)])
    t_model = np.logspace(np.log10(2e-4), np.log10(13), 1500)
    color = MODEL_COLOR['COMBINED']

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(8, 9),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    add_data_to_ax(ax, mask_combined_fit)

    rng = np.random.default_rng(52)
    for i in rng.choice(len(chain), min(200, len(chain)), replace=False):
        y_samp = model_combined(t_model, *theta_to_params_combined(chain[i]))
        ax.plot(t_model, np.where(y_samp > 0, y_samp, np.nan),
                '-', color=color, lw=0.3, alpha=0.04, zorder=4)

    y_best = model_combined(t_model, *p_best)
    ax.plot(t_model, np.where(y_best > 0, y_best, np.nan),
            '-', color=color, lw=2.0, zorder=6,
            label=f'Best fit  chi2_r={chi2_r:.2f}')
    for (clabel, cy, ls, ccolor, desc) in get_components_combined(t_model, p_best):
        ax.plot(t_model, np.where(cy > 0, cy, np.nan),
                color=ccolor, ls=ls, lw=1.4, zorder=5, label=f'{clabel}: {desc}')

    ax.axvline(p_best[4], color='dimgray', ls=':', lw=1.1, alpha=0.9,
               label=f'fitted t0_D={p_best[4]:.4f} d')
    format_main_ax(ax)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel('Flux density (Jy)', fontsize=11)
    ax.set_title('Combined original-frame model: FS SBPL + second-peak DSBPL', fontsize=10)
    ax.legend(fontsize=6.2, loc='best')
    add_minutes_axis(ax)
    plot_residuals(ax_res, mask_combined_fit, model_combined, p_best, color)

    plt.suptitle('GRB 260207A', fontsize=14, y=0.995)
    plt.savefig('GRB260207A_emcee_combined.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: GRB260207A_emcee_combined.png")

def save_injection_plot(mask_injection_fit, chain, lp, chi2_r):
    p_best = theta_to_params_injection(chain[np.argmax(lp)])
    t_model = np.logspace(np.log10(2e-4), np.log10(13), 1500)
    color = MODEL_COLOR['INJECTION']

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(8, 9),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    add_data_to_ax(ax, mask_injection_fit)

    rng = np.random.default_rng(62)
    for i in rng.choice(len(chain), min(200, len(chain)), replace=False):
        y_samp = model_injection(t_model, *theta_to_params_injection(chain[i]))
        ax.plot(t_model, np.where(y_samp > 0, y_samp, np.nan),
                '-', color=color, lw=0.3, alpha=0.04, zorder=4)

    y_best = model_injection(t_model, *p_best)
    ax.plot(t_model, np.where(y_best > 0, y_best, np.nan),
            '-', color=color, lw=2.0, zorder=6,
            label=f'Best fit  chi2_r={chi2_r:.2f}')
    for (clabel, cy, ls, ccolor, desc) in get_components_injection(t_model, p_best):
        ax.plot(t_model, np.where(cy > 0, cy, np.nan),
                color=ccolor, ls=ls, lw=1.4, zorder=5, label=f'{clabel}: {desc}')

    t_start, t_break, t_end = p_best[4], p_best[5], p_best[6]
    for tx, lbl in [(t_start, r'$t_s$'), (t_break, r'$t_1$'), (t_end, r'$t_2$')]:
        ax.axvline(tx, color='dimgray', ls=':', lw=0.9, alpha=0.75,
                   label=f'{lbl}={tx:.4f} d')

    format_main_ax(ax)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel('Flux density (Jy)', fontsize=11)
    ax.set_title('Forward shock with monotonic energy injection', fontsize=10)
    ax.legend(fontsize=5.9, loc='best')
    add_minutes_axis(ax)
    plot_residuals(ax_res, mask_injection_fit, model_injection, p_best, color)

    plt.suptitle('GRB 260207A', fontsize=14, y=0.995)
    plt.savefig('GRB260207A_emcee_injection.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: GRB260207A_emcee_injection.png")

def save_background_plot(bg_models, output_path='GRB260207A_tess_bg.png'):
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
    for item in bg_models:
        if len(item) == 4:
            tag, chain, lp, theta_fn = item
            c_bg_index = -1
        else:
            tag, chain, lp, theta_fn, c_bg_index = item
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
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


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
    x_all     = rawdata[:, 0] - trigger_tjd
    y_all     = -rawdata[:, 1] / (200 * 0.8 * 0.99)
    yerr_all  =  rawdata[:, 2] / (200 * 0.8 * 0.99)
    zp        = 2416 * 10**(-0.4 * 20.44)
    flux_all  = y_all    * zp
    eflux_all = yerr_all * zp

    # Include 6 h pre-burst for background constraint, out to 12 d post-burst
    mask_plot = (x_all >= -1.0) & (x_all <= 12.0)
    x_plot  = x_all[mask_plot]
    y_plot  = flux_all[mask_plot]
    ye_plot = eflux_all[mask_plot]

    if args.nomodel:
        # --- Extended mask: first cadence onward through end of data
        mask_ext = (x_all > -HALF_C) & (x_all <= x_all.max())
        xp   = x_all[mask_ext]
        yp   = y_all[mask_ext]
        yep  = yerr_all[mask_ext]

        # --- Pre-burst baseline: weighted mean of all t in [-6 h, 0]
        pre = (x_all >= -1.0) & (x_all <= 0)
        wt      = 1.0 / yerr_all[pre]**2
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

        # ax.errorbar(t_master, F_master, yerr=eF_master, fmt='D',
        #             color='mediumseagreen', markersize=8, elinewidth=1.4,
        #             capsize=4, markeredgecolor='k', markeredgewidth=0.6,
        #             label=f'MASTER T+{t_master*1440:.1f} min')

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
    mask_fit = (x_plot >= -1.0) & (x_plot <= FS_FIT_MAX)
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
    decay_best = decay_alpha_from_p(pFS[2])

    print_params('FORWARD SHOCK MODEL', thetaFS, NAMES_FS, qFS,
                 ['F0 (uJy)', 'tb (min)', 'p', 'C_bg (uJy)'],
                 [1e6, 1440, 1, 1e6],
                 chi2_rFS)
    print(f"  {'rise slope':>24s}:     0.500   (fixed, flux ∝ t^0.5)")
    print(f"  {'decay slope':>24s}:    {-decay_best: .3f}   "
          f"(fixed by best-fit p as -3(p-1)/4)")

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
    ax.set_title('Forward shock only: rise t$^{0.5}$, decay t$^{-3(p-1)/4}$', fontsize=10)
    ax.legend(fontsize=6.5, loc='lower left')
    add_minutes_axis(ax)
    plot_residuals(ax_res, mask_fit, model_FS, p_best, color)

    plt.suptitle('GRB 260207A', fontsize=14, y=0.995)
    plt.savefig('GRB260207A_emcee_forward_shock.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: GRB260207A_emcee_forward_shock.png")

    # -----------------------------------------------------------
    # Fit full trigger-frame refreshed forward shock.
    # -----------------------------------------------------------
    injection_fit_max = COMBINED_FIT_MAX
    mask_injection_fit = (x_plot >= -1.0) & (x_plot <= injection_fit_max)
    xI = x_plot[mask_injection_fit]
    yI = y_plot[mask_injection_fit]
    yeI = ye_plot[mask_injection_fit]
    theta0_injection = theta0_injection_from_fs(thetaFS)

    print("\nEnergy-injection initial guess:")
    print(f"  {'t_start':>18s}: {theta0_injection[4]:9.5f} d")
    print(f"  {'t_break':>18s}: {theta0_injection[5]:9.5f} d")
    print(f"  {'t_end':>18s}: {theta0_injection[6]:9.5f} d")
    print(f"  {'R1':>18s}: {np.exp(theta0_injection[7]):9.3f}")
    print(f"  {'R2':>18s}: {np.exp(theta0_injection[8]):9.3f}")

    print(f"\nN_injection_fit = {len(xI)},  t in "
          f"[{xI.min()*1440:.2f} min, {xI.max():.3f} d]")
    print(f"Launching FS energy-injection fit ({mode_str})...")
    chainInjection, lpInjection = _fit_injection(
        xI, yI, yeI, theta0_injection, args.quick)

    thetaInjection, pInjection, qInjection = summarize(
        chainInjection, lpInjection, theta_to_params_injection, NAMES_INJECTION)
    ymod_inj = windowed_eval(model_injection, xI, pInjection)
    chi2_injection = np.sum(((yI - ymod_inj) / yeI)**2)
    chi2_rInjection = chi2_injection / (len(xI) - len(NAMES_INJECTION))

    print_params('FORWARD SHOCK ENERGY-INJECTION MODEL',
                 thetaInjection, NAMES_INJECTION, qInjection,
                 ['F0_FS (uJy)', 'tb_FS (min)', 'p', 'C_bg (uJy)',
                  't_start (d)', 't_break (d)', 't_end (d)',
                  'R1', 'R2'],
                 [1e6, 1440, 1, 1e6, 1, 1, 1, 1, 1],
                 chi2_rInjection)

    F0_FS, tb_FS, p_FS, C_bg, t_start, t_break, t_end, log_r1, log_r2 = pInjection
    g = energy_flux_index(p_FS)
    R1 = np.exp(log_r1)
    R2 = np.exp(log_r2)
    e1 = log_r1 / np.log(t_break / t_start)
    e2 = log_r2 / np.log(t_end / t_break)
    s1 = 8.0 * e1 / (3.0 - e1) if e1 < 3.0 else np.inf
    s2 = 8.0 * e2 / (3.0 - e2) if e2 < 3.0 else np.inf
    print("\nDerived injection diagnostics:")
    print(f"  {'g=(p+3)/4':>24s}: {g:9.3f}")
    print(f"  {'R_final':>24s}: {R1*R2:9.3f}")
    print(f"  {'final flux boost':>24s}: {(R1*R2)**g:9.3f}")
    print(f"  {'e1':>24s}: {e1:9.3f}")
    print(f"  {'e2':>24s}: {e2:9.3f}")
    print(f"  {'s1':>24s}: {s1:9.3f}")
    print(f"  {'s2':>24s}: {s2:9.3f}")

    save_corner('INJECTION', chainInjection)
    save_injection_plot(mask_injection_fit, chainInjection, lpInjection,
                        chi2_rInjection)

    # -----------------------------------------------------------
    # TESS background plot
    # -----------------------------------------------------------
    save_background_plot([
        ('Forward shock', chainFS, lpFS, theta_to_params_FS, -1),
        ('INJECTION', chainInjection, lpInjection, theta_to_params_injection, 3),
    ], output_path='GRB260207A_tess_bg_injection.png')
