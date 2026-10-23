"""
Histogram residuals for the internal-flare and energy-injection fits.

The first run performs quick emcee refits using the same model functions as
internal_model.py and injection_model.py, then caches the best-fit parameters.
Subsequent runs reuse that cache unless --refit is supplied.
"""

import argparse
import importlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

import internal_model as internal
import injection_model as injection


CACHE_PATH = Path("GRB260207A_residual_hist_fit_cache.npz")
OUTPUT_PATH = Path("GRB260207A_combined_fit_residual_histograms.png")
TIME_MIN = 0.8
TIME_MAX = 1.0


def load_light_curve(module):
    rawdata = np.loadtxt("lc_GRB260207A_cand41148_geo")
    x_all = rawdata[:, 0] + module.HALF_C - module.trigger_tjd
    y_all = -rawdata[:, 1] / (200 * 0.8 * 0.99)
    yerr_all = rawdata[:, 2] / (200 * 0.8 * 0.99)
    zp = 2416 * 10 ** (-0.4 * 20.44)
    flux_all = y_all * zp
    eflux_all = yerr_all * zp

    pre = ((x_all >= module.PREBURST_MIN) &
           (x_all <= module.PREBURST_MAX))
    jitter = module.estimate_additive_jitter(flux_all[pre], eflux_all[pre])
    eflux_eff_all = np.sqrt(eflux_all**2 + jitter**2)

    mask_plot = (x_all >= module.PREBURST_MIN) & (x_all <= 12.0)
    return x_all[mask_plot], flux_all[mask_plot], eflux_eff_all[mask_plot]


def best_theta(chain, lp):
    return chain[int(np.argmax(lp))]


def fit_internal_model(quick=True):
    x_plot, y_plot, ye_plot = load_light_curve(internal)
    internal.x_plot = x_plot
    internal.y_plot = y_plot
    internal.ye_plot = ye_plot

    mask_fs = ((x_plot >= internal.PREBURST_MIN) &
               (x_plot <= internal.FS_FIT_MAX))
    chain_fs, lp_fs = internal._fit_FS(
        internal.as_bin_edges(x_plot[mask_fs]), y_plot[mask_fs], ye_plot[mask_fs], quick=quick)
    theta_fs = best_theta(chain_fs, lp_fs)
    internal.ALPHA_RISE_INTERNAL = internal.fs_rise_posterior_median(chain_fs)
    internal.S_AB = internal.fs_smoothness_posterior_median(chain_fs)
    internal.P_INTERNAL = internal.fs_p_posterior_median(chain_fs)

    theta0_internal = internal.theta0_internal_from_fs(theta_fs)
    mask_internal = ((x_plot >= internal.PREBURST_MIN) &
                     (x_plot <= internal.INTERNAL_FIT_MAX))
    chain_internal, lp_internal = internal._fit_internal(
        internal.as_bin_edges(x_plot[mask_internal]), y_plot[mask_internal], ye_plot[mask_internal],
        theta0_internal, quick=quick)
    return best_theta(chain_internal, lp_internal)


def fit_injection_combined(quick=True):
    x_plot, y_plot, ye_plot = load_light_curve(injection)
    injection.x_plot = x_plot
    injection.y_plot = y_plot
    injection.ye_plot = ye_plot

    mask_fs = ((x_plot >= injection.PREBURST_MIN) &
               (x_plot <= injection.FS_FIT_MAX))
    chain_fs, lp_fs = injection._fit_FS(
        injection.as_bin_edges(x_plot[mask_fs]), y_plot[mask_fs], ye_plot[mask_fs], quick=quick)
    theta_fs = best_theta(chain_fs, lp_fs)
    injection.ALPHA_RISE_INJECTION = injection.fs_rise_posterior_median(chain_fs)
    injection.S_AB = injection.fs_smoothness_posterior_median(chain_fs)
    injection.P_INJECTION = injection.fs_p_posterior_median(chain_fs)

    theta0_injection = injection.theta0_injection_from_fs(theta_fs)
    mask_injection = ((x_plot >= injection.PREBURST_MIN) &
                      (x_plot <= injection.INJECTION_FIT_MAX))
    chain_injection, lp_injection = injection._fit_injection(
        injection.as_bin_edges(x_plot[mask_injection]), y_plot[mask_injection], ye_plot[mask_injection],
        theta0_injection, quick=quick)
    return best_theta(chain_injection, lp_injection)


def load_or_fit(refit=False, quick=True):
    if CACHE_PATH.exists() and not refit:
        cache = np.load(CACHE_PATH)
        theta_internal = cache["theta_internal"]
        theta_injection = cache["theta_injection"]
        injection_cache_version = (int(cache["injection_model_version"])
                                   if "injection_model_version" in cache else None)
        internal_cache_version = (int(cache["internal_model_version"])
                                  if "internal_model_version" in cache else None)
        fixed_keys = {
            "p_internal", "alpha_rise_internal", "s_internal",
            "p_injection", "alpha_rise_injection", "s_injection",
        }
        if (fixed_keys.issubset(cache.files) and
                len(theta_internal) == len(internal.NAMES_INTERNAL) and
                internal_cache_version == internal.INTERNAL_MODEL_VERSION and
                len(theta_injection) == len(injection.NAMES_INJECTION) and
                injection_cache_version == injection.INJECTION_MODEL_VERSION):
            internal.P_INTERNAL = float(cache["p_internal"])
            internal.ALPHA_RISE_INTERNAL = float(cache["alpha_rise_internal"])
            internal.S_AB = float(cache["s_internal"])
            injection.P_INJECTION = float(cache["p_injection"])
            injection.ALPHA_RISE_INJECTION = float(cache["alpha_rise_injection"])
            injection.S_AB = float(cache["s_injection"])
            return theta_internal, theta_injection, "cache"
        print("Cached fit is stale; refitting.")

    theta_internal = fit_internal_model(quick=quick)

    # _fit_FS exists in both modules with the same name; reloading keeps the
    # globals and priors for the injection run independent from the internal run.
    importlib.reload(injection)
    theta_injection = fit_injection_combined(quick=quick)

    np.savez(
        CACHE_PATH,
        theta_internal=theta_internal,
        theta_injection=theta_injection,
        internal_model_version=internal.INTERNAL_MODEL_VERSION,
        injection_model_version=injection.INJECTION_MODEL_VERSION,
        p_internal=internal.P_INTERNAL,
        alpha_rise_internal=internal.ALPHA_RISE_INTERNAL,
        s_internal=internal.S_AB,
        p_injection=injection.P_INJECTION,
        alpha_rise_injection=injection.ALPHA_RISE_INJECTION,
        s_injection=injection.S_AB,
    )
    return theta_internal, theta_injection, "refit"


def residual_set(module, theta, theta_to_params, model_fn):
    x_plot, y_plot, ye_plot = load_light_curve(module)
    params = theta_to_params(theta)
    win = (x_plot >= TIME_MIN) & (x_plot <= TIME_MAX) & np.isfinite(y_plot) & np.isfinite(ye_plot)
    model = module.windowed_eval(model_fn, x_plot[win], params)
    resid = (y_plot[win] - model) / ye_plot[win]
    return resid[np.isfinite(resid)]


def histogram_mode(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    idx = int(np.argmax(counts))
    return 0.5 * (edges[idx] + edges[idx + 1])


def add_stat_lines(ax, values, bins, color_model):
    stats = [
        ("Model: 0", 0.0, color_model, "-"),
        ("Mean", float(np.mean(values)), "#D55E00", "--"),
        ("Median", float(np.median(values)), "#009E73", "-."),
        ("Mode", histogram_mode(values, bins), "#CC79A7", ":"),
    ]
    for label, xval, color, ls in stats:
        ax.axvline(xval, color=color, ls=ls, lw=2.0, label=f"{label} = {xval:.2f}")


def make_plot(theta_internal, theta_injection):
    residuals = {
        "Internal FS+flare fit": residual_set(
            internal, theta_internal, internal.theta_to_params_internal,
            internal.model_internal),
        "Energy-injection combined fit": residual_set(
            injection, theta_injection, injection.theta_to_params_injection,
            injection.model_injection),
    }

    all_resid = np.concatenate(list(residuals.values()))
    lo, hi = np.percentile(all_resid, [1, 99])
    pad = max(1.0, 0.15 * (hi - lo))
    bins = np.linspace(lo - pad, hi + pad, 28)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    hist_colors = ["#0072B2", "#E69F00"]
    model_colors = ["#000000", "#555555"]

    for ax, (title, values), hist_color, model_color in zip(
            axes, residuals.items(), hist_colors, model_colors):
        ax.hist(values, bins=bins, color=hist_color, alpha=0.72,
                edgecolor="white", linewidth=0.8)
        add_stat_lines(ax, values, bins, model_color)
        ax.set_title(f"{title}\nN={len(values)}", fontsize=10)
        ax.set_xlabel("Residual (sigma)")
        ax.grid(True, axis="y", alpha=0.25, lw=0.6)
        ax.legend(fontsize=8, frameon=False, loc="upper right")

    axes[0].set_ylabel("Number of points")
    fig.suptitle("GRB 260207A residuals from 0.8-1.0 days post-burst", y=0.98)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return residuals, bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit", action="store_true",
                        help="Ignore cached best fits and rerun quick emcee fits.")
    parser.add_argument("--full", action="store_true",
                        help="Use adaptive emcee fits instead of quick fits.")
    args = parser.parse_args()

    theta_internal, theta_injection, source = load_or_fit(
        refit=args.refit, quick=not args.full)
    residuals, bins = make_plot(theta_internal, theta_injection)

    print(f"Fit source: {source}")
    print(f"Saved: {OUTPUT_PATH}")
    for label, values in residuals.items():
        print(
            f"{label}: N={len(values)}, mean={np.mean(values):.3f}, "
            f"median={np.median(values):.3f}, mode={histogram_mode(values, bins):.3f}"
        )


if __name__ == "__main__":
    main()
