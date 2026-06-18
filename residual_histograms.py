"""
Histogram residuals for the internal and energy-injection combined fits.

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
    x_all = rawdata[:, 0] - module.trigger_tjd
    y_all = -rawdata[:, 1] / (200 * 0.8 * 0.99)
    yerr_all = rawdata[:, 2] / (200 * 0.8 * 0.99)
    zp = 2416 * 10 ** (-0.4 * 20.44)

    mask_plot = (x_all >= -1.0) & (x_all <= 12.0)
    return x_all[mask_plot], y_all[mask_plot] * zp, yerr_all[mask_plot] * zp


def best_theta(chain, lp):
    return chain[int(np.argmax(lp))]


def fit_internal_combined(quick=True):
    x_plot, y_plot, ye_plot = load_light_curve(internal)
    internal.x_plot = x_plot
    internal.y_plot = y_plot
    internal.ye_plot = ye_plot

    mask_fs = (x_plot >= -1.0) & (x_plot <= internal.FS_FIT_MAX)
    chain_fs, lp_fs = internal._fit_FS(
        x_plot[mask_fs], y_plot[mask_fs], ye_plot[mask_fs], quick=quick)
    theta_fs = best_theta(chain_fs, lp_fs)
    p_fs = internal.theta_to_params_FS(theta_fs)

    fs_subtracted = y_plot - internal.model_FS(x_plot, *p_fs)
    timing_max = internal.FS_FIT_MAX + internal.SHIFT_FIT_MAX
    tburst_offset, _, _ = internal.compute_effective_t90(
        x_plot, fs_subtracted, 0.0, timing_max)

    t_shift_all = x_plot - tburst_offset
    mask_shift_fit = (
        (t_shift_all > 0) &
        (t_shift_all <= internal.SHIFT_PLOT_MAX) &
        (t_shift_all <= internal.SHIFT_FIT_MAX)
    )
    chain_shift, lp_shift = internal._fit_shift(
        t_shift_all[mask_shift_fit],
        y_plot[mask_shift_fit] - internal.model_FS(x_plot[mask_shift_fit], *p_fs),
        ye_plot[mask_shift_fit],
        quick=quick)
    p_shift = internal.theta_to_params_shift(best_theta(chain_shift, lp_shift))

    internal.set_combined_t0_prior(tburst_offset)
    theta0_combined = internal.theta0_combined_from_fits(
        theta_fs, p_shift, tburst_offset)

    mask_combined = (x_plot >= -1.0) & (x_plot <= internal.COMBINED_FIT_MAX)
    chain_combined, lp_combined = internal._fit_combined(
        x_plot[mask_combined], y_plot[mask_combined], ye_plot[mask_combined],
        theta0_combined, quick=quick)
    return best_theta(chain_combined, lp_combined)


def fit_injection_combined(quick=True):
    x_plot, y_plot, ye_plot = load_light_curve(injection)
    injection.x_plot = x_plot
    injection.y_plot = y_plot
    injection.ye_plot = ye_plot

    mask_fs = (x_plot >= -1.0) & (x_plot <= injection.FS_FIT_MAX)
    chain_fs, lp_fs = injection._fit_FS(
        x_plot[mask_fs], y_plot[mask_fs], ye_plot[mask_fs], quick=quick)
    theta_fs = best_theta(chain_fs, lp_fs)

    theta0_injection = injection.theta0_injection_from_fs(theta_fs)
    mask_injection = (x_plot >= -1.0) & (x_plot <= injection.COMBINED_FIT_MAX)
    chain_injection, lp_injection = injection._fit_injection(
        x_plot[mask_injection], y_plot[mask_injection], ye_plot[mask_injection],
        theta0_injection, quick=quick)
    return best_theta(chain_injection, lp_injection)


def load_or_fit(refit=False, quick=True):
    if CACHE_PATH.exists() and not refit:
        cache = np.load(CACHE_PATH)
        return cache["theta_internal"], cache["theta_injection"], "cache"

    theta_internal = fit_internal_combined(quick=quick)

    # _fit_FS exists in both modules with the same name; reloading keeps the
    # globals and priors for the injection run independent from the internal run.
    importlib.reload(injection)
    theta_injection = fit_injection_combined(quick=quick)

    np.savez(
        CACHE_PATH,
        theta_internal=theta_internal,
        theta_injection=theta_injection,
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
        "Internal combined FS+DSBPL": residual_set(
            internal, theta_internal, internal.theta_to_params_combined,
            internal.model_combined),
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
