"""Plots for the light curve fits, styled after ../injection_model.py."""
import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

X_MIN, X_MAX = 1e-4, 13
MINUTE_TICKS = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 17000]

def _corner_title(values):
    q16, q50, q84 = np.percentile(values, [16, 50, 84])
    upper, lower = q84 - q50, q50 - q16
    # Keep two significant figures on the error, so tinj doesn't round to 0.01.
    smallest = min(upper, lower)
    decimals = 2 if smallest <= 0 else max(2, 1 - int(np.floor(np.log10(smallest))))
    return rf'${q50:.{decimals}f}^{{+{upper:.{decimals}f}}}_{{-{lower:.{decimals}f}}}$'


def save_corner(path, samples, labels):
    kwargs = dict(
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        label_kwargs={'fontsize': 15, 'labelpad': 12},
    )
    try:
        with plt.rc_context({'text.usetex': True}):
            fig = corner.corner(samples, **kwargs)
    except Exception:  # no working LaTeX install
        fig = corner.corner(samples, **kwargs)

    axes = np.array(fig.axes).reshape(len(labels), len(labels))
    for i in range(len(labels)):
        axes[i, i].set_title(_corner_title(samples[:, i]), fontsize=13)
    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=11, pad=3)
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.xaxis.labelpad = 16
        ax.yaxis.labelpad = 16
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _plot_data(ax, x, y, yerr, mask_fit):
    pos = y > 0
    floor = blended_transform_factory(ax.transData, ax.transAxes)
    for mask, color, size, line, alpha, s, z, label in (
        (~mask_fit, 'gray', 2, 0.3, 0.4, 18, 1.5, 'Excluded'),
        (mask_fit, 'black', 3, 0.4, 0.8, 22, 2, 'Fitted'),
    ):
        ax.errorbar(x[mask & pos], y[mask & pos], yerr=yerr[mask & pos], fmt='.',
                    color=color, markersize=size, elinewidth=line, alpha=alpha,
                    capsize=0, zorder=z, label=label)
        # Non-positive fluxes can't be drawn on a log axis, so mark them along the floor.
        nonpos = mask & ~pos
        if nonpos.any():
            ax.scatter(x[nonpos], np.full(nonpos.sum(), 0.025), marker='v', s=s,
                       color=color, alpha=alpha, transform=floor, clip_on=False,
                       zorder=z + 2, label=f'{label} <= 0')


def _plot_residuals(ax, x, resid, mask_fit, color):
    ax.axhline(0, color=color, lw=1.2, zorder=5)
    for level in (3, -3):
        ax.axhline(level, color='gray', lw=0.8, ls='--', alpha=0.5)

    ymax = max(7.0, min(30.0, 1.2 * np.percentile(np.abs(resid), 99)))
    for mask, point, size, line, alpha, s, z, label in (
        (~mask_fit, 'gray', 2, 0.3, 0.45, 18, 1.5, 'Excluded'),
        (mask_fit, 'black', 3, 0.3, 0.8, 22, 2, 'Fitted'),
    ):
        inside = mask & (np.abs(resid) <= ymax)
        ax.errorbar(x[inside], resid[inside], yerr=np.ones(inside.sum()), fmt='.',
                    color=point, markersize=size, elinewidth=line, alpha=alpha,
                    capsize=0, zorder=z, label=label)
        # Points beyond the limits are pinned to the edge of the panel.
        for sign, marker in ((1, '^'), (-1, 'v')):
            beyond = mask & (sign * resid > ymax)
            ax.scatter(x[beyond], np.full(beyond.sum(), sign * 0.96 * ymax), marker=marker,
                       s=s, color=point, alpha=alpha, zorder=z + 0.2)

    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel('Days post-burst', fontsize=11)
    ax.set_ylabel('Residuals (σ)', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, which='both', alpha=0.2, lw=0.5)


def _add_minutes_axis(ax):
    top = ax.twiny()
    top.set_xscale('log')
    top.set_xticks([m / 1440 for m in MINUTE_TICKS])
    top.set_xticklabels([str(m) for m in MINUTE_TICKS], fontsize=8)
    top.set_xlim(ax.get_xlim())  # set_xticks can expand the limits
    top.set_xlabel('Minutes post-burst', fontsize=10)


def save_lightcurve(path, t, y, yerr, mask_fit, model, theta_best, samples, chi2_r, *,
                    title, color, cadence, to_jy, components=(), vlines=()):
    """Save the light curve and residual panels.

    t, y and yerr are the post-trigger data, with t the bin start in days and y in
    the model's counts. mask_fit marks the bins used in the fit. model(t, theta)
    returns binned counts for bins starting at t.

    components is a list of (label, fn(t), linestyle, color) and vlines a list of
    (time, label).
    """
    visible = t < X_MAX
    t, y, yerr, mask_fit = t[visible], y[visible], yerr[visible], mask_fit[visible]

    # Bin centers, except the bin straddling the trigger: its center can be negative,
    # so it's plotted at the middle of the part after the trigger.
    x = (np.maximum(t, 0) + t + cadence) / 2
    # The model is binned, so a curve plotted at bin-center x uses a bin starting
    # cadence/2 earlier.
    x_model = np.geomspace(2e-4, X_MAX, 1500)
    t_model = x_model - cadence / 2

    y_jy, yerr_jy = y * to_jy, yerr * to_jy
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})

    _plot_data(ax, x, y_jy, yerr_jy, mask_fit)

    rng = np.random.default_rng(62)
    for i in rng.choice(len(samples), min(200, len(samples)), replace=False):
        ax.plot(x_model, _positive(model(t_model, samples[i]) * to_jy), '-',
                color=color, lw=0.3, alpha=0.035, zorder=4)
    ax.plot(x_model, _positive(model(t_model, theta_best) * to_jy), '-', color=color,
            lw=2.4, zorder=7, label=f'Best fit  chi2_r={chi2_r:.2f}')
    for label, fn, linestyle, line_color in components:
        ax.plot(x_model, _positive(fn(t_model) * to_jy), color=line_color, ls=linestyle,
                lw=1.6, zorder=6, label=label)
    for time, label in vlines:
        ax.axvline(time, color='dimgray', ls=':', lw=0.9, alpha=0.75,
                   label=f'{label}={time:.4f} d')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(X_MIN, X_MAX)
    pos = y_jy > 0
    ax.set_ylim(yerr_jy[pos].min() * 0.3, y_jy[pos].max() * 3)
    ax.grid(True, which='both', alpha=0.2, lw=0.5)
    ax.tick_params(labelbottom=False)
    ax.spines['bottom'].set_visible(False)
    ax_res.spines['top'].set_visible(False)
    ax.set_ylabel('Flux density (Jy)', fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6.5, loc='upper left')

    _plot_residuals(ax_res, x, (y - model(t, theta_best)) / yerr, mask_fit, color)
    _add_minutes_axis(ax)

    fig.suptitle('GRB 260207A', fontsize=14, y=0.995)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _positive(values):
    return np.where(values > 0, values, np.nan)
