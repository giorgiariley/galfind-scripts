import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

# (Optional) put this once near the top of your script to set global font sizes
plt.rcParams.update({
    "axes.labelsize": 15,   # axis label font
    "xtick.labelsize": 13,  # tick label font
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

def _median_err_linear(p16, p50, p84):
    """Return median symmetric error in *linear* units from 16/50/84 arrays."""
    ok = np.isfinite(p16) & np.isfinite(p50) & np.isfinite(p84)
    if not np.any(ok): 
        return None
    lo = p50[ok] - p16[ok]
    hi = p84[ok] - p50[ok]
    return float(np.nanmedian(0.5*(np.maximum(lo,0) + np.maximum(hi,0))))

def _draw_rep_err(ax, x_frac=0.06, y_frac=0.86, xerr=None, yerr=None, color='black'):
    """
    Draw one representative error bar at a fractional location in axes coords.
    Returns (x_frac, y_frac).
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # convert frac -> data coords
    x0 = xmin + x_frac * (xmax - xmin)
    if ax.get_yscale() == 'log':
        y0 = 10**(np.log10(ymin) + y_frac*(np.log10(ymax)-np.log10(ymin)))
    else:
        y0 = ymin + y_frac * (ymax - ymin)

    ax.errorbar([x0], [y0],
                xerr=None if xerr is None else [[xerr],[xerr]],
                yerr=None if yerr is None else [[yerr],[yerr]],
                fmt='none', ecolor=color, elinewidth=1.2, capsize=3, zorder=10)
    return x_frac, y_frac

def _label_rep_err(ax, x_frac, y_frac, text="Typical error bar", dx=0.02, dy=0.02):
    ax.text(x_frac + dx, y_frac + dy, text,
            transform=ax.transAxes, fontsize=12,
            va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.75))


def load_bagpipes_table(fits_path):
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[1].data)
    return table

def _p16_p50_p84(table, col50):
    if not col50.endswith('_50'):
        raise ValueError(f"Expected a _50 column, got {col50}")
    stem = col50[:-3]
    p50 = table[col50]
    p16 = table[stem + '_16'] if (stem + '_16') in table.colnames else None
    p84 = table[stem + '_84'] if (stem + '_84') in table.colnames else None
    return p16, p50, p84

def plot_EW_vs_UV_colour(table, line_column, line_label, output_path):
    # y
    y16, y50, y84 = _p16_p50_p84(table, line_column)
    # x
    x16, x50, x84 = _p16_p50_p84(table, 'UV_colour_50') if 'UV_colour_50' in table.colnames else (None, table['UV_colour_50'], None)

    burst = table['burstiness_50']
    Ha_rest_50 = table['Halpha_EW_rest_50']

    # validity
    valid = np.isfinite(y50) & np.isfinite(x50) & np.isfinite(burst) & np.isfinite(Ha_rest_50) & (y50 > 0)

    x = x50[valid]
    y = y50[valid]
    burst = burst[valid]
    Ha_rest = Ha_rest_50[valid]

    # asymmetric errors if present
    def asy(err16, mid, err84):
        if err16 is None or err84 is None:
            return None
        err16 = err16[valid]; err84 = err84[valid]
        lower = mid - err16
        upper = err84 - mid
        return np.vstack([lower, upper])

    xerr = asy(x16, x50, x84) if isinstance(x50, np.ndarray) else None
    yerr = asy(y16, y50, y84)

    # coloring rule: special vs other
    special = (burst <= 0.5) & (Ha_rest <= 25)
    other   = ~special

    plt.figure(figsize=(8,6), facecolor='white')
    ax = plt.gca()

    # points only (no per-point error bars)
    ax.scatter(x[other],   y[other],   color='royalblue', alpha=0.5, edgecolor='none', label='All other galaxies')
    ax.scatter(x[special], y[special], color='tomato',    alpha=0.7, edgecolor='none', label='Burstiness ≤ 0.5 and Hα ≤ 25 Å')

    # y on log scale, start at 10^-1.5
    ax.set_yscale('log')
    ymin = 10**-1.5
    ax.set_ylim(ymin, None)

    ax.set_xlabel("UV Colour (mag)")
    ax.set_ylabel(f"{line_label} Equivalent Width (Å)")
    ax.grid(True)

    # representative y error (use your y16/50/84 arrays)
    rep_yerr = _median_err_linear(y16, y50, y84)
    if rep_yerr is not None and np.isfinite(rep_yerr):
        xf, yf = _draw_rep_err(ax, x_frac=0.8, y_frac=0.85, xerr=None, yerr=rep_yerr, color='black')
        _label_rep_err(ax, xf, yf, text="Typical error bar", dx=-0.06, dy=0.05)


    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_Halpha_vs_OIII(table, output_path):
    z = table['input_redshift']

    # use observed percentiles, convert to rest after
    Ha16, Ha50, Ha84 = _p16_p50_p84(table, 'Halpha_EW_obs_50')
    O316, O350, O384 = _p16_p50_p84(table, 'OIII_5007_EW_obs_50')

    burst = table['burstiness_50']
    Ha_rest_50 = table['Halpha_EW_rest_50']

    valid = (
        np.isfinite(z) & np.isfinite(Ha50) & np.isfinite(O350) &
        np.isfinite(burst) & np.isfinite(Ha_rest_50) &
        (Ha50 > 0) & (O350 > 0)
    )

    # convert to rest
    Ha  = (Ha50 / (1 + z))[valid]
    O3  = (O350 / (1 + z))[valid]
    burst = burst[valid]
    Ha_rest = Ha_rest_50[valid]

    # Representative error (median of all)
    Ha_err = np.median((Ha84[valid] - Ha16[valid]) / (2 * (1 + z[valid])))
    O3_err = np.median((O384[valid] - O316[valid]) / (2 * (1 + z[valid])))

    special = (burst <= 0.5) & (Ha_rest <= 25)
    other   = ~special

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(O3[other],   Ha[other],   color='royalblue', alpha=0.5, edgecolor='none', label='All other galaxies')
    ax.scatter(O3[special], Ha[special], color='tomato',    alpha=0.7, edgecolor='none', label='Burstiness ≤ 0.5 and Hα ≤ 25 Å')

    # log–log, lower bounds at 10^-1.5
    xmin = 10**-1.5; ymin = 10**-1.5
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(xmin, None); ax.set_ylim(ymin, None)

    ax.set_xlabel("[OIII] 5007 EW (rest-frame Å)")
    ax.set_ylabel("Hα EW (rest-frame Å)")
    ax.grid(True)

    # representative errors: use medians of rest-frame errors
    rep_xerr = _median_err_linear((O316/(1+z))[valid], (O350/(1+z))[valid], (O384/(1+z))[valid])
    rep_yerr = _median_err_linear((Ha16/(1+z))[valid], (Ha50/(1+z))[valid], (Ha84/(1+z))[valid])
    if (rep_xerr is not None) or (rep_yerr is not None):
        xf, yf = _draw_rep_err(ax, x_frac=0.10, y_frac=0.78, xerr=rep_xerr, yerr=rep_yerr, color='black')
        _label_rep_err(ax, 0.7, 0.6, text="Typical error bar", dx=0.01, dy=0.02)


    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_HaNII_vs_OIIIHb(table, output_path):
    z = table['input_redshift']
    burst = table['burstiness_50']
    Ha_rest_50 = table['Halpha_EW_rest_50']

    # components, observed percentiles
    comps_left  = ['Halpha_EW_obs', 'NII_6548_EW_obs', 'NII_6584_EW_obs']
    comps_right = ['OIII_5007_EW_obs', 'OIII_4959_EW_obs', 'Hbeta_EW_obs']

    def sum_p(table, comps, suffix):
        return np.sum([table[f"{c}_{suffix}"] for c in comps], axis=0)

    L16, L50, L84 = sum_p(table, comps_left, '16'), sum_p(table, comps_left, '50'), sum_p(table, comps_left, '84')
    R16, R50, R84 = sum_p(table, comps_right,'16'), sum_p(table, comps_right,'50'), sum_p(table, comps_right,'84')

    valid = (
        np.isfinite(z) & np.isfinite(L50) & np.isfinite(R50) &
        np.isfinite(burst) & np.isfinite(Ha_rest_50) &
        (L50 > 0) & (R50 > 0)
    )

    # to rest
    X = (R50 / (1 + z))[valid]
    Y = (L50 / (1 + z))[valid]
    burst = burst[valid]
    Ha_rest = Ha_rest_50[valid]

    def asy_sum_rest(p16, p50, p84):
        p16 = (p16 / (1 + z))[valid]
        p84 = (p84 / (1 + z))[valid]
        mid = (p50 / (1 + z))[valid]
        lower = mid - p16
        upper = p84 - mid
        return np.vstack([lower, upper])

    xerr = asy_sum_rest(R16, R50, R84)
    yerr = asy_sum_rest(L16, L50, L84)

    special = (burst <= 0.5) & (Ha_rest <= 25)
    other   = ~special

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(X[other],   Y[other],   color='royalblue', alpha=0.5, edgecolor='none', label='All other galaxies')
    ax.scatter(X[special], Y[special], color='tomato',    alpha=0.7, edgecolor='none', label='Burstiness ≤ 0.5 and Hα ≤ 25 Å')

    ax.set_xscale('log'); ax.set_yscale('log')
    lim = 10**-1.5
    ax.set_xlim(lim, None); ax.set_ylim(lim, None)

    ax.set_xlabel("[OIII] + Hβ EW (rest-frame Å)")
    ax.set_ylabel("Hα + [NII] EW (rest-frame Å)")
    ax.grid(True)

    # representative errors from summed components in rest frame
    rep_xerr = _median_err_linear((R16/(1+z))[valid], (R50/(1+z))[valid], (R84/(1+z))[valid])
    rep_yerr = _median_err_linear((L16/(1+z))[valid], (L50/(1+z))[valid], (L84/(1+z))[valid])
    xf, yf = _draw_rep_err(ax, x_frac=0.10, y_frac=0.78, xerr=rep_xerr, yerr=rep_yerr, color='black')
    _label_rep_err(ax, 0.7, 0.6, text="Typical error bar", dx=0.01, dy=0.02)


    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()





def main():
    fits_path = "/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits"
    table_pipes = load_bagpipes_table(fits_path)

    # Plot [OIII] EW
    plot_EW_vs_UV_colour(
        table=table_pipes,
        line_column='OIII_5007_EW_rest_50',
        line_label='[OIII] 5007 Å',
        output_path="OIII_EW_vs_UV_colour.png"
    )

    # Plot Hα EW
    plot_EW_vs_UV_colour(
        table=table_pipes,
        line_column='Halpha_EW_rest_50',
        line_label='Hα',
        output_path="Halpha_EW_vs_UV_colour.png"
    )

    # Plot Hα vs [OIII]
    plot_Halpha_vs_OIII(
        table=table_pipes,
        output_path="Halpha_vs_OIII_EW.png"
    )

    # Plot Hα + [NII] vs [OIII] + Hβ
    plot_HaNII_vs_OIIIHb(
        table=table_pipes,
        output_path="HaNII_vs_OIIIHb_EW.png"
    )



if __name__ == "__main__":
    main()
