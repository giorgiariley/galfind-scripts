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
    other = ~special

    plt.figure(figsize=(8,6), facecolor='white')
    # red points and errors
    plt.scatter(x[other], y[other], color='royalblue', alpha=0.5, edgecolor='none', label='All other galaxies')
    if xerr is not None or yerr is not None:
        plt.errorbar(
            x[other], y[other],
            xerr=xerr[:, other] if xerr is not None else None,
            yerr=yerr[:, other] if yerr is not None else None,
            fmt='none', ecolor='royalblue', elinewidth=0.8, capsize=2, alpha=0.25
        )

    # blue points and errors on top
    plt.scatter(x[special], y[special], color='tomato', alpha=0.7, edgecolor='none', label='Burstiness ≤ 0.5 and Hα ≤ 25 Å')
    if xerr is not None or yerr is not None:
        plt.errorbar(
            x[special], y[special],
            xerr=xerr[:, special] if xerr is not None else None,
            yerr=yerr[:, special] if yerr is not None else None,
            fmt='none', ecolor='tomato', elinewidth=0.8, capsize=2, alpha=0.25
        )

    plt.xlabel("UV Colour (mag)")
    plt.ylabel(f"{line_label} Equivalent Width (Å)")
    # plt.yscale('log')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(output_path, dpi=200); plt.close()
    print(f"Saved plot to: {output_path}")

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

    def asy_rest(p16, p50, p84):
        if p16 is None or p84 is None:
            return None
        p16 = (p16 / (1 + z))[valid]; p84 = (p84 / (1 + z))[valid]
        lower = p50[valid]/(1+z[valid]) - p16
        upper = p84 - p50[valid]/(1+z[valid])
        return np.vstack([lower, upper])

    xerr = asy_rest(O316, O350, O384)
    yerr = asy_rest(Ha16, Ha50, Ha84)

    special = (burst <= 0.5) & (Ha_rest <= 25)
    other = ~special

    plt.figure(figsize=(8,6), facecolor='white')
    # red points and errors
    plt.scatter(O3[other], Ha[other], color='royalblue', alpha=0.5, edgecolor='none', label='All other galaxies')
    if xerr is not None or yerr is not None:
        plt.errorbar(
            O3[other], Ha[other],
            xerr=xerr[:, other] if xerr is not None else None,
            yerr=yerr[:, other] if yerr is not None else None,
            fmt='none', ecolor='royalblue', elinewidth=0.8, capsize=2, alpha=0.25
        )

    # blue points and errors on top
    plt.scatter(O3[special], Ha[special], color='tomato', alpha=0.7, edgecolor='none', label='Burstiness ≤ 0.5 and Hα ≤ 25 Å')
    if xerr is not None or yerr is not None:
        plt.errorbar(
            O3[special], Ha[special],
            xerr=xerr[:, special] if xerr is not None else None,
            yerr=yerr[:, special] if yerr is not None else None,
            fmt='none', ecolor='tomato', elinewidth=0.8, capsize=2, alpha=0.25
        )

    plt.xlabel("[OIII] 5007 EW (rest-frame Å)")
    plt.ylabel("Hα EW (rest-frame Å)")
    # plt.yscale('log')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(output_path, dpi=200); plt.close()
    print(f"📁 Saved Hα vs. [OIII] rest-frame plot to: {output_path}")

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
    other = ~special
    print(len(X[special]), len(Y[special]), len(X[other]), len(Y[other]))
    plt.figure(figsize=(8,6), facecolor='white')
    # red points and errors
    plt.scatter(X[other], Y[other], color='royalblue', alpha=0.5, edgecolor='none', label='All other galaxies')
    if xerr is not None or yerr is not None:
        plt.errorbar(
            X[other], Y[other],
            xerr=xerr[:, other] if xerr is not None else None,
            yerr=yerr[:, other] if yerr is not None else None,
            fmt='none', ecolor='royalblue', elinewidth=0.8, capsize=2, alpha=0.25
        )

    # blue points and errors on top
    plt.scatter(X[special], Y[special], color='tomato', alpha=0.7, edgecolor='none', label='Burstiness ≤ 0.5 and Hα ≤ 25 Å')
    if xerr is not None or yerr is not None:
        plt.errorbar(
            X[special], Y[special],
            xerr=xerr[:, special] if xerr is not None else None,
            yerr=yerr[:, special] if yerr is not None else None,
            fmt='none', ecolor='tomato', elinewidth=0.8, capsize=2, alpha=0.25
        )

    plt.xlabel("[OIII] + Hβ EW (rest-frame Å)")
    plt.ylabel("Hα + [NII] EW (rest-frame Å)")
    # plt.yscale('log')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(output_path, dpi=200); plt.close()
    print(f"📁 Saved plot with error bars to: {output_path}")





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
