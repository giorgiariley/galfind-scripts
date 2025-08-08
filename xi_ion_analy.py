import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from matplotlib.lines import Line2D

def load_bagpipes_table(fits_path):
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[1].data)
    return table

def _get_p16_p50_p84(table, col_with_50):
    """
    Returns (p16, p50, p84) arrays for a given column named like 'foo_50'.
    If p16/p84 are missing, returns (None, p50, None).
    """
    if not col_with_50.endswith('_50'):
        raise ValueError(f"Column name must end with _50, got {col_with_50}")
    stem = col_with_50[:-3]  # drop '_50'
    p50 = table[col_with_50]
    p16 = table[stem + '_16'] if (stem + '_16') in table.colnames else None
    p84 = table[stem + '_84'] if (stem + '_84') in table.colnames else None
    return p16, p50, p84

def plot_xiion_against(table, x_column_50, output_path, xlabel=None, xlim=None, ylim=None, highlight_mask=None,
                       capsize=2.0, ealpha=0.25):
    """
    x_column_50 must be the p50 column name, e.g. 'burstiness_50' or 'Ha_NII_EW_50'.
    Draws asymmetric error bars from the _16 and _84 siblings if present.
    """
    # y - get p16/p50/p84, then log10 with asymmetric errors
    y16, y50, y84 = _get_p16_p50_p84(table, 'xi_ion_caseB_rest_50')
    # convert to log10 space for plotting and errors
    y = np.log10(y50)
    yerr = None
    if (y16 is not None) and (y84 is not None):
        # Asymmetric errors in log space
        yerr_lower = np.log10(y50) - np.log10(y16)
        yerr_upper = np.log10(y84) - np.log10(y50)
        yerr = np.vstack([yerr_lower, yerr_upper])

    # x - get p16/p50/p84 if available
    x16, x50, x84 = _get_p16_p50_p84(table, x_column_50)
    x = x50
    xerr = None
    if (x16 is not None) and (x84 is not None):
        xerr_lower = x50 - x16
        xerr_upper = x84 - x50
        xerr = np.vstack([xerr_lower, xerr_upper])

    burstiness = table['burstiness_50']
    halpha_EW = table['Halpha_EW_rest_50']

    # Validity mask
    valid = (
        np.isfinite(y) &
        np.isfinite(x) &
        np.isfinite(burstiness) &
        np.isfinite(halpha_EW)
    )

    colours = np.where(((burstiness[valid] <= 1) & (halpha_EW[valid] <= 100)), 'crimson', 'teal')

    plt.figure(figsize=(8, 6), facecolor='white')

    # Scatter first
    plt.scatter(x[valid], y[valid], alpha=0.6, c=colours, edgecolor='none')

    # Error bars - draw per colour group so ecolor matches
    if (xerr is not None) or (yerr is not None):
        mask_crimson = valid.copy()
        mask_crimson[valid] = colours == 'crimson'
        mask_teal = valid.copy()
        mask_teal[valid] = colours == 'teal'

        for m, ecolor in [(mask_crimson, 'crimson'), (mask_teal, 'teal')]:
            if np.any(m):
                plt.errorbar(
                    x[m], y[m],
                    xerr=xerr[:, m] if xerr is not None else None,
                    yerr=yerr[:, m] if yerr is not None else None,
                    fmt='none', ecolor=ecolor, elinewidth=0.8, alpha=ealpha, capsize=capsize
                )

    # Overlay highlighted sample
    if highlight_mask is not None:
        highlight_valid = valid & highlight_mask
        plt.scatter(
            x[highlight_valid], y[highlight_valid],
            edgecolor='black', facecolor='gold', s=80, linewidth=1.0,
            marker='*', label='Selected: low BB, low $\\xi_{\\rm ion}$, low $\\beta$'
        )

    plt.xlabel(xlabel if xlabel else x_column_50.replace("_", " ").capitalize())
    plt.ylabel(r"$\xi_{\mathrm{ion}}^{\mathrm{CaseB}}$ (Hz erg$^{-1}$)")
    plt.title(fr"$\xi_{{\rm ion}}$ vs {xlabel if xlabel else x_column_50}")
    plt.grid(True)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Burstiness <= 1 & Halpha EW < 100Å ',
               markerfacecolor='crimson', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Burstiness > 1 & Halpha EW > 100Å',
               markerfacecolor='teal', markersize=8)
    ]
    plt.legend(handles=legend_elements + (
        [Line2D([0], [0], marker='*', color='w', label='Selected extremes',
                markerfacecolor='gold', markeredgecolor='black', markersize=12)] if highlight_mask is not None else []
    ))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"📁 Saved plot with error bars to: {output_path}")



# Load FITS table
fits_path = "/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits"
table = load_bagpipes_table(fits_path)
# Load Balmer break info
balmer_data = np.loadtxt('/nvme/scratch/work/Griley/galfind_scripts/Balmerbreak/Balmer_output/balmer_breaks2.txt', skiprows=1)
balmer_ids_str = np.array([str(int(x)) for x in balmer_data[:, 1]])
balmer_breaks = balmer_data[:, 2]

# Match to Bagpipes table
table_ids = np.array([str(x).strip() for x in table['#ID']])
id_to_bb = dict(zip(balmer_ids_str, balmer_breaks))

# Build mask aligned with table
bb_array = np.array([id_to_bb.get(str(x).strip(), np.nan) for x in table['#ID']])
log_xi_ion = np.log10(table['xi_ion_caseB_rest_50'])
UV_beta = table['beta_C94_50']

# highlight_mask = (
#     (bb_array <= 0.35) &
#     (log_xi_ion <= 24.6) &
#     (UV_beta <= -2.6)
# )

# Add summed EW percentile columns
for stem, parts in [
    ('Ha_NII_EW', ['Halpha_EW_rest', 'NII_6548_EW_rest', 'NII_6584_EW_rest']),
    ('Hb_OIII_EW', ['Hbeta_EW_rest', 'OIII_4959_EW_rest', 'OIII_5007_EW_rest']),
]:
    for p in ['16', '50', '84']:
        table[f'{stem}_{p}'] = np.sum([table[f'{part}_{p}'] for part in parts], axis=0)


plot_xiion_against(table, 'burstiness_50', 'xiion_vs_burstiness.png', xlabel="Burstiness (SFR ratio)")
plot_xiion_against(table, 'stellar_mass_50', 'xiion_vs_mass.png', xlabel=r"log$_{10}$(Mass [$M_\odot$])")
plot_xiion_against(table, 'OIII_5007_EW_rest_50', 'xiion_vs_OIII.png', xlabel="[OIII] 5007 EW (rest-frame Å)")

for col in [
    'Halpha_EW_rest_50','Hbeta_EW_rest_50','OIII_5007_EW_rest_50',
    'OIII_4959_EW_rest_50','NII_6584_EW_rest_50','NII_6548_EW_rest_50'
]:
    nice_label = col.replace("_EW_rest_50", "").replace("_", " ") + " EW (rest-frame Å)"
    if col == "NII_6584_EW_rest_50":
        plot_xiion_against(table, col, f"xiion_vs_{col}.png", xlabel=nice_label, xlim=(0, 120))
    elif col == "NII_6548_EW_rest_50":
        plot_xiion_against(table, col, f"xiion_vs_{col}.png", xlabel=nice_label, xlim=(0, 50))
    elif col == "Halpha_EW_rest_50":
        plot_xiion_against(table, col, f"xiion_vs_{col}.png", xlabel=nice_label, xlim=(0, 2500))
    else:
        plot_xiion_against(table, col, f"xiion_vs_{col}.png", xlabel=nice_label)

# Combined EWs - now use the _50 columns you just created
plot_xiion_against(table, 'Ha_NII_EW_50', 'xiion_vs_Ha+NII.png',
                   xlabel='Hα + [NII] EW (rest-frame Å)', xlim=(0, 2500))
plot_xiion_against(table, 'Hb_OIII_EW_50', 'xiion_vs_Hb+OIII.png',
                   xlabel='Hβ + [OIII] EW (rest-frame Å)')
