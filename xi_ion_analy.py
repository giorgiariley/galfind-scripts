import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table


def load_bagpipes_table(fits_path):
    """Load the Bagpipes output table from a FITS file."""
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[1].data)
    return table


def plot_xiion_against(table, x_column, output_path, xlabel=None, xlim=None, ylim=None):
    """
    General-purpose function to plot ξ_ion (case B, rest-frame) against any other column.
    Points with burstiness < 2 are colour-coded differently.
    Allows optional axis clipping to remove outliers.
    """
    xi_ion = table['xi_ion_caseB_rest_50']
    x = table[x_column]
    burstiness = table['burstiness_50']

    # Validity mask
    valid = np.isfinite(xi_ion) & np.isfinite(x) & np.isfinite(burstiness)

    # Colour based on burstiness < 2
    colours = np.where(burstiness[valid] < 2, 'crimson', 'teal')

    plt.figure(figsize=(8, 6), facecolor='white')
    plt.scatter(x[valid], xi_ion[valid], alpha=0.6, c=colours, edgecolor='none')

    plt.xlabel(xlabel if xlabel else x_column.replace("_", " ").capitalize())
    plt.ylabel(r"$\xi_{\mathrm{ion}}^{\mathrm{CaseB}}$ (Hz erg$^{-1}$)")
    plt.title(fr"$\xi_{{\rm ion}}$ vs {xlabel if xlabel else x_column}")
    plt.grid(True)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Legend handle for colours
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Burstiness < 2',
               markerfacecolor='crimson', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Burstiness ≥ 2',
               markerfacecolor='teal', markersize=8)
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📁 Saved plot to: {output_path}")


# Load FITS table
fits_path = "/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits"
table = load_bagpipes_table(fits_path)

# Add summed EW columns
table['Ha_NII_EW'] = (
    table['Halpha_EW_rest_50'] +
    table['NII_6548_EW_rest_50'] +
    table['NII_6584_EW_rest_50']
)

table['Hb_OIII_EW'] = (
    table['Hbeta_EW_rest_50'] +
    table['OIII_4959_EW_rest_50'] +
    table['OIII_5007_EW_rest_50']
)

# Define emission line columns
ew_columns = [
    'Halpha_EW_rest_50',
    'Hbeta_EW_rest_50',
    'OIII_5007_EW_rest_50',
    'OIII_4959_EW_rest_50',
    'NII_6584_EW_rest_50',
    'NII_6548_EW_rest_50',
]

# General plots
plot_xiion_against(table, 'burstiness_50', 'xiion_vs_burstiness.png', xlabel="Burstiness (SFR ratio)")
plot_xiion_against(table, 'stellar_mass_50', 'xiion_vs_mass.png', xlabel=r"log$_{10}$(Mass [$M_\odot$])")
plot_xiion_against(table, 'OIII_5007_EW_rest_50', 'xiion_vs_OIII.png', xlabel="[OIII] 5007 EW (rest-frame Å)")

# Loop over and generate plots for EWs with specific x-axis clipping
for col in ew_columns:
    output_file = f"xiion_vs_{col}.png"
    nice_label = col.replace("_EW_rest_50", "").replace("_", " ") + " EW (rest-frame Å)"

    # Custom x-axis clipping
    if col == "NII_6584_EW_rest_50":
        plot_xiion_against(table, col, output_file, xlabel=nice_label, xlim=(0, 120))
    elif col == "NII_6548_EW_rest_50":
        plot_xiion_against(table, col, output_file, xlabel=nice_label, xlim=(0, 50))
    else:
        plot_xiion_against(table, col, output_file, xlabel=nice_label)

# Combined EW plots
plot_xiion_against(
    table, 'Ha_NII_EW', 'xiion_vs_Ha+NII.png',
    xlabel='Hα + [NII] EW (rest-frame Å)', xlim=(0, 2500)
)

plot_xiion_against(
    table, 'Hb_OIII_EW', 'xiion_vs_Hb+OIII.png',
    xlabel='Hβ + [OIII] EW (rest-frame Å)'
)

