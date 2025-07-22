import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

def load_bagpipes_table(fits_path):
    """Load the Bagpipes output table from a FITS file."""
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[1].data)
    return table

def plot_EW_vs_UV_colour(table, line_column, line_label, output_path):
    """
    Generic plotting function for emission line EW vs. UV colour.
    
    Parameters:
    - table: astropy Table
    - line_column: name of the EW column in table (e.g., 'OIII_5007_EW_obs_50')
    - line_label: label to display on y-axis (e.g., '[OIII] 5007 Å')
    - output_path: where to save the plot
    """
    EWs_obs = table[line_column]
    UV = table['UV_colour_50']
    burstiness = table['burstiness_50']
    

    valid = (EWs_obs > 0) & np.isfinite(EWs_obs) & np.isfinite(burstiness)
    EWs = table[line_column] / (1 + table['input_redshift'])

    # Additional filter for Hα outliers
    if line_column.lower().startswith("halpha"):
        valid &= EWs < 30000  # remove extreme values

    EWs = EWs[valid]
    UV = UV[valid]
    burstiness = burstiness[valid]

    mask_low_b = burstiness < 1
    mask_high_b = burstiness >= 1

    plt.figure(figsize=(8, 6), facecolor='white')
    plt.scatter(UV[mask_low_b], EWs[mask_low_b], color='steelblue', alpha=0.6, label='SFR ratio < 1', edgecolor='none')
    plt.scatter(UV[mask_high_b], EWs[mask_high_b], color='tomato', alpha=0.6, label='SFR ratio ≥ 1', edgecolor='none')

    plt.xlabel("UV Colour (mag)")
    plt.ylabel(f"{line_label} Equivalent Width (Å)")
    plt.title(f"{line_label} EW vs. UV Colour")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to: {output_path}")

def plot_Halpha_vs_OIII(table, output_path):
    """
    Plot Hα EW vs. [OIII] EW in rest-frame, coloured by burstiness.
    """
    z = table['input_redshift']
    EW_Halpha_obs = table['Halpha_EW_obs_50']
    EW_OIII_obs = table['OIII_5007_EW_obs_50']
    burstiness = table['burstiness_50']

    # Convert to rest-frame
    EW_Halpha_rest = EW_Halpha_obs / (1 + z)
    EW_OIII_rest = EW_OIII_obs / (1 + z)

    # Validity mask
    valid = (
        np.isfinite(EW_Halpha_rest) & np.isfinite(EW_OIII_rest) & np.isfinite(burstiness) &
        (EW_Halpha_rest > 0) & (EW_OIII_rest > 0) &
        (EW_Halpha_rest < 3000)  # Adjusted threshold for rest-frame
    )

    EW_Halpha_rest = EW_Halpha_rest[valid]
    EW_OIII_rest = EW_OIII_rest[valid]
    burstiness = burstiness[valid]

    # Burstiness mask
    mask_low_b = burstiness < 1
    mask_high_b = burstiness >= 1

    # Plotting
    plt.figure(figsize=(8, 6), facecolor='white')
    plt.scatter(EW_OIII_rest[mask_low_b], EW_Halpha_rest[mask_low_b],
                color='steelblue', alpha=0.6, label='SFR ratio < 1', edgecolor='none')
    plt.scatter(EW_OIII_rest[mask_high_b], EW_Halpha_rest[mask_high_b],
                color='tomato', alpha=0.6, label='SFR ratio ≥ 1', edgecolor='none')

    plt.xlabel("[OIII] 5007 Å Equivalent Width (rest-frame Å)")
    plt.ylabel("Hα Equivalent Width (rest-frame Å)")
    plt.title("Hα vs. [OIII] Equivalent Width (Rest-frame)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📁 Saved Hα vs. [OIII] rest-frame plot to: {output_path}")

def plot_HaNII_vs_OIIIHb(table, output_path):
    """
    Plot (Hα + [NII]) vs ([OIII] + Hβ) in rest-frame with 1σ error bars,
    coloured by burstiness. EWs are summed, and errors are combined in quadrature.
    """

    # === Extract observed EWs and percentiles ===
    Ha_50 = table['Halpha_EW_obs_50']
    NII_6548_50 = table['NII_6548_EW_obs_50']
    NII_6584_50 = table['NII_6584_EW_obs_50']
    OIII_5007_50 = table['OIII_5007_EW_obs_50']
    OIII_4959_50 = table['OIII_4959_EW_obs_50']
    Hb_50 = table['Hbeta_EW_obs_50']

    # Error estimates (1σ from percentiles)
    def error(line):
        return 0.5 * (table[f"{line}_EW_obs_84"] - table[f"{line}_EW_obs_16"])

    Ha_err = error("Halpha")
    NII_6548_err = error("NII_6548")
    NII_6584_err = error("NII_6584")
    OIII_5007_err = error("OIII_5007")
    OIII_4959_err = error("OIII_4959")
    Hb_err = error("Hbeta")

    # === Sum EWs ===
    EW_Ha_NII_obs = Ha_50 + NII_6548_50 + NII_6584_50
    EW_OIII_Hb_obs = OIII_5007_50 + OIII_4959_50 + Hb_50

    # === Combine errors in quadrature ===
    err_Ha_NII_obs = np.sqrt(Ha_err**2 + NII_6548_err**2 + NII_6584_err**2)
    err_OIII_Hb_obs = np.sqrt(OIII_5007_err**2 + OIII_4959_err**2 + Hb_err**2)

    z = table['input_redshift']
    burstiness = table['burstiness_50']

    # === Convert to rest-frame ===
    EW_Ha_NII = EW_Ha_NII_obs / (1 + z)
    EW_OIII_Hb = EW_OIII_Hb_obs / (1 + z)
    err_Ha_NII = err_Ha_NII_obs / (1 + z)
    err_OIII_Hb = err_OIII_Hb_obs / (1 + z)

    # === Validity mask ===
    valid = (
        np.isfinite(EW_Ha_NII) & np.isfinite(EW_OIII_Hb) & np.isfinite(burstiness) &
        np.isfinite(err_Ha_NII) & np.isfinite(err_OIII_Hb) &
        (EW_Ha_NII > 0) & (EW_OIII_Hb > 0) &
        ((EW_Ha_NII - err_Ha_NII) > 0) &
        ((EW_OIII_Hb - err_OIII_Hb) > 0)
)


    EW_Ha_NII = EW_Ha_NII[valid]
    EW_OIII_Hb = EW_OIII_Hb[valid]
    err_Ha_NII = err_Ha_NII[valid]
    err_OIII_Hb = err_OIII_Hb[valid]
    burstiness = burstiness[valid]

    # === Colour-code by burstiness ===
    mask_low_b =  (burstiness < 2) & (burstiness >= 0.5)
    mask_high_b = burstiness >= 2
    mask_lowest_b = burstiness < 0.5

    # === Plot ===
    plt.figure(figsize=(8, 6), facecolor='white')

    # Low burstiness (blue)
    plt.errorbar(EW_OIII_Hb[mask_low_b], EW_Ha_NII[mask_low_b],
                 xerr=err_OIII_Hb[mask_low_b], yerr=err_Ha_NII[mask_low_b],
                 fmt='o', color='steelblue', alpha=0.5, label='burstiness < 2', capsize=2)

    # High burstiness (red)
    plt.errorbar(EW_OIII_Hb[mask_high_b], EW_Ha_NII[mask_high_b],
                 xerr=err_OIII_Hb[mask_high_b], yerr=err_Ha_NII[mask_high_b],
                 fmt='o', color='tomato', alpha=0.5, label='burstiness ≥ 2', capsize=2)
    
    # Lowest burstiness (green)
    plt.errorbar(EW_OIII_Hb[mask_lowest_b], EW_Ha_NII[mask_lowest_b],
                 xerr=err_OIII_Hb[mask_lowest_b], yerr=err_Ha_NII[mask_lowest_b],
                 fmt='o', color='forestgreen', alpha=0.5, label='burstiness < 0.5', capsize=2)

    plt.xlabel("[OIII] + Hβ Equivalent Width (rest-frame Å)")
    plt.ylabel("Hα + [NII] Equivalent Width (rest-frame Å)")
    plt.title("(Hα + [NII]) vs ([OIII] + Hβ) EW with Burstiness Colour Coding")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📁 Saved plot with error bars to: {output_path}")




def main():
    fits_path = "/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits"
    table_pipes = load_bagpipes_table(fits_path)

    # Plot [OIII] EW
    plot_EW_vs_UV_colour(
        table=table_pipes,
        line_column='OIII_5007_EW_obs_50',
        line_label='[OIII] 5007 Å',
        output_path="OIII_EW_vs_UV_colour.png"
    )

    # Plot Hα EW
    plot_EW_vs_UV_colour(
        table=table_pipes,
        line_column='Halpha_EW_obs_50',
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
