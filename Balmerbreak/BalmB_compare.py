import astropy.units as u
from typing import Union
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
from galfind.Data import morgan_version_to_dir
from galfind import Catalogue, EAZY, SED_code
from galfind import galfind_logger, Multiple_Mask_Selector, Multiple_SED_fit_Selector, Min_Instrument_Unmasked_Band_Selector, Unmasked_Band_Selector, Bluewards_LyLim_Non_Detect_Selector, Bluewards_Lya_Non_Detect_Selector, Redwards_Lya_Detect_Selector, Chi_Sq_Lim_Selector, Chi_Sq_Diff_Selector, Robust_zPDF_Selector, Sextractor_Bands_Radius_Selector    
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

# Load Balmer break file
balmer_file = "balmer_break_outputs/balmer_breaks.txt"
data = np.loadtxt(balmer_file, skiprows=1)  # skip header

indices = data[:, 0].astype(int)            # galaxy indices
balmer_breaks = data[:, 1]                   # Balmer break strength (mag)
# redshifts = data[:, 2]                      # if needed

#Plot it for my bagpipes output or so that I have right?
# Load the file
hdulist = fits.open("/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits")
table_bagpipes = Table(hdulist[1].data)
hdulist.close()

# Bagpipes variables of interest
variables = {
    "burstiness_50": "Burstiness",
    "sfr_50": "Star Formation Rate [M☉/yr]",
    "ssfr_50": "Specific SFR [yr⁻¹]",
    "stellar_mass_50" : "Stellar Mass [M☉]",
}

# Create output directory for plots
os.makedirs("balmer_break_outputs/comparison_plots", exist_ok=True)

# Loop through variables and plot
for var_key, var_label in variables.items():
    y = table_bagpipes[var_key]
    valid = (
            (~np.isnan(balmer_breaks)) & 
            (~np.isnan(y)) & 
            (balmer_breaks != 0) & 
            (y != 0)
        )

    if np.sum(valid) == 0:
        print(f"No valid data for {var_key}, skipping.")
        continue

    plt.figure(figsize=(8, 6))
    plt.scatter(y[valid], balmer_breaks[valid], alpha=0.7)

    # Apply log scale for SFR and sSFR only
    if var_key in ["sfr_50", "ssfr_50"]:
        plt.xscale("log")

    plt.ylabel("Balmer Break Strength (mag)")
    plt.xlabel(var_label)
    plt.title(f"Balmer Break vs {var_label}")
    plt.grid(True)
    plt.tight_layout()
    plot_path = f"balmer_break_outputs/comparison_plots/balmer_vs_{var_key}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot: {plot_path}")





