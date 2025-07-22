import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# === Constants ===
PIXEL_SCALE = 0.03  # arcsec/pixel

def load_bagpipes_table(fits_path):
    """Load the Bagpipes output table from a FITS file."""
    with fits.open(fits_path) as hdulist:
        return Table(hdulist[1].data)

def load_galfit_table(fits_path):
    """Load the GALFIT output table from a FITS file."""
    with fits.open(fits_path) as hdulist:
        return Table(hdulist[1].data)

def load_redshifts(balmer_file):
    """Load redshifts from a balmer_breaks.txt file."""
    data = np.loadtxt(balmer_file, skiprows=1)
    indices = data[:, 0].astype(int)
    redshifts = data[:, 2]
    return indices, redshifts
def plot_size_vs_burstiness(table_pipes, table_galfit, redshifts, output_path):
    """
    Plot galaxy size (kpc) vs. burstiness, removing unreliable fits.
    Highlights galaxies with OIII_EW_rest < 200, UV < 0, and burstiness < 1.
    """

    # Extract GALFIT data
    radius_pix = table_galfit['r_e']
    radius_arcsec = radius_pix * PIXEL_SCALE
    sersic_index = table_galfit['n']
    chi2_red = table_galfit['red_chi2']

    # Convert to kpc
    DA_kpc_per_arcsec = np.array([
        (cosmo.angular_diameter_distance(z).to(u.kpc) / u.radian.to(u.arcsec)).value
        for z in redshifts
    ])
    radius_kpc = radius_arcsec * DA_kpc_per_arcsec

    # Extract Bagpipes outputs
    burstiness = table_pipes['burstiness_50']
    UV = table_pipes['UV_colour_50']
    O3_EW_obs = table_pipes['OIII_5007_EW_obs_50']
    z_pipes = table_pipes['input_redshift']

    # Convert observed EW to rest-frame
    O3_EW_rest = O3_EW_obs / (1 + z_pipes)

    # === Count before filtering ===
    mask_special_all = (O3_EW_rest < 200) & (UV < 0) & (burstiness < 1)
    mask_burstiness_low_all = (burstiness < 1)

    print(f"🔍 Special subset (OIII_rest < 200, UV < 0, burstiness < 1) BEFORE filtering: {np.sum(mask_special_all)}")
    print(f"🔍 Galaxies with burstiness < 1 BEFORE filtering: {np.sum(mask_burstiness_low_all)}")

    # === Filtering unreliable fits ===
    flag_bad_z = (redshifts <= 0.01) | (redshifts >= 16)
    flag_bad_radius = (radius_kpc <= 0.01) | (radius_kpc > 10)
    flag_bad_n = sersic_index >= 10
    flag_bad_chi2 = chi2_red > 5
    flag_unreliable = flag_bad_z | flag_bad_radius | flag_bad_n | flag_bad_chi2

    valid = (
        np.isfinite(radius_kpc) & np.isfinite(burstiness) &
        np.isfinite(UV) & np.isfinite(O3_EW_rest) & ~flag_unreliable
    )

    # Apply filtering
    radius_kpc = radius_kpc[valid]
    burstiness = burstiness[valid]
    UV = UV[valid]
    O3_EW_rest = O3_EW_rest[valid]

    # === Count after filtering ===
    mask_special = (O3_EW_rest < 200) & (UV < 0) & (burstiness < 1)
    mask_burstiness_low = (burstiness < 1)

    print(f"✅ Special subset AFTER filtering: {np.sum(mask_special)}")
    print(f"✅ Galaxies with burstiness < 1 AFTER filtering: {np.sum(mask_burstiness_low)}")

    # === Plot ===
    plt.figure(figsize=(8, 6), facecolor='white')
    plt.scatter(radius_kpc[~mask_special], burstiness[~mask_special],
                color='mediumslateblue', alpha=0.7, edgecolor='none', label='Other galaxies')
    plt.scatter(radius_kpc[mask_special], burstiness[mask_special],
                color='crimson', alpha=0.8, edgecolor='none', label='Low OIII, blue UV, low burstiness')

    plt.xlabel("Size (kpc)")
    plt.xscale("log")
    plt.ylabel("Burstiness")
    plt.title("Galaxy Size vs. Burstiness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📁 Saved colour-coded plot to: {output_path}")


def plot_OIII_EW_vs_burstiness(table_pipes, output_path):
    """
    Plot rest-frame [OIII] 5007 Å EW vs burstiness, colour-coded by UV colour.
    """

    z_pipes = table_pipes['input_redshift']
    O3_EW_obs = table_pipes['OIII_5007_EW_obs_50']
    burstiness = table_pipes['burstiness_50']
    UV_colour = table_pipes['UV_colour_50']

    # Calculate rest-frame EW
    O3_EW_rest = O3_EW_obs / (1 + z_pipes)

    # Filter valid data
    valid = (
        (O3_EW_rest > 0) & 
        np.isfinite(O3_EW_rest) & 
        np.isfinite(burstiness) & 
        np.isfinite(UV_colour)
    )

    O3_EW_rest = O3_EW_rest[valid]
    burstiness = burstiness[valid]
    UV_colour = UV_colour[valid]

    plt.figure(figsize=(8, 6), facecolor='white')
    sc = plt.scatter(
        burstiness, O3_EW_rest,
        c=UV_colour, cmap='viridis', alpha=0.7, edgecolor='none'
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("UV Colour (mag)")

    plt.xlabel("Burstiness (SFR ratio)")
    plt.ylabel("[OIII] 5007 Å Equivalent Width (rest-frame Å)")
    plt.title("[OIII] 5007 Å EW vs Burstiness (Colour-coded by UV)")
    plt.grid(True)

    # Reasonable axis limits
    plt.xlim(0, np.percentile(burstiness, 99))
    plt.ylim(0, np.percentile(O3_EW_rest, 99))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📁 Saved [OIII] EW vs Burstiness plot to: {output_path}")


def main():
    fits_pipes = "/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits"
    fits_galfit = "/raid/scratch/work/Griley/GALFIND_WORK/GALFIT/output/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/F444W/sersic/results.fits"
    balmer_file = "/nvme/scratch/work/Griley/galfind_scripts/Balmerbreak/balmer_break_outputs1/balmer_breaks.txt"

    table_pipes = load_bagpipes_table(fits_pipes)
    table_galfit = load_galfit_table(fits_galfit)
    _, redshifts = load_redshifts(balmer_file)

    plot_size_vs_burstiness(table_pipes, table_galfit, redshifts, output_path="size_vs_burstiness_extreme.png")

    plot_OIII_EW_vs_burstiness(table_pipes, output_path="OIII_EW_vs_burstiness.png")


if __name__ == "__main__":
    main()
