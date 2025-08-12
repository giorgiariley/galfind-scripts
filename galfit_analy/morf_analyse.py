import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.io import fits

def load_bagpipes_table(fits_path):
    """Load the Bagpipes output table from a FITS file."""
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[1].data)
    return table

def load_morf_table(filename):
    """Load the Morfometryka table from a FITS file."""
    return Table.read(filename)

def compute_concentration_asymmetry(table):
    """Compute concentration and asymmetry from Morfometryka output."""
    concentration = table['C1'] * 5  # Convert to conventional C
    asymmetry_log = np.log10(table['A0'])  # Already log10(A)
    return concentration, asymmetry_log

def add_bershady_boundaries():
    """Plot Bershady et al. (2000) classification boundaries in log(A)–C space."""
    logA_vals = np.linspace(-2.0, 0.5, 200)
    C_late = 2.44 * logA_vals + 5.49
    plt.plot(C_late, logA_vals, color='black', linestyle='--', linewidth=1)

def make_ca_plot(conc, log_asym, title='Concentration–Asymmetry Diagram',
                 colour=None, label=None, save_path=None, add_bershady=True):
    """Plot concentration vs. log-asymmetry with optional colour coding and Bershady lines."""
    plt.figure(figsize=(7, 6), facecolor='white')

    # Plot non-PSBs first (background)
    non_psb = ~colour
    plt.scatter(conc[non_psb], log_asym[non_psb],
                color='royalblue', alpha=0.7, s=20,
                edgecolor='none', label='Other')

    # Plot PSBs on top
    plt.scatter(conc[colour], log_asym[colour],
                color='tomato', alpha=0.8, s=20,
                edgecolor='none', label='Burstiness<=1, Halpha EW<=100Å')

    if add_bershady:
        add_bershady_boundaries()

    # Add horizontal merger boundary at A = 0.35 (log10 scale)
    merger_threshold = np.log10(0.35)
    plt.axhline(y=merger_threshold, color='black', linestyle='-', linewidth=1)

    # Annotate regions
    plt.text(1.2, -0.15, "Merger", fontsize=12)
    plt.text(1.1, -1.0, "LTG", fontsize=12)
    plt.text(3.0, -1.5, "ETG", fontsize=12)

    plt.xlabel("Concentration (C)")
    plt.ylabel("log₁₀(Asymmetry)")
    plt.title(title)
    plt.xlim(1, 4)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # File paths
    filename_morf = 'combined_morfometryka.fits'
    fits_path_pipes = "/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits"

    # Load data
    morf_table = load_morf_table(filename_morf)
    table_bagpipes = load_bagpipes_table(fits_path_pipes)
    # Convert ID columns to string for matching
    morf_ids = morf_table['# rootname9.65'].astype(str)
    bagpipes_ids = table_bagpipes['#ID'].astype(str)

    # Find common IDs and their indices
    common_ids, morf_idx, bagpipes_idx = np.intersect1d(morf_ids, bagpipes_ids, return_indices=True)

    # Subset both tables to matching rows
    morf_table = morf_table[morf_idx]
    table_bagpipes = table_bagpipes[bagpipes_idx]

    # Filter unphysical values
    A0 = morf_table['A0']
    C1 = morf_table['C1']
    valid = (
        (A0 > 0.01) &
        (A0 < 1.5) &
        ((C1 * 5) > 1.0) &
        ((C1 * 5) < 5.0) &
        np.isfinite(A0) &
        np.isfinite(C1)
    )

    # Apply filter
    morf_table = morf_table[valid]
    table_bagpipes = table_bagpipes[valid]

    # Compute plot inputs
    conc = morf_table['C1'] * 5
    log_asym = np.log10(morf_table['A0'])

    halpha = table_bagpipes['Halpha_EW_rest_50']
    burstiness = table_bagpipes['burstiness_50']
    is_extreme_psb = (halpha <= 100) & (burstiness <= 1)
    make_ca_plot(conc, log_asym,
             title="C–log(A) Plot with Extreme PSBs Highlighted",
             colour=is_extreme_psb,  # Boolean mask
             save_path="concentration_asymmetry_plot.png")
