import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

def load_morf_table(filename):
    """Load the Morfometryka table from a FITS file."""
    return Table.read(filename)

def compute_concentration_asymmetry(table):
    """Compute concentration and asymmetry from Morfometryka output."""
    concentration = table['C1'] * 5  # Convert to conventional C
    asymmetry_log = np.log10(table['A0'] )     # Already log10(A)
    return concentration, asymmetry_log

def add_bershady_boundaries():
    """Plot Bershady et al. (2000) classification boundaries in log(A)–C space."""
    logA_vals = np.linspace(-2.0, 0.5, 200)

    # C as a function of log A
    C_early_intermediate = 21.5 * logA_vals + 31.2
    C_late = 2.44 * logA_vals + 5.49

    plt.plot(C_early_intermediate, logA_vals, 'r--', label='Early/Intermediate boundary')
    plt.plot(C_late, logA_vals, 'b--', label='Late-type boundary')

def make_ca_plot(conc, log_asym, title='Concentration–Asymmetry Diagram',
                 colour=None, label=None, save_path=None, add_bershady=True):
    """Plot concentration vs. log-asymmetry with optional colour coding and Bershady lines."""
    plt.figure(figsize=(7, 6), facecolor='white')

    # Scatter plot
    sc = plt.scatter(conc, log_asym, c=colour if colour is not None else 'grey',
                     cmap='viridis', alpha=0.7, edgecolor='none', label=label)

    if colour is not None:
        plt.colorbar(sc, label='Colour Scale')

    # if add_bershady:
    #     add_bershady_boundaries()

    plt.xlabel("Concentration (C)")
    plt.ylabel("log₁₀(Asymmetry)")
    plt.title(title)
    plt.xlim(0, 5)
    # plt.ylim(-2.2, 0.6)
    # plt.yscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    if label is not None:
        plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    filename = 'combined_morfometryka.fits'
    morf_table = load_morf_table(filename)

    conc, log_asym = compute_concentration_asymmetry(morf_table)
    plt.hist(morf_table['A0'], bins=50)

    make_ca_plot(conc, log_asym,
                 title="C–log(A) Plot from Morfometryka with Bershady Classification",
                 save_path="concentration_asymmetry_plot.png")
