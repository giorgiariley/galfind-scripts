import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

def load_balmer_breaks(filename):
    data = np.loadtxt(filename, skiprows=1)
    indices = data[:, 0].astype(int)
    balmer_breaks = data[:, 1]
    return indices, balmer_breaks

def load_bagpipes_table(fits_file):
    return Table.read(fits_file)

def extract_parameters(table_bagpipes, indices, balmer_breaks):
    beta = table_bagpipes['beta_C94_50'][indices]
    xi_ion = table_bagpipes['xi_ion_caseB_rest_50'][indices]

    # Filter unphysical beta
    mask = beta >= -5
    beta = beta[mask]
    xi_ion = xi_ion[mask]
    balmer_breaks = balmer_breaks[mask]

    # Clip beta values: beta <= -2.8 -> -2.8
    beta = np.clip(beta, a_min=-2.8, a_max=None)

    # Clean xi_ion for log (replace <= 0 with nan to avoid issues)
    xi_ion = np.where(xi_ion <= 0, np.nan, xi_ion)

    return beta, xi_ion, balmer_breaks

def scatter_coloured_by(x, y, c, xlabel, ylabel, c_label, filename,
                       add_beta_line=False, add_beta_pct_legend=False):
    plt.figure(figsize=(7,6))
    sc = plt.scatter(x, y, c=c, cmap='viridis', s=30, edgecolor='k', alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar(sc)
    cbar.set_label(c_label)

    if add_beta_line:
        vline = -2.8
        plt.axvline(vline, color='grey', linestyle=':', linewidth=2)
        pct_below = np.sum(x <= vline) / len(x) * 100
        legend_text = f'β ≤ {vline} ({pct_below:.1f}%)'
        plt.legend([legend_text], loc='best')

    if add_beta_pct_legend:
        pct_below_eq = np.sum(x <= -2.8) / len(x) * 100
        legend_text = f'β ≤ -2.8: {pct_below_eq:.1f}%'
        plt.legend([legend_text], loc='best')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    balmer_file = "/nvme/scratch/work/Griley/galfind_scripts/Balmerbreak/Balmer_output/balmer_breaks2.txt"
    fits_file = "/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits"

    indices, balmer_breaks = load_balmer_breaks(balmer_file)
    table_bagpipes = load_bagpipes_table(fits_file)
    beta, xi_ion, balmer_breaks = extract_parameters(table_bagpipes, indices, balmer_breaks)

    # Log transform xi_ion (ignoring NaNs)
    xi_ion_log = np.log10(xi_ion)

    # 1. beta vs log10(xi_ion) coloured by Balmer break (beta on x-axis: add line)
    scatter_coloured_by(beta, xi_ion_log, balmer_breaks,
                       'UV beta slope value (clipped at -2.8)', 'log10(xi ion)', 'Balmer break',
                       'beta_vs_logxi_coloured_by_balmer.png',
                       add_beta_line=True)

    # 2. beta vs Balmer break coloured by log10(xi_ion) (beta on x-axis: add line)
    scatter_coloured_by(beta, balmer_breaks, xi_ion_log,
                       'UV beta slope value (clipped at -2.8)', 'Balmer break', 'log10(xi ion)',
                       'beta_vs_balmer_coloured_by_logxi.png',
                       add_beta_line=True)

    # 3. log10(xi_ion) vs Balmer break coloured by beta (no line, but add percentage in legend)
    scatter_coloured_by(xi_ion_log, balmer_breaks, beta,
                       'log10(xi ion)', 'Balmer break', 'UV beta slope value (clipped at -2.8)',
                       'logxi_vs_balmer_coloured_by_beta.png',
                       add_beta_line=False,
                       add_beta_pct_legend=True)

    print("Plots saved!")

if __name__ == "__main__":
    main()
