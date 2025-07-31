import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from astropy.io import fits
from astropy.table import Table



def load_hdf5_data(file_path, dataset_name):
    """Load a dataset from an HDF5 file."""
    with h5py.File(file_path, "r") as h5_file:
        return h5_file[dataset_name][:]


def plot_histogram_comparison(data_prior, posterior1, posterior2, bins, output_file, title, xlabel, ylabel):
    """Plot prior vs Bagpipes and EAZY posteriors as normalised histograms."""
    plt.figure(figsize=(8, 6))
    counts_prior, bin_edges = np.histogram(data_prior, bins=bins, density=True)
    counts_post1, _ = np.histogram(posterior1, bins=bin_edges, density=True)
    counts_post2, _ = np.histogram(posterior2, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.step(bin_centers, counts_prior, where='mid', label="Prior (D4000)", color='blue', linewidth=2)
    plt.fill_between(bin_centers, counts_prior, step='mid', alpha=0.4, color='lightblue')

    plt.step(bin_centers, counts_post1, where='mid', label="Bagpipes Posterior", color='darkred', linewidth=2)
    plt.fill_between(bin_centers, counts_post1, step='mid', alpha=0.4, color='salmon')

    plt.step(bin_centers, counts_post2, where='mid', label="EAZY Posterior", color='green', linewidth=2)
    plt.fill_between(bin_centers, counts_post2, step='mid', alpha=0.4, color='lightgreen')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def compute_kl_divergences(prior_samples, posterior_samples, bins=30):
    """Compute KL divergence for each galaxy's posterior vs the global prior."""
    prior_counts, bin_edges = np.histogram(prior_samples, bins=bins, density=True)
    prior_counts += 1e-10  # avoid divide-by-zero

    kl_values = []
    for post in posterior_samples:
        post_counts, _ = np.histogram(post, bins=bin_edges, density=True)
        post_counts += 1e-10
        kl = entropy(post_counts, prior_counts)
        kl_values.append(kl)

    return np.array(kl_values)


def main():
    # File paths
    h5_file_path = "/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/priors/Bagpipes_sfh_cont_bursty_z6.0_Calzetti_log_10_Z_log_10_BPASS_zfix_z6.0.h5"
    dataset_name = "D4000"
    balmer_break_file = "/nvme/scratch/work/Griley/galfind_scripts/Balmerbreak/Balmer_output/balmer_breaks2.txt"
    output_file = "d4000_histogram_with_all_posteriors.png"
    hdulist =  fits.open('/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits')
    table_objects = Table(hdulist[1].data)
    table_bagpipes = Table(hdulist[4].data)

    # Load prior
    d4000_data = load_hdf5_data(h5_file_path, dataset_name)
    F444W = table_objects['FLUX_APER_F444W'] #this is for all 61000 galaxies 
    id = table_bagpipes['#ID'] #this is for the 1000 galaxies in the Bagpipes catalogue


    # Get full and Bagpipes object IDs
    full_ids = table_objects['NUMBER']      # All 61k galaxy IDs
    bagpipes_ids = table_bagpipes['#ID']    # ~1000 galaxies with Bagpipes fits

    # Create mask: select only rows in table_objects with IDs in bagpipes_ids
    mask = np.isin(full_ids, bagpipes_ids)

    # Apply mask to F444W or any other data from table_objects if needed
    F444W_masked = F444W[mask]

    # Also apply the same mask to your Balmer break and redshift arrays
    data = np.loadtxt(balmer_break_file, usecols=(1, 2, 3))  # Columns 2, 3, 4
    balmer_break_values_pipes = data[:, 0]
    balmer_break_values_EZ = data[:, 1]
    #redshifts = data[:, 2]
    
    # Plot comparison histograms
    plot_histogram_comparison(
        data_prior=d4000_data,
        posterior1=balmer_break_values_pipes,
        posterior2=balmer_break_values_EZ,
        bins=30,
        output_file=output_file,
        title="Prior vs Posterior Balmer Breaks (Bagpipes & EAZY)",
        xlabel="Balmer Break Magnitude (D4000)",
        ylabel="Probability Density"
    )

    # Simulate Bagpipes posteriors using Gaussian kernels
    posterior_samples = [np.random.normal(loc=val, scale=0.02, size=1000) for val in balmer_break_values_pipes]

    # Compute KL divergence (posterior vs prior)
    kl_values = compute_kl_divergences(d4000_data, posterior_samples, bins=30)

    # Plot KL divergence vs Log F444W
    plt.figure(figsize=(8, 6))
    plt.scatter(F444W_masked, kl_values, c=balmer_break_values_pipes, cmap='plasma', s=30, alpha=0.8) #change x to 444 flux or log (ie mag)
    plt.xlabel("log F444W ")
    plt.xscale('log')
    plt.ylabel("KL Divergence (Bagpipes D4000)")
    plt.title("Information Gained on D4000 vs Log F444W")
    plt.colorbar(label="Bagpipes D4000")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("kl_vs_logF444W.png")
    plt.show()

    # Plot KL divergence vs D4000
    plt.figure(figsize=(8, 6))
    plt.scatter(balmer_break_values_pipes, kl_values, c=F444W_masked, cmap='plasma', s=30, alpha=0.8) #change x to 444 flux or log (ie mag)
    plt.xlabel("D4000 ")
    plt.ylabel("KL Divergence (Bagpipes D4000)")
    plt.title("Information Gained on D4000 vs D4000")
    plt.colorbar(label="LogF444W")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("kl_vs_D4000.png")
    plt.show()

if __name__ == "__main__":
    main()
