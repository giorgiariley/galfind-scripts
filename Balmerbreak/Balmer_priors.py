import h5py
import matplotlib.pyplot as plt
import numpy as np

def load_hdf5_data(file_path, dataset_name):
    """
    Load a dataset from an HDF5 file.
    
    Parameters:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset to extract.
    
    Returns:
        np.ndarray: The extracted dataset.
    """
    with h5py.File(file_path, "r") as h5_file:
        return h5_file[dataset_name][:]

def extract_balmer_break_values(file_path):
    """
    Extract Balmer break values (column 2) from a text file.
    
    Parameters:
        file_path (str): Path to the text file.
    
    Returns:
        np.ndarray: Array of Balmer break values.
    """
    # Load the file and extract the second column
    data = np.loadtxt(file_path, usecols=1)
    return data

def plot_histogram_comparison(data_prior, posterior1, posterior2, bins, output_file, title, xlabel, ylabel):
    """
    Plot three overlaid histograms (prior and two posteriors),
    all normalised to have area = 1 (i.e., probability densities).
    """
    plt.figure(figsize=(8, 6))

    # Use same bin edges for all
    counts_prior, bin_edges = np.histogram(data_prior, bins=bins, density=True)
    counts_post1, _ = np.histogram(posterior1, bins=bin_edges, density=True)
    counts_post2, _ = np.histogram(posterior2, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot Prior
    plt.step(bin_centers, counts_prior, where='mid', label="Prior (D4000)", color='blue', linewidth=2)
    plt.fill_between(bin_centers, counts_prior, step='mid', alpha=0.4, color='lightblue')

    # Plot Bagpipes Posterior
    plt.step(bin_centers, counts_post1, where='mid', label="Selected Bagpipes Posterior", color='darkred', linewidth=2)
    plt.fill_between(bin_centers, counts_post1, step='mid', alpha=0.4, color='salmon')

    # Plot EAZY Posterior
    plt.step(bin_centers, counts_post2, where='mid', label="Selected EAZY Posterior", color='green', linewidth=2)
    plt.fill_between(bin_centers, counts_post2, step='mid', alpha=0.4, color='lightgreen')

    # Formatting
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()



    
# Main function
def main():
    # File paths
    h5_file_path = "/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/priors/Bagpipes_sfh_cont_bursty_z6.0_Calzetti_log_10_Z_log_10_BPASS_zfix_z6.0.h5"
    dataset_name = "D4000"
    balmer_break_file_pipes = "/nvme/scratch/work/Griley/galfind_scripts/Balmerbreak/overplot_output/balmer_breaks2.txt"
    balmer_break_file_EZ = "/nvme/scratch/work/Griley/galfind_scripts/Balmerbreak/overplot_output/balmer_breaksEZ.txt"
    output_file = "d4000_histogram_with_all_posteriors.png"

    # Load data
    d4000_data = load_hdf5_data(h5_file_path, dataset_name)
    balmer_break_values_pipes = extract_balmer_break_values(balmer_break_file_pipes)
    balmer_break_values_EZ = extract_balmer_break_values(balmer_break_file_EZ)

    # Single histogram plot
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


# Run the script
if __name__ == "__main__":
    main()