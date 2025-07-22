import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

def load_bagpipes_table(fits_path):
    """Load the Bagpipes output table from a FITS file."""
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[1].data)
    return table

def plot_burstiness_histogram(burstiness_values, output_path="burstiness_histogram.png"):
    """Plot a histogram of burstiness values."""
    plt.figure(figsize=(8, 6), facecolor='white')
    plt.hist(burstiness_values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Burstiness")
    plt.ylabel("Number of galaxies")
    plt.title("Histogram of Burstiness")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved histogram to: {output_path}")

def print_low_burstiness(burstiness_values, threshold=0.5):
    """Print and count galaxies with burstiness below threshold."""
    low_burst = burstiness_values[burstiness_values < threshold]
    print(f"Number of galaxies with burstiness < {threshold}: {len(low_burst)}")
    print(low_burst)

def main():
    fits_path = "/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits"
    table_pipes = load_bagpipes_table(fits_path)
    burstiness = table_pipes['burstiness_50']
    
    plot_burstiness_histogram(burstiness)
    print_low_burstiness(burstiness, threshold=0.5)

if __name__ == "__main__":
    main()
