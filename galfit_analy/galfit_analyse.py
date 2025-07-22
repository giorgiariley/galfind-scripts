#this code is where i will analyse things from the galfit output
#such as a raidus against n plot
from astropy.io import fits
from astropy.table import Table
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import matplotlib.pyplot as plt

# Load the sersic.fits file
hdulist = fits.open("/raid/scratch/work/Griley/GALFIND_WORK/GALFIT/output/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/F444W/sersic/results.fits")
table_galfit = Table(hdulist[1].data)
hdulist.close()

# Load the bagpipes file
hdulist = fits.open("/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits")
table_bagpipes = Table(hdulist[1].data)
hdulist.close()

#also need to extract redshifts, asiest wya is to extract from the balmer break file i already made
balmer_file = "/nvme/scratch/work/Griley/galfind_scripts/Balmerbreak/balmer_break_outputs1/balmer_breaks.txt"
data = np.loadtxt(balmer_file, skiprows=1)  # skip header
indices = data[:, 0].astype(int)            # galaxy indices
redshifts = data[:, 2]                      # if needed

# Extract GALFIT outputs
ID = table_galfit['id']  # Galaxy IDs
sersic_index = table_galfit['n']
radius_pixels  = table_galfit['r_e']  # Effective radius in pixels
radius_arcsec = radius_pixels * 0.03  # Convert to arcseconds (assuming pixel scale of 0.03 arcsec/pixel)
chi2_red = table_galfit['red_chi2']  # replace with actual column name if different


# Extract Bagpipes outputs
burstiness = table_bagpipes['burstiness_50']  # Burstiness parameter
stellar_mass = table_bagpipes['stellar_mass_50']  # Stellar mass (log10)
UV = table_bagpipes['UV_colour_50']  # UV luminosity (log10)
VJ = table_bagpipes['VJ_colour_50']  # V-J colour (log10)

# === cosmological conversion ===
# Get angular diameter distances (in kpc) for each galaxy
DA_kpc_per_arcsec = np.array([
    (cosmo.angular_diameter_distance(z).to(u.kpc) / u.radian.to(u.arcsec)).value
    for z in redshifts
])

radius_kpc = radius_arcsec * DA_kpc_per_arcsec  # Final radius in kpc
# === Flagging unreliable fits ===
#ensuring redshifts are all physical
flag_bad_z = (redshifts <= 0.01) | (redshifts >= 16)

# Radius limits: reasonable sizes
flag_bad_radius = (radius_kpc <= 0.01) | (radius_kpc > 10)

# Sersic index limits
flag_bad_n = (sersic_index >= 10)

# High reduced chi-squared: bad fits
flag_bad_chi2 = (chi2_red > 5)  # Adjust threshold if needed

# Combine flags
flag_unreliable = flag_bad_radius | flag_bad_n | flag_bad_chi2| flag_bad_z

# Save flags to table if you want to keep track
table_galfit['flag_bad_radius'] = flag_bad_radius
table_galfit['flag_bad_n'] = flag_bad_n
table_galfit['flag_bad_chi2'] = flag_bad_chi2
table_galfit['flag_bad_z'] = flag_bad_z
table_galfit['flag_unreliable'] = flag_unreliable

# === Apply clean mask ===
reliable_mask = ~flag_unreliable
# Apply to all relevant arrays
radius_kpc_clean = radius_kpc[reliable_mask]
sersic_index_clean = sersic_index[reliable_mask]
burstiness_clean = burstiness[reliable_mask]
stellar_mass_clean = stellar_mass[reliable_mask]
UV_clean = UV[reliable_mask]

# Create a scatter plot of radius vs Sersic index (only physical radii)
plt.figure(figsize=(10, 6), facecolor='white')
sc = plt.scatter(stellar_mass_clean, radius_kpc_clean, c=burstiness_clean, cmap='magma', alpha=0.7)
plt.yscale('log')  # Log scale for radius
plt.ylabel('Effective Radius (kpc)')
# plt.xlim(0.001, 10)
plt.xlabel('Stellar Mass (log10 M☉)')
plt.title('Stellar Mass vs Effective Radius (rₑ ≤ 10 kpc), colour-coded by burstiness')
plt.colorbar(sc, label='burstiness')
# Add completeness limit
plt.axvline(8.1, color='blue', linestyle='--', label='90% completeness')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mass_vs_radius_coloured_burstiness_with_completeness.png")
plt.show()


#lets use the fixed n run to plot effective radius against other properties 
plt.figure(figsize=(10, 6), facecolor='white')
sc = plt.scatter(radius_kpc_clean, burstiness_clean, alpha=0.7)
plt.xscale('log')  # Log scale for radius
plt.xlabel('Effective Radius (kpc)')
plt.ylabel('Burstiness')
plt.title('Burtsiness vs Effective Radius for fixed n = 1')
plt.grid(True)
plt.tight_layout()
plt.savefig("b_radius_n=1.png")
plt.show()

# Create a scatter plot of colour vs mass/radius
plt.figure(figsize=(10, 6), facecolor='white')
sc = plt.scatter(radius_kpc_clean, UV_clean, c=burstiness_clean, cmap='magma', alpha=0.7)
plt.xscale('log')  # Log scale for radius
plt.ylabel('U-V colour')
# plt.xlim(0.001, 10)
plt.xlabel('Radius (kpc)')
plt.title('Effective radius vs UV colour (rₑ ≤ 10 kpc) for fixed n')
plt.colorbar(sc, label='burstiness')
# Add completeness limit
# plt.axvline(8.1, color='blue', linestyle='--', label='90% completeness')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("radius_vs_UV_b__complete_fixedn.png")
plt.show()

def bin_median(x, y, bins):
    """
    Bin x and compute the median and std of y in each bin.

    Parameters:
        x (array): x-values to bin on
        y (array): y-values to compute stats for
        bins (array): bin edges

    Returns:
        bin_centres, median_y, std_y, counts
    """
    inds = np.digitize(x, bins) # Assign each x to a bin
    bin_centres = 0.5 * (bins[:-1] + bins[1:]) # Calculate bin centres  
    medians, stds, counts = [], [], [] 

    # Loop through each bin and calculate median, std, and count
    for i in range(1, len(bins)):
        mask = inds == i  # Get mask for current bin
        if np.any(mask): # Check if there are any values in the bin
            medians.append(np.median(y[mask]))
            stds.append(np.std(y[mask]))
            counts.append(np.sum(mask))
        else:
            medians.append(np.nan)
            stds.append(np.nan)
            counts.append(0)

    return bin_centres, np.array(medians), np.array(stds), np.array(counts)


def plot_binned_mass_radius(stellar_mass, radius_kpc, burstiness, bins, savepath=None):
    """
    Plot the mass–radius relation split into bursty and smouldering galaxies.

    Parameters:
        stellar_mass (array): log10 stellar mass
        radius_kpc (array): effective radius in kpc
        burstiness (array): burstiness parameter
        bins (array): bin edges for stellar mass
        savepath (str): optional path to save the figure
    """
    # Define bursty and smouldering masks
    mask_bursty = burstiness > 2
    mask_smoulder = burstiness < 2

    # Bin the data
    mass_centres, radius_bursty, std_bursty, _ = bin_median(
        stellar_mass[mask_bursty], radius_kpc[mask_bursty], bins
    )
    _, radius_smoulder, std_smoulder, _ = bin_median(
        stellar_mass[mask_smoulder], radius_kpc[mask_smoulder], bins
    )

    # Plot
    plt.figure(figsize=(10, 6), facecolor='white')
    plt.errorbar(mass_centres, radius_bursty, yerr=std_bursty, fmt='-o', color='crimson', label='Bursty (burstiness > 2)')
    plt.errorbar(mass_centres, radius_smoulder, yerr=std_smoulder, fmt='-s', color='navy', label='Smouldering (burstiness < 2)')
    plt.axvline(8.1, linestyle='--', color='blue', label='90% completeness')
    plt.yscale('log')
    plt.xlabel('Stellar Mass (log₁₀ M☉)')
    plt.ylabel('Median Radius (kpc)')
    plt.title('Mass–Radius Relation (Binned by 0.5 dex)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def binned_trend_vs_redshift(z, y, y_label, mass=None, mass_cut=None, bin_width=0.5, title='', save_name='plot.png'):
    """
    Plot binned median of y vs redshift, optionally applying a stellar mass cut.
    """
    if mass_cut is not None and mass is not None:
        mask = mass > mass_cut
        z = z[mask]
        y = y[mask]
    
    # Define redshift bins
    z_bins = np.arange(np.floor(min(z)), np.ceil(max(z)) + bin_width, bin_width)
    bin_centres = (z_bins[:-1] + z_bins[1:]) / 2

    medians, stds = [], []

    inds = np.digitize(z, z_bins)
    for i in range(1, len(z_bins)):
        bin_mask = inds == i
        if np.any(bin_mask):
            medians.append(np.median(y[bin_mask]))
            stds.append(np.std(y[bin_mask]))
        else:
            medians.append(np.nan)
            stds.append(np.nan)

# Convert to arrays for easier masking
    bin_centres = np.array(bin_centres)
    medians = np.array(medians)
    stds = np.array(stds)

    # Mask out nan values to avoid broken lines
    valid_mask = ~np.isnan(medians)

    # Plotting
    plt.figure(figsize=(8, 6), facecolor='white')
    plt.scatter(
    z, y, alpha=0.2, s=10, label='Individual galaxies', color='grey', zorder=0
)
    plt.errorbar(
        bin_centres[valid_mask],
        medians[valid_mask],
        yerr=stds[valid_mask],
        fmt='o-', capsize=4,
        label=f"{'With' if mass_cut else 'Without'} mass cut"
    )
    plt.xlabel('Redshift')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_name)
    plt.show()


mass_bins = np.arange(7, 11.5, 0.5)

plot_binned_mass_radius(
    stellar_mass=stellar_mass_clean,
    radius_kpc=radius_kpc_clean,
    burstiness=burstiness_clean,
    bins=mass_bins,
    savepath="mass_radius_binned_bursty_vs_smouldering.png"
)

binned_trend_vs_redshift(
    z=redshifts[reliable_mask],
    y=sersic_index_clean,
    y_label='Sérsic Index',
    mass=stellar_mass_clean,
    mass_cut=8.1,
    bin_width=0.5,
    title='Sérsic Index vs Redshift (Δz = 0.5)',
    save_name='sersic_vs_z_masscut.png'
)

binned_trend_vs_redshift(
    z=redshifts[reliable_mask],
    y=sersic_index_clean,
    y_label='Sérsic Index',
    mass=None,
    mass_cut=None,
    bin_width=0.5,
    title='Sérsic Index vs Redshift (Δz = 0.5, no mass cut)',
    save_name='sersic_vs_z_nomasscut.png'
)

binned_trend_vs_redshift(
    z=redshifts[reliable_mask],
    y=radius_kpc_clean,
    y_label='Effective Radius (kpc)',
    mass=stellar_mass_clean,
    mass_cut=8.1,
    bin_width=0.5,
    title='Radius vs Redshift (Δz = 0.5)',
    save_name='radius_vs_z_masscut.png'
)

binned_trend_vs_redshift(
    z=redshifts[reliable_mask],
    y=radius_kpc_clean,
    y_label='Effective Radius (kpc)',
    mass=None,
    mass_cut=None,
    bin_width=0.5,
    title='Radius vs Redshift (Δz = 0.5, no mass cut)',
    save_name='radius_vs_z_nomasscut.png'
)

