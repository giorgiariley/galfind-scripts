import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


# Load Bagpipes table
fits_path = "/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits"

def load_bagpipes_table(fits_path):
    """Load the Bagpipes output table from a FITS file."""
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[1].data)
    return table

# Load the sersic.fits file
hdulist = fits.open("/raid/scratch/work/Griley/GALFIND_WORK/GALFIT/output/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/F444W/sersic/results.fits")
table_galfit = Table(hdulist[1].data)
hdulist.close()

# --- Scaling code ---
def compute_flux_ratio(filter, table):
    """Computes the auto/aper flux ratio and clips extreme values."""
    ratio = table[f'FLUX_AUTO_{filter}'] / table[f'FLUX_APER_{filter}']
    return np.clip(ratio, 1, 10)  # prevent under/over-correction

def scale_mass_sfr_log(table, filter):
    """Scales log10 mass and SFR columns by log10(flux_auto / flux_aperture)."""
    R = compute_flux_ratio(filter, table)
    logR = np.log10(R)
    table['Stellar_Mass_50'] += logR
    table['sfr_50'] += logR
    return table

# Load both Bagpipes tables (HDU 1: photometry; HDU 4: physical params)
with fits.open(fits_path) as hdulist:
    table_objects = Table(hdulist[1].data)     # photometry (FLUX_...)
    table_bagpipes = Table(hdulist[4].data)    # stellar mass, sfr, burstiness, etc.

# Match rows by ID
ids_objects = table_objects['NUMBER'].astype(str)
ids_bagpipes = table_bagpipes['#ID'].astype(str)
common_ids, idx_obj, idx_bag = np.intersect1d(ids_objects, ids_bagpipes, return_indices=True)

# Subset both tables to matched rows
table_objects_matched = table_objects[idx_obj]
table_bagpipes = table_bagpipes[idx_bag]

# Apply flux-based scaling to matched Bagpipes table
filter_name = 'F444W'
R = compute_flux_ratio(filter_name, table_objects_matched)
logR = np.log10(R)
table_bagpipes['stellar_mass_50'] += logR
table_bagpipes['sfr_50'] += logR


# Extract GALFIT outputs
redshifts = table_bagpipes['input_redshift']  # Redshifts from bagpipes
ID = table_galfit['id']  # Galaxy IDs
sersic_index = table_galfit['n']
radius_pixels  = table_galfit['r_e']  # Effective radius in pixels
radius_arcsec = radius_pixels * 0.03  # Convert to arcseconds (assuming pixel scale of 0.03 arcsec/pixel)
chi2_red = table_galfit['red_chi2']  # replace with actual column name if different
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

# --- Apply reliability mask ---
stellar_mass_scaled = table_bagpipes['stellar_mass_50'][reliable_mask]
burstiness_clean = table_bagpipes['burstiness_50'][reliable_mask]
halpha_clean = table_bagpipes['Halpha_EW_rest_50'][reliable_mask]

# --- Identify extreme PSBs ---
is_extreme_psb = (burstiness_clean <= 1) & (halpha_clean <= 200)

# --- Plot: Mass vs Radius, highlighting extreme PSBs ---
plt.figure(figsize=(10, 6), facecolor='white')

# Plot other galaxies first (in tomato)
plt.scatter(
    stellar_mass_scaled[~is_extreme_psb], radius_kpc_clean[~is_extreme_psb],
    color='tomato', alpha=0.6, edgecolor='none', label='Others'
)

# Plot extreme PSBs on top (in royal blue)
plt.scatter(
    stellar_mass_scaled[is_extreme_psb], radius_kpc_clean[is_extreme_psb],
    color='royalblue', alpha=0.9, edgecolor='black', linewidth=0.2, label='Extreme PSBs'
)

plt.yscale('log')
plt.xlabel('Stellar Mass (log₁₀ M☉, scaled)')
plt.ylabel('Effective Radius (kpc)')
plt.title('Stellar Mass vs Effective Radius\n(Extreme PSBs: burstiness ≤ 1 & Hα EW ≤ 200 Å)')
plt.axvline(8.1, color='gray', linestyle='--', label='90% completeness')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("mass_vs_radius_extreme_psbs.png")
plt.show()
