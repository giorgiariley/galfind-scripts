import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

def load_fits_table(fits_path, hdu_index=1):
    """ FITS table loader"""
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[hdu_index].data)
    return table

def match_tables_by_id(table1, table2, col1, col2):
    """Returns matched versions of table1 and table2"""
    ids1 = table1[col1].astype(str)
    ids2 = table2[col2].astype(str)
    _, idx1, idx2 = np.intersect1d(ids1, ids2, return_indices=True)
    return table1[idx1], table2[idx2]

def compute_flux_ratio(table, filter_name):
    """Computes the auto/aper flux ratio and clips extreme values."""
    ratio = table[f'FLUX_AUTO_{filter_name}'] / table[f'FLUX_APER_{filter_name}']
    return np.clip(ratio, 1, 10)

def scale_mass_sfr_log(table, logR):
    """Scales log10 mass column by flux ratio."""
    table['stellar_mass_50'] += logR
    return table

def get_radius_kpc(table_galfit, redshifts, pixel_scale=0.03):
    """Calculates effective radius in kpc given GALFIT table and redshifts."""
    radius_pixels = table_galfit['r_e']
    radius_arcsec = radius_pixels * pixel_scale
    DA_kpc_per_arcsec = np.array([
        (cosmo.angular_diameter_distance(z).to(u.kpc) / u.radian.to(u.arcsec)).value
        for z in redshifts
    ])
    return radius_arcsec * DA_kpc_per_arcsec

def flag_unreliable_fits(table_galfit, radius_kpc, redshifts):
    """Returns a boolean mask of reliable fits."""
    sersic_index = table_galfit['n']
    chi2_red = table_galfit['red_chi2']
    flag_bad_z = (redshifts <= 0.01) | (redshifts >= 16)
    flag_bad_radius = (radius_kpc <= 0.01) | (radius_kpc > 10)
    flag_bad_n = (sersic_index >= 10)
    flag_bad_chi2 = (chi2_red > 5)
    return ~(flag_bad_radius | flag_bad_n | flag_bad_chi2 | flag_bad_z)

def plot_mass_vs_radius(stellar_mass, radius_kpc, is_extreme_psb, savefig=None):
    """Scatter plot of mass vs. effective radius, highlighting extreme PSBs."""
    plt.figure(figsize=(10, 6), facecolor='white')
    # Plot others
    plt.scatter(
        stellar_mass[~is_extreme_psb], radius_kpc[~is_extreme_psb],
        color='tomato', alpha=0.6, edgecolor='none', label='burstiness > 1 and Hα EW > 200 Å'
    )
    # Plot extreme PSBs
    plt.scatter(
        stellar_mass[is_extreme_psb], radius_kpc[is_extreme_psb],
        color='royalblue', alpha=0.9, edgecolor='black', linewidth=0.2, label='Extreme PSBs (burstiness ≤ 1 & Hα EW ≤ 200 Å)'
    )
    plt.yscale('log')
    plt.xlabel('Stellar Mass (log$_{10}$ M$_\odot$, scaled)')
    plt.ylabel('Effective Radius (kpc)')
    plt.title('Stellar Mass vs Effective Radius')
    plt.axvline(8.1, color='gray', linestyle='--', label='90% completeness')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)

def main(
    phot_fits, bagpipes_fits, galfit_fits, 
    filter_name='F444W', phot_idcol='NUMBER', bagpipes_idcol='#ID'
):
    # Load tables
    table_objects = load_fits_table(phot_fits, hdu_index=1)
    with fits.open(bagpipes_fits) as hdulist:
        table_bagpipes = Table(hdulist[4].data) # assumes hdu=4 for Bagpipes outputs
    table_galfit = load_fits_table(galfit_fits, hdu_index=1)

    # Match photometry and Bagpipes tables by ID
    table_objects_matched, table_bagpipes = match_tables_by_id(
        table_objects, table_bagpipes, phot_idcol, bagpipes_idcol
    )

    # Scale mass and SFR
    R = compute_flux_ratio(table_objects_matched, filter_name)
    logR = np.log10(R)
    scale_mass_sfr_log(table_bagpipes, logR)

    # Extract redshifts and compute radius in kpc
    redshifts = table_bagpipes['input_redshift']
    radius_kpc = get_radius_kpc(table_galfit, redshifts)

    # Mask for reliable fits
    reliable_mask = flag_unreliable_fits(table_galfit, radius_kpc, redshifts)
    radius_kpc_clean = radius_kpc[reliable_mask]

    # Apply mask to Bagpipes parameters (assumes ordering matches GALFIT)
    stellar_mass_scaled = table_bagpipes['stellar_mass_50'][reliable_mask]
    burstiness_clean = table_bagpipes['burstiness_50'][reliable_mask]
    halpha_clean = table_bagpipes['Halpha_EW_rest_50'][reliable_mask]

    # Extreme PSB mask
    is_extreme_psb = (burstiness_clean <= 1) & (halpha_clean <= 200)

    # Plot
    plot_mass_vs_radius(stellar_mass_scaled, radius_kpc_clean, is_extreme_psb, savefig="mass_vs_radius_extreme_psbs.png")

if __name__ == "__main__":
    main(
        phot_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        bagpipes_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        galfit_fits="/raid/scratch/work/Griley/GALFIND_WORK/GALFIT/output/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/F444W/sersic/results.fits",
        filter_name='F444W',
        phot_idcol='NUMBER',
        bagpipes_idcol='#ID'
    )
