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

def match_three_tables_by_id(table1, table2, table3, col1, col2, col3):
    """
    Returns matched versions of all three tables, containing only the rows where the IDs match.
    """
    ids1 = table1[col1].astype(str)
    ids2 = table2[col2].astype(str)
    ids3 = table3[col3].astype(str)
    # Find intersection of all three ID lists
    common_ids = np.intersect1d(np.intersect1d(ids1, ids2), ids3)
    # Find indices for each table
    idx1 = np.nonzero(np.in1d(ids1, common_ids))[0]
    idx2 = np.nonzero(np.in1d(ids2, common_ids))[0]
    idx3 = np.nonzero(np.in1d(ids3, common_ids))[0]
    return table1[idx1], table2[idx2], table3[idx3]


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
    # === Load the external comparison data ===
    external_fits = "/raid/scratch/work/Griley/GALFIND_WORK/EPOCHS_XI_structural_parameters.fits"
    external_table = load_fits_table(external_fits, hdu_index=1)  # Change hdu_index if not 1

    # Extract mass and radius (log10 mass, radius in kpc)
    external_mass = external_table['stellar_mass_50']
    external_radius_kpc = external_table['re_kpc']

    # Overplot on existing axes
    plt.scatter(
    external_mass, external_radius_kpc,
    marker='o', facecolors='none', edgecolors='slategrey', s=30, label='Westcott:EPOCHS-XI'
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
    plt.close()

def plot_mass_vs_radius_by_zbin(
    stellar_mass, radius_kpc, is_extreme_psb, redshifts,
    bins = [3, 5, 8, 16], save_prefix="mass_vs_radius_zbin"
):
    """
    Plot mass–radius in redshift bins.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    bin_indices = np.digitize(redshifts, bins)
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) < 1:
            continue  # Skip empty bins

        plt.figure(figsize=(9, 6), facecolor='white')
        plt.scatter(
            stellar_mass[mask & ~is_extreme_psb], radius_kpc[mask & ~is_extreme_psb],
            color='tomato', alpha=0.6, label='burstiness > 1 and Hα EW > 200 Å'
        )
        plt.scatter(
            stellar_mass[mask & is_extreme_psb], radius_kpc[mask & is_extreme_psb],
            color='royalblue', alpha=0.9, edgecolor='black', linewidth=0.2,
            label='Extreme PSBs (burstiness ≤ 1 & Hα EW ≤ 200 Å)'
        )
        
        plt.yscale('log')
        plt.xlabel('Stellar Mass (log$_{10}$ M$_\odot$, scaled)')
        plt.ylabel('Effective Radius (kpc)')
        plt.title(f'Mass vs Radius: {bins[i-1]} < z ≤ {bins[i]}')
        plt.axvline(8.1, color='gray', linestyle='--', label='90% completeness')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_z{bins[i-1]}_{bins[i]}.png")
        print(f"Saving plot for {bins[i-1]} < z <= {bins[i]}, n={np.sum(mask)}")
        plt.close()

def plot_radius_vs_redshift(redshifts, radius_kpc, is_extreme_psb, savefig=None, mass_cut_applied=False):
    """Scatter plot of redshift vs. effective radius, highlighting extreme PSBs."""
    plt.figure(figsize=(10, 6), facecolor='white')
    # Plot others
    plt.scatter(
        redshifts[~is_extreme_psb], radius_kpc[~is_extreme_psb],
        color='tomato', alpha=0.6, edgecolor='none', label='burstiness > 1 and Hα EW > 200 Å'
    )
    # Plot extreme PSBs
    plt.scatter(
        redshifts[is_extreme_psb], radius_kpc[is_extreme_psb],
        color='royalblue', alpha=0.9, edgecolor='black', linewidth=0.2, label='Extreme PSBs (burstiness ≤ 1 & Hα EW ≤ 200 Å)'
    )
    # External comparison data, as before...
    external_fits = "/raid/scratch/work/Griley/GALFIND_WORK/EPOCHS_XI_structural_parameters.fits"
    external_table = load_fits_table(external_fits, hdu_index=1)
    external_mass = external_table['stellar_mass_50']
    external_radius_kpc = external_table['re_kpc']
    external_redshift = external_table['zbest_fsps_larson']
    plt.scatter(
        external_redshift, external_radius_kpc,
        marker='o', facecolors='none', edgecolors='slategrey', s=30, label='Westcott:EPOCHS-XI'
    )

    plt.yscale('log')
    plt.xlabel('Redshift')
    plt.ylabel('Effective Radius (kpc)')
    # Dynamic title
    if mass_cut_applied:
        plt.title('Redshift vs Effective Radius (log M$_* > 8.1$ cut applied)')
    else:
        plt.title('Redshift vs Effective Radius (no mass cut)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.close()

def plot_binned_mass_vs_radius(stellar_mass, radius_kpc, is_extreme_psb, nbins=16, savefig=None):
    """
    Plots median effective radius vs. stellar mass in mass bins,
    for non-extreme PSBs, extreme PSBs, and Westcott comparison sample.
    """
    plt.figure(figsize=(9,6))
    bins = np.linspace(np.nanmin(stellar_mass), np.nanmax(stellar_mass), nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # -- Other galaxies (not extreme PSBs)
    medians_other, lower_other, upper_other = [], [], []
    for i in range(nbins):
        mask = (stellar_mass >= bins[i]) & (stellar_mass < bins[i+1]) & (~is_extreme_psb)
        vals = radius_kpc[mask]
        if np.sum(mask) > 0:
            medians_other.append(np.nanmedian(vals))
            lower_other.append(np.nanpercentile(vals, 16))
            upper_other.append(np.nanpercentile(vals, 84))
        else:
            medians_other.append(np.nan)
            lower_other.append(np.nan)
            upper_other.append(np.nan)
    plt.errorbar(bin_centers, medians_other, yerr=[np.array(medians_other)-np.array(lower_other), np.array(upper_other)-np.array(medians_other)],
                 fmt='o-', color='tomato', alpha=0.7, label='Other galaxies')

    # -- Extreme PSBs
    medians_psb, lower_psb, upper_psb = [], [], []
    for i in range(nbins):
        mask = (stellar_mass >= bins[i]) & (stellar_mass < bins[i+1]) & is_extreme_psb
        vals = radius_kpc[mask]
        if np.sum(mask) > 0:
            medians_psb.append(np.nanmedian(vals))
            lower_psb.append(np.nanpercentile(vals, 16))
            upper_psb.append(np.nanpercentile(vals, 84))
        else:
            medians_psb.append(np.nan)
            lower_psb.append(np.nan)
            upper_psb.append(np.nan)
    plt.errorbar(bin_centers, medians_psb, yerr=[np.array(medians_psb)-np.array(lower_psb), np.array(upper_psb)-np.array(medians_psb)],
                 fmt='o-', color='royalblue', alpha=0.7, label='Extreme PSBs')

    # --- Binned Westcott Data ---
    # Load Westcott data
    external_fits = "/raid/scratch/work/Griley/GALFIND_WORK/EPOCHS_XI_structural_parameters.fits"
    external_table = load_fits_table(external_fits, hdu_index=1)
    external_mass = external_table['stellar_mass_50']
    external_radius_kpc = external_table['re_kpc']
    # Use same bins for fair comparison
    medians_west, lower_west, upper_west = [], [], []
    for i in range(nbins):
        mask = (external_mass >= bins[i]) & (external_mass < bins[i+1])
        vals = external_radius_kpc[mask]
        if np.sum(mask) >= 5:
            medians_west.append(np.nanmedian(vals))
            lower_west.append(np.nanpercentile(vals, 16))
            upper_west.append(np.nanpercentile(vals, 84))
        else:
            medians_west.append(np.nan)
            lower_west.append(np.nan)
            upper_west.append(np.nan)
    plt.errorbar(bin_centers, medians_west, yerr=[np.array(medians_west)-np.array(lower_west), np.array(upper_west)-np.array(medians_west)],
                 fmt='o-', color='slategrey', alpha=1, markerfacecolor='none', label='Westcott:EPOCHS-XI')

    plt.yscale('log')
    plt.xlabel('Stellar Mass (log$_{10}$ M$_\odot$, scaled)')
    plt.ylabel('Median Effective Radius (kpc)')
    plt.title('Binned: Stellar Mass vs Effective Radius')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axvline(8.1, color='gray', linestyle='--', label='90% completeness')
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()
    plt.close()


def plot_binned_redshift_vs_radius(redshifts, radius_kpc, is_extreme_psb, nbins=12, savefig=None, mass_cut_applied=False):
    """
    Plots median effective radius vs. redshift in bins, for 'other galaxies', extreme PSBs, and Westcott.
    """
    plt.figure(figsize=(9,6))
    bins = np.linspace(np.nanmin(redshifts), np.nanmax(redshifts), nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # -- Other galaxies (not extreme PSBs)
    medians_other, lower_other, upper_other = [], [], []
    for i in range(nbins):
        mask = (redshifts >= bins[i]) & (redshifts < bins[i+1]) & (~is_extreme_psb)
        vals = radius_kpc[mask]
        if np.sum(mask) > 0:
            medians_other.append(np.nanmedian(vals))
            lower_other.append(np.nanpercentile(vals, 16))
            upper_other.append(np.nanpercentile(vals, 84))
        else:
            medians_other.append(np.nan)
            lower_other.append(np.nan)
            upper_other.append(np.nan)
    plt.errorbar(bin_centers, medians_other, yerr=[np.array(medians_other)-np.array(lower_other), np.array(upper_other)-np.array(medians_other)],
                 fmt='o-', color='tomato', alpha=0.7, label='Other galaxies')

    # -- Extreme PSBs
    medians_psb, lower_psb, upper_psb = [], [], []
    for i in range(nbins):
        mask = (redshifts >= bins[i]) & (redshifts < bins[i+1]) & is_extreme_psb
        vals = radius_kpc[mask]
        if np.sum(mask) > 0:
            medians_psb.append(np.nanmedian(vals))
            lower_psb.append(np.nanpercentile(vals, 16))
            upper_psb.append(np.nanpercentile(vals, 84))
        else:
            medians_psb.append(np.nan)
            lower_psb.append(np.nan)
            upper_psb.append(np.nan)
    plt.errorbar(bin_centers, medians_psb, yerr=[np.array(medians_psb)-np.array(lower_psb), np.array(upper_psb)-np.array(medians_psb)],
                 fmt='o-', color='royalblue', alpha=0.7, label='Extreme PSBs')

    # --- Binned Westcott Data ---
    external_fits = "/raid/scratch/work/Griley/GALFIND_WORK/EPOCHS_XI_structural_parameters.fits"
    external_table = load_fits_table(external_fits, hdu_index=1)
    external_redshift = external_table['zbest_fsps_larson']
    external_radius_kpc = external_table['re_kpc']
    # Use same bins for fair comparison
    medians_west, lower_west, upper_west = [], [], []
    for i in range(nbins):
        mask = (external_redshift >= bins[i]) & (external_redshift < bins[i+1])
        vals = external_radius_kpc[mask]
        if np.sum(mask) >= 5:
            medians_west.append(np.nanmedian(vals))
            lower_west.append(np.nanpercentile(vals, 16))
            upper_west.append(np.nanpercentile(vals, 84))
        else:
            medians_west.append(np.nan)
            lower_west.append(np.nan)
            upper_west.append(np.nan)
    plt.errorbar(bin_centers, medians_west, yerr=[np.array(medians_west)-np.array(lower_west), np.array(upper_west)-np.array(medians_west)],
                 fmt='o-', color='slategrey', alpha=1, markerfacecolor='none', label='Westcott:EPOCHS-XI')

    plt.yscale('log')
    plt.xlabel('Redshift')
    plt.ylabel('Median Effective Radius (kpc)')
    if mass_cut_applied:
        plt.title('Binned: Redshift vs Effective Radius (log M$_* > 8.1$ cut)')
    else:
        plt.title('Binned: Redshift vs Effective Radius (no mass cut)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()
    plt.close()



def main(
    phot_fits, bagpipes_fits, galfit_fits, 
    filter_name='F444W', phot_idcol='NUMBER', bagpipes_idcol='#ID'
):
    # Load your tables as before:
    table_objects = load_fits_table(phot_fits, hdu_index=1)
    with fits.open(bagpipes_fits) as hdulist:
        table_bagpipes = Table(hdulist[4].data)
    table_galfit = load_fits_table(galfit_fits, hdu_index=1)

    # Match ALL THREE by ID:
    table_objects_matched, table_bagpipes_matched, table_galfit_matched = match_three_tables_by_id(
        table_objects, table_bagpipes, table_galfit,
        phot_idcol, bagpipes_idcol, 'id'
    )

    # Scale mass and SFR
    R = compute_flux_ratio(table_objects_matched, filter_name)
    logR = np.log10(R)
    scale_mass_sfr_log(table_bagpipes_matched, logR)

    # Extract redshifts and compute radius in kpc
    redshifts = table_bagpipes_matched['input_redshift']
    radius_kpc = get_radius_kpc(table_galfit_matched, redshifts)

    # Mask for reliable fits
    reliable_mask = flag_unreliable_fits(table_galfit_matched, radius_kpc, redshifts)
    radius_kpc_clean = radius_kpc[reliable_mask]

    # Apply mask to Bagpipes parameters (assumes ordering matches GALFIT)
    stellar_mass_scaled = table_bagpipes_matched['stellar_mass_50'][reliable_mask]
    burstiness_clean = table_bagpipes_matched['burstiness_50'][reliable_mask]
    halpha_clean = table_bagpipes_matched['Halpha_EW_rest_50'][reliable_mask]

    # Extreme PSB mask
    is_extreme_psb = (burstiness_clean <= 1) & (halpha_clean <= 100)
    
    plot_binned_mass_vs_radius(stellar_mass_scaled, radius_kpc_clean, is_extreme_psb, nbins=16, savefig='binned_mass_vs_radius.png')
    mass_cut_mask = stellar_mass_scaled > 8.1
    plot_binned_redshift_vs_radius(redshifts[reliable_mask][mass_cut_mask], radius_kpc_clean[mass_cut_mask], is_extreme_psb[mass_cut_mask], nbins=12, savefig='binned_redshift_vs_radius_masscut.png', mass_cut_applied=True)



if __name__ == "__main__":
    main(
        phot_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        bagpipes_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        galfit_fits="/raid/scratch/work/Griley/GALFIND_WORK/GALFIT/output/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/F444W/sersic/results.fits",
        filter_name='F444W',
        phot_idcol='NUMBER',
        bagpipes_idcol='#ID'
    )


