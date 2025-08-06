import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import os

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

def load_mass_pdf(pdf_dir, galaxy_id):
    """Loads the Bagpipes PDF for stellar mass for a given galaxy ID."""
    pdf_path = os.path.join(pdf_dir, f"{galaxy_id}.txt")
    # Try to handle various file shapes (grid or sample)
    samples = np.loadtxt(pdf_path)
    # Sometimes it's just a list of samples
    if samples.ndim == 1:
        return samples

    else:
        raise ValueError("Unknown PDF file format")
    

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

def monte_carlo_mass_radius(
    table_galfit_matched, table_bagpipes_matched, 
    pdf_dir, id_column, n_samples=500
):
    """
    Monte Carlo sampling with proper error handling and physical constraints
    """
    all_mass = []
    all_radius = []
    galaxy_ids = table_galfit_matched[id_column]
    redshifts = table_bagpipes_matched['input_redshift']

    for i, gal_id in enumerate(galaxy_ids):
        # Load stellar mass PDF samples for this galaxy
        try:
            mass_samples = load_mass_pdf(pdf_dir, gal_id)
        except Exception as e:
            print(f"Skipping galaxy {gal_id}: {e}")
            continue
            
        # Draw n_samples from mass_samples
        if len(mass_samples) > n_samples:
            mass_samples = np.random.choice(mass_samples, n_samples, replace=False)
        else:
            mass_samples = np.random.choice(mass_samples, n_samples, replace=True)

        # Get radius parameters
        r_e_pix = table_galfit_matched['r_e'][i]
        r_e_err_pix = table_galfit_matched['r_e_u1'][i]
        
        # **CRITICAL FIX**: Apply physical constraints to radius sampling
        # Use log-normal sampling to avoid negative values and extreme outliers
        # Convert to log space for sampling
        log_r_e = np.log(max(r_e_pix, 0.1))  # Ensure positive value
        log_r_e_err = r_e_err_pix / max(r_e_pix, 0.1)  # Relative error
        
        # Sample in log space and convert back
        log_r_e_samples = np.random.normal(log_r_e, min(log_r_e_err, 0.5), n_samples)
        r_e_samples_pix = np.exp(log_r_e_samples)
        
        # Apply hard physical constraints on pixel radius
        r_e_samples_pix = np.clip(r_e_samples_pix, 0.5, 100)  # 0.5-100 pixels is reasonable
        
        # Convert to kpc
        z = redshifts[i]
        r_e_samples_kpc = get_radius_kpc(
            Table({'r_e': r_e_samples_pix}), 
            np.full(n_samples, z)
        )

        # **CRITICAL FIX**: Apply strict physical constraints on final radius
        # Galaxies should be between 0.05 and 15 kpc effective radius
        valid = (r_e_samples_kpc > 0.05) & (r_e_samples_kpc < 15.0) & np.isfinite(r_e_samples_kpc)
        
        r_e_samples_kpc = r_e_samples_kpc[valid]
        mass_samples = mass_samples[valid]
        
        if len(r_e_samples_kpc) > 0:  # Only add if we have valid samples
            all_mass.extend(mass_samples)
            all_radius.extend(r_e_samples_kpc)

    return np.array(all_mass), np.array(all_radius)


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

def plot_mass_vs_radius_with_hexbin(
    stellar_mass_samples, radius_samples,
    stellar_mass, radius_kpc, is_extreme_psb, nbins=16, savefig=None
):
    """
    Fixed hexbin plot with proper log scale handling
    """
    plt.figure(figsize=(12, 8), facecolor='white')
    
    # Filter out invalid samples and apply physical constraints
    valid_samples = (
        (stellar_mass_samples > 6.0) & (stellar_mass_samples < 12.0) &
        (radius_samples > 0.05) & (radius_samples < 15.0) &
        np.isfinite(stellar_mass_samples) & np.isfinite(radius_samples)
    )
    
    mass_clean = stellar_mass_samples[valid_samples]
    radius_clean = radius_samples[valid_samples]
    
    print(f"Valid samples for hexbin: {len(mass_clean)} out of {len(stellar_mass_samples)}")
    print(f"Mass range: {mass_clean.min():.2f} to {mass_clean.max():.2f}")
    print(f"Radius range: {radius_clean.min():.3f} to {radius_clean.max():.2f} kpc")
    
    if len(mass_clean) == 0:
        print("No valid samples for hexbin plot!")
        return
    
    # **CRITICAL FIX**: Transform radius to log space BEFORE hexbin
    # This ensures hexagons maintain their shape on the log-scale plot
    log_radius_clean = np.log10(radius_clean)
    
    # Calculate extent in transformed coordinates
    mass_extent = [mass_clean.min() - 0.1, mass_clean.max() + 0.1]
    log_radius_extent = [np.log10(0.05), np.log10(15.0)]  # Physical limits in log space
    extent = [mass_clean.min(), mass_clean.max(), 0.05, 15]

    
    # Create hexbin in linear space (mass vs log_radius)
    hb = plt.hexbin(
        mass_clean, radius_clean,  # Use log-transformed radius
        gridsize=35,
        bins='log',
        cmap='Blues',
        mincnt=5,
        alpha=0.7,
        extent=extent,
        zorder=1
    )
    
    # Add colorbar
    cb = plt.colorbar(hb, label='log₁₀(MC samples)', shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=10)
    
    # **BINNED STATISTICS** (same as before, but plotted in log space)
    mass_range = np.nanmax(stellar_mass) - np.nanmin(stellar_mass)
    bins = np.linspace(np.nanmin(stellar_mass) - 0.05*mass_range, 
                      np.nanmax(stellar_mass) + 0.05*mass_range, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Other galaxies (not extreme PSBs)
    medians_other, lower_other, upper_other = [], [], []
    for i in range(nbins):
        mask = (stellar_mass >= bins[i]) & (stellar_mass < bins[i+1]) & (~is_extreme_psb)
        vals = radius_kpc[mask]
        if np.sum(mask) >= 2:
            medians_other.append(np.nanmedian(vals))
            lower_other.append(np.nanpercentile(vals, 16))
            upper_other.append(np.nanpercentile(vals, 84))
        else:
            medians_other.append(np.nan)
            lower_other.append(np.nan)
            upper_other.append(np.nan)
    
    plt.errorbar(
        bin_centers, medians_other, 
        yerr=[np.array(medians_other)-np.array(lower_other), 
              np.array(upper_other)-np.array(medians_other)],
        fmt='o-', color='red', alpha=1.0, linewidth=3, markersize=8,
        markeredgecolor='darkred', markeredgewidth=1.5,
        zorder=10, label='Other galaxies (median ± 1σ)',
        capsize=4, capthick=2
    )

    # Extreme PSBs
    medians_psb, lower_psb, upper_psb = [], [], []
    for i in range(nbins):
        mask = (stellar_mass >= bins[i]) & (stellar_mass < bins[i+1]) & is_extreme_psb
        vals = radius_kpc[mask]
        if np.sum(mask) >= 2:
            medians_psb.append(np.nanmedian(vals))
            lower_psb.append(np.nanpercentile(vals, 16))
            upper_psb.append(np.nanpercentile(vals, 84))
        else:
            medians_psb.append(np.nan)
            lower_psb.append(np.nan)
            upper_psb.append(np.nan)
    
    plt.errorbar(
        bin_centers, medians_psb, 
        yerr=[np.array(medians_psb)-np.array(lower_psb), 
              np.array(upper_psb)-np.array(medians_psb)],
        fmt='s-', color='navy', alpha=1.0, linewidth=3, markersize=8,
        markeredgecolor='white', markeredgewidth=1.5,
        zorder=11, label='Extreme PSBs (median ± 1σ)',
        capsize=4, capthick=2
    )
    
    # External comparison data
    try:
        external_fits = "/raid/scratch/work/Griley/GALFIND_WORK/EPOCHS_XI_structural_parameters.fits"
        external_table = load_fits_table(external_fits, hdu_index=1)
        external_mass = external_table['stellar_mass_50']
        external_radius_kpc = external_table['re_kpc']
        
        valid_ext = (
            (external_mass > 6.0) & (external_mass < 12.0) &
            (external_radius_kpc > 0.05) & (external_radius_kpc < 15.0) &
            np.isfinite(external_mass) & np.isfinite(external_radius_kpc)
        )
        
        plt.scatter(
            external_mass[valid_ext], external_radius_kpc[valid_ext],
            marker='D', facecolors='gold', edgecolors='darkorange', 
            s=40, alpha=0.8, linewidth=1, zorder=12,
            label='Westcott: EPOCHS-XI'
        )
    except Exception as e:
        print(f"Could not load external comparison data: {e}")

    # **SET LOG SCALE AFTER CREATING HEXBIN**
    plt.yscale('log')
    
    # Formatting
    plt.xlabel('Stellar Mass (log₁₀ M☉)', fontsize=14, fontweight='bold')
    plt.ylabel('Effective Radius (kpc)', fontsize=14, fontweight='bold')
    plt.title('Stellar Mass vs Effective Radius\nMonte Carlo Distribution + Binned Trends', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Mass completeness line
    plt.axvline(8.1, color='gray', linestyle='--', linewidth=3, alpha=0.8, 
                label='90% mass completeness', zorder=5)
    
    # Set axis limits in actual coordinates (matplotlib handles the log transform)
    plt.xlim(mass_extent[0], mass_extent[1])
    plt.ylim(0.05, 15.0)  # Linear limits, will be transformed to log
    
    # Grid and legend
    plt.grid(True, linestyle=':', alpha=0.5, zorder=0)
    plt.legend(fontsize=11, loc='upper left', framealpha=0.9)
    plt.tight_layout()
    
    # Statistics text box
    textstr = f'MC samples: {len(mass_clean):,}\nRadius: {radius_clean.min():.2f}-{radius_clean.max():.2f} kpc'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved fixed hexbin plot to {savefig}")
    
    plt.show()
    plt.close()

def main(
    phot_fits, bagpipes_fits, galfit_fits, 
    filter_name='F444W', phot_idcol='NUMBER', bagpipes_idcol='#ID',     
    pdf_dir = "/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/pdfs/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/temp/stellar_mass"
):
    # Load tables (existing code)
    table_objects = load_fits_table(phot_fits, hdu_index=1)
    with fits.open(bagpipes_fits) as hdulist:
        table_bagpipes = Table(hdulist[4].data)
    table_galfit = load_fits_table(galfit_fits, hdu_index=1)

    # Match tables (existing code)
    table_objects_matched, table_bagpipes_matched, table_galfit_matched = match_three_tables_by_id(
        table_objects, table_bagpipes, table_galfit,
        phot_idcol, bagpipes_idcol, 'id'
    )

    # Scale mass and SFR (existing code)
    R = compute_flux_ratio(table_objects_matched, filter_name)
    logR = np.log10(R)
    scale_mass_sfr_log(table_bagpipes_matched, logR)

    # Extract redshifts and compute radius
    redshifts = table_bagpipes_matched['input_redshift']
    radius_kpc = get_radius_kpc(table_galfit_matched, redshifts)

    # **CRITICAL FIX**: Apply reliability mask AND physical constraints
    reliable_mask = flag_unreliable_fits(table_galfit_matched, radius_kpc, redshifts)
    
    # Additional physical constraints
    physical_mask = (radius_kpc > 0.05) & (radius_kpc < 15.0) & np.isfinite(radius_kpc)
    
    # Combine masks
    final_mask = reliable_mask & physical_mask
    
    print(f"Reliable fits: {np.sum(reliable_mask)} out of {len(reliable_mask)}")
    print(f"Physical constraints: {np.sum(physical_mask)} out of {len(physical_mask)}")
    print(f"Final sample: {np.sum(final_mask)} galaxies")

    # Apply final mask
    radius_kpc_clean = radius_kpc[final_mask]
    stellar_mass_scaled = table_bagpipes_matched['stellar_mass_50'][final_mask]
    burstiness_clean = table_bagpipes_matched['burstiness_50'][final_mask]
    halpha_clean = table_bagpipes_matched['Halpha_EW_rest_50'][final_mask]
    redshifts_clean = redshifts[final_mask]

    # Extreme PSB mask
    is_extreme_psb = (burstiness_clean <= 1) & (halpha_clean <= 100)
    
    # **ONLY SAMPLE FROM CLEAN, RELIABLE GALAXIES**
    mass_samples, radius_samples = monte_carlo_mass_radius(
        table_galfit_matched[final_mask], table_bagpipes_matched[final_mask],
        pdf_dir, id_column='id', n_samples=500
    )

    # Create improved plot
    plot_mass_vs_radius_with_hexbin(
        mass_samples, radius_samples,
        stellar_mass_scaled, radius_kpc_clean, is_extreme_psb,
        nbins=16, savefig='mass_vs_radius_hexbin_fixed.png'
    )

if __name__ == "__main__":
    main(
        phot_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        bagpipes_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        galfit_fits="/raid/scratch/work/Griley/GALFIND_WORK/GALFIT/output/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/F444W/sersic/results.fits",
        filter_name='F444W',
        phot_idcol='NUMBER',
        bagpipes_idcol='#ID'
    )




