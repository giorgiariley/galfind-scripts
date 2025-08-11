import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import os
import matplotlib.ticker as mticker

def pretty_log_y_as_decimals(ax, tick_candidates=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)):
    ax.set_yscale('log')
    ymin, ymax = ax.get_ylim()
    ticks = [t for t in tick_candidates if ymin <= t <= ymax]
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos: ('{:.3f}'.format(y)).rstrip('0').rstrip('.'))
    )
    # keep minor ticks for grid but don’t label them
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2,10)*0.1))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

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
    table['stellar_mass_16'] += logR
    table['stellar_mass_84'] += logR
    return table

def get_radius_kpc(table_galfit, redshifts, pixel_scale=0.03):
    """Calculates effective radius in kpc given GALFIT table and redshifts."""
    radius_pixels = table_galfit['r_e']
    radius_err_pixels = table_galfit['r_e_u1']
    radius_arcsec = radius_pixels * pixel_scale
    radius_err_arcsec = radius_err_pixels * pixel_scale
    DA_kpc_per_arcsec = np.array([
        (cosmo.angular_diameter_distance(z).to(u.kpc) / u.radian.to(u.arcsec)).value
        for z in redshifts
    ])
    return radius_arcsec * DA_kpc_per_arcsec, radius_err_arcsec * DA_kpc_per_arcsec

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
        r_e_samples_kpc, _ = get_radius_kpc(Table({'r_e': r_e_samples_pix}), np.full(n_samples, z))


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

def plot_mass_vs_radius(stellar_mass, stellar_mass_16, stellar_mass_84, radius_kpc, radius_kpc_err, is_extreme_psb, savefig=None):
    """Scatter plot of mass vs. effective radius, highlighting extreme PSBs."""
    plt.figure(figsize=(10, 6), facecolor='white')

    stellar_mass = np.array(stellar_mass)
    stellar_mass_16 = np.array(stellar_mass_16)
    stellar_mass_84 = np.array(stellar_mass_84)
    #load errors
    stellar_mass_err_lower = stellar_mass - stellar_mass_16
    stellar_mass_err_upper = stellar_mass_84 - stellar_mass

    
    # Plot others
    plt.errorbar(
    stellar_mass[~is_extreme_psb],
    radius_kpc[~is_extreme_psb],
    xerr=[stellar_mass_err_lower[~is_extreme_psb], stellar_mass_err_upper[~is_extreme_psb]], yerr = radius_kpc_err[~is_extreme_psb],
    fmt='o',
    color='tomato',
    alpha=0.6,
    ecolor='tomato',
    elinewidth=0.8,
    capsize=2,
    label='All other galaxies'
    )
    # Plot extreme PSBs
    plt.errorbar(
    stellar_mass[is_extreme_psb],
    radius_kpc[is_extreme_psb],
    xerr=[stellar_mass_err_lower[is_extreme_psb], stellar_mass_err_upper[is_extreme_psb]], yerr = radius_kpc_err[is_extreme_psb],
    fmt='o',
    color='royalblue',
    alpha=0.9,
    ecolor='royalblue',
    elinewidth=0.8,
    capsize=2,
    markeredgecolor='black',
    markeredgewidth=0.2,
    label='Extreme PSBs (burstiness ≤ 1 & Hα EW ≤ 100 Å)'
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


def plot_radius_vs_redshift(redshifts, radius_kpc, is_extreme_psb, savefig=None, mass_cut_applied=False):
    """Scatter plot of redshift vs. effective radius, highlighting extreme PSBs."""
    plt.figure(figsize=(10, 6), facecolor='white')
    # Plot others
    plt.scatter(
        redshifts[~is_extreme_psb], radius_kpc[~is_extreme_psb],
        color='tomato', alpha=0.6, edgecolor='none', label='All other galaxies'
    )
    # Plot extreme PSBs
    plt.scatter(
        redshifts[is_extreme_psb], radius_kpc[is_extreme_psb],
        color='royalblue', alpha=0.9, edgecolor='black', linewidth=0.2, label='Extreme PSBs (burstiness ≤ 1 & Hα EW ≤ 100 Å)'
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

def plot_binned_mass_vs_radius(
    stellar_mass, stellar_mass_16, stellar_mass_84,
    radius_kpc, is_extreme_psb, nbins=16, savefig=None, min_per_bin=3
):
    """
    Same binning as before; just styles error bars to be fainter.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    soft = lambda c, a: mcolors.to_rgba(c, a)  # same color, custom alpha
    c_other, c_psb, c_west = 'tomato', 'royalblue', 'slategrey'

    sm   = np.asarray(stellar_mass)
    sm16 = np.asarray(stellar_mass_16)
    sm84 = np.asarray(stellar_mass_84)
    rk   = np.asarray(radius_kpc)
    good = np.isfinite(sm) & np.isfinite(sm16) & np.isfinite(sm84) & np.isfinite(rk)
    sm, sm16, sm84, rk, is_extreme_psb = sm[good], sm16[good], sm84[good], rk[good], is_extreme_psb[good]

    bins = np.linspace(np.nanmin(sm), np.nanmax(sm), nbins + 1)

    def bin_stats(mask):
        x_med, xerr_lo, xerr_hi = [], [], []
        y_med, y_lo, y_hi = [], [], []
        for i in range(nbins):
            m = (sm >= bins[i]) & (sm < bins[i+1]) & mask
            if np.sum(m) >= min_per_bin:
                m50 = sm[m]
                x_med.append(np.nanmedian(m50))
                xerr_lo.append(np.nanmedian(np.maximum(0.0, m50 - sm16[m])))
                xerr_hi.append(np.nanmedian(np.maximum(0.0, sm84[m] - m50)))

                vals = rk[m]
                y50 = np.nanmedian(vals)
                y16 = np.nanpercentile(vals, 16)
                y84 = np.nanpercentile(vals, 84)
                y_med.append(y50)
                y_lo.append(y50 - y16)
                y_hi.append(y84 - y50)
            else:
                x_med.append(np.nan); xerr_lo.append(np.nan); xerr_hi.append(np.nan)
                y_med.append(np.nan); y_lo.append(np.nan);  y_hi.append(np.nan)

        x_med = np.array(x_med); y_med = np.array(y_med)
        xerr  = np.vstack([np.array(xerr_lo), np.array(xerr_hi)])
        yerr  = np.vstack([np.array(y_lo),  np.array(y_hi)])
        finite = np.isfinite(x_med) & np.isfinite(y_med)
        return x_med[finite], xerr[:, finite], y_med[finite], yerr[:, finite]

    plt.figure(figsize=(9,6))

    # other galaxies
    x_o, xerr_o, y_o, yerr_o = bin_stats(~is_extreme_psb)
    if x_o.size:
        plt.errorbar(
            x_o, y_o, xerr=xerr_o, yerr=yerr_o,
            fmt='o-', color=c_other, lw=1.6, ms=5,
            ecolor=soft(c_other, 0.35), elinewidth=1.0, capsize=2, capthick=1,
            label='Other galaxies', zorder=3
        )

    # extreme PSBs
    x_p, xerr_p, y_p, yerr_p = bin_stats(is_extreme_psb)
    if x_p.size:
        plt.errorbar(
            x_p, y_p, xerr=xerr_p, yerr=yerr_p,
            fmt='o-', color=c_psb, lw=1.6, ms=5,
            ecolor=soft(c_psb, 0.35), elinewidth=1.0, capsize=2, capthick=1,
            label='Extreme PSBs', zorder=4
        )

    # Westcott comparison (no x-errors)
    ext = load_fits_table("/raid/scratch/work/Griley/GALFIND_WORK/EPOCHS_XI_structural_parameters.fits", hdu_index=1)
    em = np.asarray(ext['stellar_mass_50']); er = np.asarray(ext['re_kpc'])
    ok = np.isfinite(em) & np.isfinite(er); em, er = em[ok], er[ok]
    x_w, y_w, yerr_w = [], [], []
    for i in range(nbins):
        m = (em >= bins[i]) & (em < bins[i+1])
        if np.sum(m) >= 5:
            xm = np.nanmedian(em[m])
            vals = er[m]
            y50 = np.nanmedian(vals); y16 = np.nanpercentile(vals, 16); y84 = np.nanpercentile(vals, 84)
            x_w.append(xm); y_w.append(y50); yerr_w.append([y50 - y16, y84 - y50])
    if x_w:
        yerr_w = np.array(yerr_w).T
        plt.errorbar(
            np.array(x_w), np.array(y_w), yerr=yerr_w,
            fmt='o-', color=c_west, lw=1.6, ms=4, mfc='white',
            ecolor=soft(c_west, 0.35), elinewidth=1.0, capsize=2, capthick=1,
            label='Westcott:EPOCHS-XI', zorder=2
        )

    plt.yscale('log')
    plt.xlabel('Stellar Mass (log$_{10}$ M$_\\odot$, scaled)')
    plt.ylabel('Median Effective Radius (kpc)')
    plt.title('Binned: Stellar Mass vs Effective Radius')
    plt.axvline(8.1, color='gray', linestyle='--', lw=1.2, label='90% completeness')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(framealpha=0.9)
    ax = plt.gca()
# (optional) enforce your usual limits:
# ax.set_ylim(0.05, 15)
    pretty_log_y_as_decimals(ax)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300)
    plt.show()
    plt.close()



def plot_binned_redshift_vs_radius(
    redshifts, radius_kpc, is_extreme_psb,
    nbins=12, savefig=None, mass_cut_applied=False, min_per_bin=3,
    err_alpha=0.7
):
    """
    Median R_eff vs redshift (binned). y-err = [median-16th, 84th-median].
    No x-err. Error bars drawn faintly; markers/lines solid.
    """
    # clean inputs
    z = np.asarray(redshifts)
    r = np.asarray(radius_kpc)
    psb = np.asarray(is_extreme_psb, dtype=bool)
    good = np.isfinite(z) & np.isfinite(r)
    z, r, psb = z[good], r[good], psb[good]

    bins = np.linspace(np.nanmin(z), np.nanmax(z), nbins + 1)
    xcent = 0.5 * (bins[:-1] + bins[1:])

    def y_stats(sel):
        y50, ylo, yhi = [], [], []
        for i in range(nbins):
            m = (z >= bins[i]) & (z < bins[i+1]) & sel
            if np.sum(m) >= min_per_bin:
                vals = r[m]
                med = np.nanmedian(vals)
                p16 = np.nanpercentile(vals, 16)
                p84 = np.nanpercentile(vals, 84)
                y50.append(med)
                ylo.append(med - p16)
                yhi.append(p84 - med)
            else:
                y50.append(np.nan); ylo.append(np.nan); yhi.append(np.nan)
        y50 = np.array(y50)
        yerr = np.vstack([np.array(ylo), np.array(yhi)])
        keep = np.isfinite(y50)
        return xcent[keep], y50[keep], yerr[:, keep]

    fig, ax = plt.subplots(figsize=(9,6))

    # Other galaxies
    xo, yo, yerr_o = y_stats(~psb)
    if xo.size:
        ax.plot(xo, yo, 'o-', color='tomato', alpha=0.9, label='Other galaxies', zorder=3)
        ax.errorbar(xo, yo, yerr=yerr_o, fmt='none', ecolor='tomato',
                    elinewidth=1.0, capsize=3, alpha=err_alpha, zorder=2)

    # Extreme PSBs
    xp, yp, yerr_p = y_stats(psb)
    if xp.size:
        ax.plot(xp, yp, 'o-', color='royalblue', alpha=0.95,
                markeredgecolor='white', markeredgewidth=0.8,
                label='Extreme PSBs', zorder=4)
        ax.errorbar(xp, yp, yerr=yerr_p, fmt='none', ecolor='royalblue',
                    elinewidth=1.0, capsize=3, alpha=err_alpha, zorder=2)

    # Westcott comparison, binned on the same z bins (no x-err)
    external_fits = "/raid/scratch/work/Griley/GALFIND_WORK/EPOCHS_XI_structural_parameters.fits"
    ext = load_fits_table(external_fits, hdu_index=1)
    zext = np.asarray(ext['zbest_fsps_larson'])
    rext = np.asarray(ext['re_kpc'])
    ok = np.isfinite(zext) & np.isfinite(rext)
    zext, rext = zext[ok], rext[ok]

    xw, yw, yerr_w = [], [], []
    for i in range(nbins):
        m = (zext >= bins[i]) & (zext < bins[i+1])
        if np.sum(m) >= 5:
            med = np.nanmedian(rext[m])
            p16 = np.nanpercentile(rext[m], 16)
            p84 = np.nanpercentile(rext[m], 84)
            xw.append(xcent[i]); yw.append(med); yerr_w.append([med-p16, p84-med])
    if xw:
        xw = np.array(xw); yw = np.array(yw); yerr_w = np.array(yerr_w).T
        ax.plot(xw, yw, 'o-', color='slategrey', markerfacecolor='none',
                label='Westcott:EPOCHS-XI', zorder=3)
        ax.errorbar(xw, yw, yerr=yerr_w, fmt='none', ecolor='slategrey',
                    elinewidth=1.0, capsize=3, alpha=err_alpha, zorder=2)

    # Axes, styling
    ax.set_yscale('log')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Median Effective Radius (kpc)')
    ax.set_title('Binned: Redshift vs Effective Radius' + (' (log M$_* > 8.1$ cut)' if mass_cut_applied else ''))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # decimal tick labels on log y
    pretty_log_y_as_decimals(ax)

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
    # Load tables
    table_objects = load_fits_table(phot_fits, hdu_index=1)
    with fits.open(bagpipes_fits) as hdulist:
        table_bagpipes = Table(hdulist[4].data)
    table_galfit = load_fits_table(galfit_fits, hdu_index=1)

    # Match tables
    table_objects_matched, table_bagpipes_matched, table_galfit_matched = match_three_tables_by_id(
        table_objects, table_bagpipes, table_galfit,
        phot_idcol, bagpipes_idcol, 'id'
    )

    # Scale mass by flux ratio
    R = compute_flux_ratio(table_objects_matched, filter_name)
    logR = np.log10(R)
    scale_mass_sfr_log(table_bagpipes_matched, logR)

    # Get redshifts and radii
    redshifts = table_bagpipes_matched['input_redshift']
    radius_kpc, radius_kpc_err = get_radius_kpc(table_galfit_matched, redshifts)

    # Reliability and physical constraints
    reliable_mask = flag_unreliable_fits(table_galfit_matched, radius_kpc, redshifts)
    physical_mask = (radius_kpc > 0.05) & (radius_kpc < 15.0) & np.isfinite(radius_kpc)
    good_err_mask = (radius_kpc_err / radius_kpc) < 0.5

    final_mask = reliable_mask & physical_mask & good_err_mask

    # Apply mask
    radius_kpc_clean = radius_kpc[final_mask]
    stellar_mass_scaled = table_bagpipes_matched['stellar_mass_50'][final_mask]
    burstiness_clean = table_bagpipes_matched['burstiness_50'][final_mask]
    halpha_clean = table_bagpipes_matched['Halpha_EW_rest_50'][final_mask]
    redshifts_clean = redshifts[final_mask]
    stellar_mass_scaled_16 = table_bagpipes_matched['stellar_mass_16'][final_mask]
    stellar_mass_scaled_84 = table_bagpipes_matched['stellar_mass_84'][final_mask]
    radius_kpc_err = radius_kpc_err[final_mask]

    # Extreme PSB mask
    is_extreme_psb = (burstiness_clean <= 1) & (halpha_clean <= 100)
    print("median r_e, median r_e_u1:",
      np.nanmedian(table_galfit['r_e']),
      np.nanmedian(table_galfit['r_e_u1']))
    
    # sanity check one object (replace z0, r_pix0, e_pix0 with medians)
    z0 = np.nanmedian(redshifts)
    kpc_per_arcsec = (cosmo.angular_diameter_distance(z0).to(u.kpc).value) / u.radian.to(u.arcsec)
    pixscale = 0.03  # verify this!
    r_pix0   = 3.49
    e_pix0   = 0.1765

    r_kpc0   = r_pix0 * pixscale * kpc_per_arcsec
    e_kpc0   = e_pix0 * pixscale * kpc_per_arcsec
    print(r_kpc0, e_kpc0)   # should look reasonable (e.g., error ~ few % of value)


    # Monte Carlo sampling
    # mass_samples, radius_samples = monte_carlo_mass_radius(
    #     table_galfit_matched[final_mask], table_bagpipes_matched[final_mask],
    #     pdf_dir, id_column='id', n_samples=500
    # )

    # ---- PLOTS ----
    plot_mass_vs_radius(
        stellar_mass_scaled,stellar_mass_scaled_16, stellar_mass_scaled_84, radius_kpc_clean, radius_kpc_err, is_extreme_psb,
        savefig="mass_vs_radius.png"
    )

    # Apply mass cut for redshift–radius plots
    mass_cut_mask = stellar_mass_scaled > 8.1
    plot_radius_vs_redshift(
        redshifts_clean[mass_cut_mask], radius_kpc_clean[mass_cut_mask],
        is_extreme_psb[mass_cut_mask],
        savefig="radius_vs_redshift_masscut.png",
        mass_cut_applied=True
    )

    plot_binned_mass_vs_radius(
    stellar_mass_scaled, stellar_mass_scaled_16, stellar_mass_scaled_84,
    radius_kpc_clean, is_extreme_psb,
    nbins=16, savefig="binned_mass_vs_radius.png"
    )


    plot_binned_redshift_vs_radius(
        redshifts_clean[mass_cut_mask], radius_kpc_clean[mass_cut_mask],
        is_extreme_psb[mass_cut_mask],
        savefig="binned_redshift_vs_radius_masscut.png",
        mass_cut_applied=True
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




