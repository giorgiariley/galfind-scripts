import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import os
import matplotlib.ticker as mticker
from scipy.stats import ks_2samp
from matplotlib import colors as mcolors

plt.rcParams.update({
    "axes.labelsize": 15,   # axis label font
    "xtick.labelsize": 13,  # tick label font
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

def lighten(color, amount=0.5):
    """Blend a color toward white by `amount` (0=no change, 1=white)."""
    c = mcolors.to_rgb(color)
    return tuple(1 - amount*(1 - np.array(c)))

def fit_powerlaw_params(x, y, yerr=None, use_logy=True):
    """
    Fit log10(y) = a + b*(x - x0). Returns dict(a,b,x0,use_logy).
    Accepts asymmetric yerr as 2xN [lo,hi] in *linear* units.
    """
    x = np.asarray(x); y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[finite], y[finite]
    if yerr is not None:
        yerr = np.asarray(yerr)[:, finite] if (np.ndim(yerr) == 2 and yerr.shape[0] == 2) else np.asarray(yerr)[finite]
    # convert to log space if requested
    if use_logy:
        Y = np.log10(y)
        if yerr is None:
            w = None
        else:
            if yerr.ndim == 2 and yerr.shape[0] == 2:
                lo = np.clip(yerr[0], 1e-12, None); hi = np.clip(yerr[1], 1e-12, None)
                sig_lo = np.log10(y) - np.log10(np.maximum(y - lo, 1e-12))
                sig_hi = np.log10(np.maximum(y + hi, 1e-12)) - np.log10(y)
                sig = 0.5*(sig_lo + sig_hi)
            else:
                e = np.clip(yerr, 1e-12, None)
                sig = (np.log10(y + e) - np.log10(np.maximum(y - e, 1e-12)))/2.0
            w = 1.0/np.maximum(sig, 1e-6)
    else:
        Y = y
        if yerr is None:
            w = None
        else:
            if yerr.ndim == 2 and yerr.shape[0] == 2:
                yerr = 0.5*(yerr[0] + yerr[1])
            w = 1.0/np.maximum(yerr, 1e-6)

    x0 = np.nanmedian(x)
    X  = x - x0
    coeff = np.polyfit(X, Y, 1, w=w) if w is not None else np.polyfit(X, Y, 1)
    b, a = coeff[0], coeff[1]
    return dict(a=a, b=b, x0=x0, use_logy=use_logy)

def plot_powerlaw_line_over_data(ax, params, x_data, color='k', lw=2.2, zorder=3, pad_frac=0.03):
    """Draw the fitted line only across the data range of x_data (with a tiny pad)."""
    x_data = np.asarray(x_data)
    xmin, xmax = np.nanmin(x_data), np.nanmax(x_data)
    span = xmax - xmin if np.isfinite(xmax - xmin) and (xmax > xmin) else 1.0
    xr = np.linspace(xmin - pad_frac*span, xmax + pad_frac*span, 200)
    yr = params['a'] + params['b']*(xr - params['x0'])
    yline = 10**yr if params['use_logy'] else yr
    ax.plot(xr, yline, color=color, lw=lw, zorder=zorder)

def ks_size_by_mass_bin(stellar_mass, radius_kpc, is_extreme_psb,
                        bins=None, nbins=16, min_per_group=5,
                        log_radius=True, fdr=True,
                        plot=True, savefig=None):
    """
    KS-test of size (R_eff) distributions between groups in each mass bin.

    Parameters
    ----------
    stellar_mass : array (log10 M*)
    radius_kpc   : array (R_eff in kpc)
    is_extreme_psb : boolean array (True = blue sample, False = other)
    bins : arraylike or None
        If None, uses nbins between min/max(stellar_mass).
    nbins : int, used only when bins is None
    min_per_group : int, minimum N needed in BOTH groups to test
    log_radius : bool, if True test log10(R) (recommended)
    fdr : bool, apply Benjamini–Hochberg correction across bins
    plot : bool, draw a quick -log10(p) summary plot
    savefig : str or None, path to save the plot

    Returns
    -------
    results : list of dicts (one per bin) with keys:
        i, lo, hi, center, n_other, n_psb, ks, p, q, med_other, med_psb, med_diff
        (medians are in log10(kpc) if log_radius=True)
    """
    sm = np.asarray(stellar_mass)
    rk = np.asarray(radius_kpc)
    psb = np.asarray(is_extreme_psb, dtype=bool)
    good = np.isfinite(sm) & np.isfinite(rk) & np.isfinite(psb)
    sm, rk, psb = sm[good], rk[good], psb[good]

    if bins is None:
        bins = np.linspace(np.nanmin(sm), np.nanmax(sm), nbins + 1)
    else:
        bins = np.asarray(bins)
        nbins = len(bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])

    results = []
    tested_idx = []

    for i in range(nbins):
        inbin = (sm >= bins[i]) & (sm < bins[i+1])
        r_other = rk[inbin & (~psb)]
        r_psb   = rk[inbin & psb]

        if log_radius:
            r_other = np.log10(r_other)
            r_psb   = np.log10(r_psb)

        n0, n1 = len(r_other), len(r_psb)
        if (n0 < min_per_group) or (n1 < min_per_group):
            results.append(dict(i=i, lo=bins[i], hi=bins[i+1], center=centers[i],
                                n_other=n0, n_psb=n1, ks=np.nan, p=np.nan, q=np.nan,
                                med_other=np.nan, med_psb=np.nan, med_diff=np.nan))
            continue

        stat, p = ks_2samp(r_other, r_psb, alternative='two-sided', mode='auto')
        m0 = float(np.nanmedian(r_other))
        m1 = float(np.nanmedian(r_psb))

        results.append(dict(i=i, lo=bins[i], hi=bins[i+1], center=centers[i],
                            n_other=n0, n_psb=n1, ks=stat, p=p, q=np.nan,
                            med_other=m0, med_psb=m1, med_diff=m1 - m0))
        tested_idx.append(i)

    # Benjamini–Hochberg FDR across bins that were actually tested
    if fdr and tested_idx:
        pvals = np.array([results[i]['p'] for i in tested_idx])
        order = np.argsort(pvals)
        ranks = np.arange(1, len(pvals) + 1)
        q_sorted = pvals[order] * len(pvals) / ranks
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]  # monotone
        for k, idx in enumerate(order):
            results[tested_idx[idx]]['q'] = float(q_sorted[k])

    if plot and tested_idx:
        c   = np.array([results[i]['center'] for i in tested_idx])
        p   = np.array([results[i]['p'] for i in tested_idx])
        q   = np.array([results[i]['q'] for i in tested_idx])
        n0s = np.array([results[i]['n_other'] for i in tested_idx])
        n1s = np.array([results[i]['n_psb']   for i in tested_idx])

        plt.figure(figsize=(9,5))
        plt.plot(c, -np.log10(p), 'o-', label='KS p-value')
        if fdr:
            plt.plot(c, -np.log10(q), 's--', label='FDR q-value')
        plt.axhline(-np.log10(0.05), color='gray', ls='--', lw=1, label='0.05')
        plt.xlabel('Stellar Mass (log$_{10}$ M$_\\odot$)')
        plt.ylabel(r'$-\log_{10}(p)$')
        plt.title('KS test: R$_{\\rm eff}$ distributions (Other vs PSB) per mass bin'
                  + (' [log radius]' if log_radius else ''))
        # annotate bin sample sizes as N_other/N_psb near the bottom
        y0, y1 = plt.ylim()
        ytxt = y0 + 0.06*(y1 - y0)
        for xi, n0, n1 in zip(c, n0s, n1s):
            plt.text(xi, ytxt, f'{n0}/{n1}', ha='center', va='bottom', fontsize=8)
        plt.legend()
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig, dpi=200)
        plt.show()
        plt.close()

    return results

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

def plot_mass_vs_radius(stellar_mass, stellar_mass_16, stellar_mass_84,
                        radius_kpc, radius_kpc_err, is_extreme_psb, savefig=None):
    """Scatter plot of mass vs. effective radius with a single 'typical' error bar per series."""
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()

    stellar_mass      = np.array(stellar_mass)
    stellar_mass_16   = np.array(stellar_mass_16)
    stellar_mass_84   = np.array(stellar_mass_84)
    radius_kpc        = np.array(radius_kpc)
    radius_kpc_err    = np.array(radius_kpc_err)
    is_extreme_psb    = np.array(is_extreme_psb, dtype=bool)

    # per-object asymmetric mass errors
    stellar_mass_err_lower = stellar_mass - stellar_mass_16
    stellar_mass_err_upper = stellar_mass_84 - stellar_mass

    m_other = ~is_extreme_psb
    m_psb   =  is_extreme_psb

    # points only
    ax.scatter(stellar_mass[m_other], radius_kpc[m_other],
               s=18, color='Royalblue', alpha=0.6, edgecolor='none',
               label='All other galaxies')
    ax.scatter(stellar_mass[m_psb], radius_kpc[m_psb],
               s=22, color='tomato', alpha=0.9, edgecolor='black',
               linewidth=0.2,
               label='Extreme PSBs (burstiness ≤ 0.5 & Hα EW ≤ 25 Å)')

    # axes & styling
    ax.set_yscale('log')
    ax.set_xlabel('Stellar Mass (log$_{10}$ M$_\\odot$, scaled)')
    ax.set_ylabel('Effective Radius (kpc)')
    ax.axvline(8.1, color='gray', linestyle='--', label='90% completeness')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(6, 11)  # set limits before placing the example bars
    pretty_log_y_as_decimals(ax)


    # helper: place one typical error bar; returns its (x0, y0) in data coords
    def _add_typical_err(ax, mask, color, x_frac=0.035, y_frac=0.16):
        if not np.any(mask):
            return None, None
        ex_lo = np.nanmedian(stellar_mass_err_lower[mask])
        ex_hi = np.nanmedian(stellar_mass_err_upper[mask])
        ey    = np.nanmedian(radius_kpc_err[mask])
        if not (np.isfinite(ex_lo) and np.isfinite(ex_hi) and np.isfinite(ey)):
            return None, None

        xmin, xmax = ax.get_xlim()
        x0 = xmin + x_frac * (xmax - xmin)

        ymin, ymax = ax.get_ylim()
        if ax.get_yscale() == 'log':
            logy = np.log10(ymin) + y_frac * (np.log10(ymax) - np.log10(ymin))
            y0 = 10**logy
        else:
            y0 = ymin + y_frac * (ymax - ymin)

        # draw the bar
        ax.errorbar([x0], [y0],
                    xerr=[[ex_lo], [ex_hi]], yerr=[[ey], [ey]],
                    fmt='none', ecolor=color, elinewidth=1.8, capsize=4,
                    zorder=10, clip_on=False)
        return x0, y0

    # place the two typical bars (tomato a bit left of royalblue)
    x_lab, y_lab = _add_typical_err(ax, m_other, 'tomato',    x_frac=0.035, y_frac=0.16)
    _add_typical_err(ax,          m_psb,   'royalblue', x_frac=0.075, y_frac=0.12)

    # label next to the tomato bar
    if x_lab is not None:
        xmin, xmax = ax.get_xlim()
        ax.text(x_lab -0.023*(xmax - xmin), y_lab+0.05,
                'Typical error bars', fontsize=13, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))

    # external comparison
    external_fits = "/raid/scratch/work/Griley/GALFIND_WORK/EPOCHS_XI_structural_parameters.fits"
    external_table = load_fits_table(external_fits, hdu_index=1)
    ax.scatter(external_table['stellar_mass_50'], external_table['re_kpc'],
               marker='o', facecolors='none', edgecolors='slategrey', s=30,
               label='Westcott:EPOCHS-XI')

    ax.legend()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 10**0.2)  # ~25% higher in log space
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.close()

def plot_radius_vs_redshift(redshifts, radius_kpc, is_extreme_psb,
                            savefig=None, mass_cut_applied=False, radius_kpc_err=None):
    """Scatter plot of redshift vs. effective radius with two side-by-side 'typical' error bars."""
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()

    z  = np.asarray(redshifts)
    r  = np.asarray(radius_kpc)
    m  = np.asarray(is_extreme_psb, dtype=bool)
    re = None if radius_kpc_err is None else np.asarray(radius_kpc_err)

    ax.scatter(z[~m], r[~m], color='tomato',    alpha=0.6, edgecolor='none', s=18, label='All other galaxies')
    ax.scatter(z[m],  r[m],  color='royalblue', alpha=0.9, edgecolor='black', linewidth=0.2, s=22,
               label='Extreme PSBs (burstiness ≤ 0.5 & Hα EW ≤ 25 Å)')

    # Westcott comparison
    ext = load_fits_table("/raid/scratch/work/Griley/GALFIND_WORK/EPOCHS_XI_structural_parameters.fits", hdu_index=1)
    ax.scatter(ext['zbest_fsps_larson'], ext['re_kpc'],
               marker='o', facecolors='none', edgecolors='slategrey', s=30, label='Westcott:EPOCHS-XI')

    ax.set_yscale('log')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Effective Radius (kpc)')
    ax.grid(True, linestyle='--', alpha=0.5)
    pretty_log_y_as_decimals(ax)

    def _add_typical(ax, mask, color, x_frac=0.06, y_frac=0.10):
        """Return (x0, y0, ey) if drawn, else None."""
        if re is None or not np.any(mask):
            return None
        ey = np.nanmedian(re[mask])
        if not np.isfinite(ey):
            return None
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        x0 = xmin + x_frac*(xmax - xmin)
        if ax.get_yscale() == 'log':
            y0 = 10**(np.log10(ymin) + y_frac*(np.log10(ymax)-np.log10(ymin)))
        else:
            y0 = ymin + y_frac*(ymax - ymin)
        y0 = np.clip(y0, ymin + 1.2*ey, ymax - 1.2*ey)
        ax.errorbar([x0], [y0], yerr=[[ey],[ey]], fmt='none', ecolor=color,
                    elinewidth=1.8, capsize=4, zorder=10)
        return x0, y0, ey

    # two bars (offset in x so they don't overlap)
    pos_other = _add_typical(ax, ~m, 'royalblue',    x_frac=0.06,  y_frac=0.10)
    pos_psb   = _add_typical(ax,  m, 'tomato', x_frac=0.085, y_frac=0.10)

    # label slightly ABOVE the bars
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    ref = pos_other or pos_psb  # whichever exists
    if ref:
        x_bar, y_bar, ey_bar = ref
        x_label = xmin + 0.02*(xmax - xmin)             # a bit left of the bars
        y_label = min(ymax/1.02, y_bar + 1.4*ey_bar)     # just above the bar top
        ax.text(x_label, y_label, "Typical error bars",
                fontsize=13, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))

    ax.legend()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 10**0.1)  # ~25% higher in log space
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
    c_psb, c_other, c_west = 'tomato', 'royalblue', 'slategrey'

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
        pars_o = fit_powerlaw_params(x_o, y_o, yerr=yerr_o, use_logy=True)
        slope_o = pars_o['b']
        plot_powerlaw_line_over_data(plt.gca(), pars_o, x_o,
                                    color=lighten(c_other, 0.55), lw=2.4, zorder=2.5)
        plt.errorbar(
            x_o, y_o, xerr=xerr_o, yerr=yerr_o,
            fmt='o-', color=c_other, lw=1.8, ms=5,
            ecolor=soft(c_other, 0.35), elinewidth=1.0, capsize=2, capthick=1,
            label=f'Other galaxies (slope={slope_o:.2f})', zorder=3
        )


    # extreme PSBs (binned points)
    x_p, xerr_p, y_p, yerr_p = bin_stats(is_extreme_psb)
    if x_p.size:
        pars_p = fit_powerlaw_params(x_p, y_p, yerr=yerr_p, use_logy=True)
        slope_p = pars_p['b']
        plot_powerlaw_line_over_data(plt.gca(), pars_p, x_p,
                                    color=lighten(c_psb, 0.55), lw=2.4, zorder=4.2)
        plt.errorbar(
            x_p, y_p, xerr=xerr_p, yerr=yerr_p,
            fmt='o-', color=c_psb, lw=1.8, ms=5,
            ecolor=soft(c_psb, 0.35), elinewidth=1.0, capsize=2, capthick=1,
            label=f'Extreme PSBs (slope={slope_p:.2f})', zorder=4.5
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
    nbins=12, savefig=None, mass_cut_applied=True, min_per_bin=3,
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

        # Other galaxies (binned series + fit)
    xo, yo, yerr_o = y_stats(~psb)
    if xo.size:
        pars_o = fit_powerlaw_params(xo, yo, yerr=yerr_o, use_logy=True)
        slope_o = pars_o['b']
        # light fit line behind
        plot_powerlaw_line_over_data(ax, pars_o, xo,
                                     color=lighten('royalblue', 0.55),
                                     lw=2.4, zorder=3.0)
        # dark binned series with label incl. slope
        ax.plot(xo, yo, 'o-', color='royalblue', alpha=0.95,
                label=f'Other galaxies (slope={slope_o:.2f})', zorder=3.5)
        ax.errorbar(xo, yo, yerr=yerr_o, fmt='none', ecolor='royalblue',
                    elinewidth=1.0, capsize=3, alpha=err_alpha, zorder=3.3)

    # Extreme PSBs (binned series + fit)
    xp, yp, yerr_p = y_stats(psb)
    if xp.size:
        pars_p = fit_powerlaw_params(xp, yp, yerr=yerr_p, use_logy=True)
        slope_p = pars_p['b']
        plot_powerlaw_line_over_data(ax, pars_p, xp,
                                     color=lighten('tomato', 0.55),
                                     lw=2.4, zorder=4.0)
        ax.plot(xp, yp, 'o-', color='tomato', alpha=0.95,
                markeredgecolor='white', markeredgewidth=0.8,
                label=f'Extreme PSBs (slope={slope_p:.2f})', zorder=4.5)
        ax.errorbar(xp, yp, yerr=yerr_p, fmt='none', ecolor='tomato',
                    elinewidth=1.0, capsize=3, alpha=err_alpha, zorder=4.3)



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
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # decimal tick labels on log y
    pretty_log_y_as_decimals(ax)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()
    plt.close()

def plot_size_vs_param_in_mass_bins(
    logM, Re_kpc, param, param_label,
    mass_bins = (8.1, 8.6, 9.0, 9.5,),
    nbins_param=5, min_per_bin=1, savefig=None,
    logy=True, logx=False,
    # NEW optional knobs (safe defaults match your old behavior)
    use_global_bins=False,        # False = per-mass quantile bins (original behavior)
    param_edges=None,             # e.g., (0,150,300,600,1200) to force fixed Hα-EW bins
    ealpha=0.25, capsize=3
):
    """
    For each mass bin, bin 'param' and plot median R_e ± (16–84%) vs param.
    By default uses per-mass quantile bins (original behavior).
    """
    logM = np.asarray(logM); Re = np.asarray(Re_kpc); param = np.asarray(param)
    good = np.isfinite(logM) & np.isfinite(Re) & np.isfinite(param)
    logM, Re, param = logM[good], Re[good], param[good]

    n_massbins = len(mass_bins) - 1
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_massbins))
    plt.figure(figsize=(8.5,6))
    ax = plt.gca()

    # If you request global bins or explicit edges, prepare them once
    global_edges = None
    if param_edges is not None:
        global_edges = np.unique(np.asarray(param_edges, dtype=float))
    elif use_global_bins:
        global_edges = np.unique(np.quantile(param, np.linspace(0,1,nbins_param+1)))

    for i in range(n_massbins):
        m = (logM >= mass_bins[i]) & (logM < mass_bins[i+1])
        if m.sum() < min_per_bin:
            continue

        x = param[m]; y = Re[m]

        # Decide bin edges for this mass bin
        if global_edges is not None:
            edges = global_edges
        else:
            edges = np.unique(np.quantile(x, np.linspace(0,1,nbins_param+1)))

        if len(edges) < 3:
            continue

        xc, y50, ylo, yhi = [], [], [], []
        for j in range(len(edges)-1):
            sel = (x >= edges[j]) & (x < edges[j+1])
            if sel.sum() >= max(8, int(0.5*min_per_bin)):
                # x-position = median param in the bin (matches your original)
                xc.append(np.nanmedian(x[sel]))
                med = np.nanmedian(y[sel])
                p16 = np.nanpercentile(y[sel], 16)
                p84 = np.nanpercentile(y[sel], 84)
                y50.append(med); ylo.append(med - p16); yhi.append(p84 - med)

        if xc:
            xc   = np.array(xc)
            y50  = np.array(y50)
            yerr = np.vstack([np.array(ylo), np.array(yhi)])
            c = colors[i]
            plt.errorbar(
                xc, y50, yerr=yerr,
                fmt='o-', color=c, lw=1.6, ms=5,
                ecolor=c, elinewidth=1.0, capsize=capsize, alpha=0.95,  # line
            )
            # make the bars a bit softer without hiding the line
            plt.errorbar(
                xc, y50, yerr=yerr,
                fmt='none', ecolor=c, elinewidth=2.0, alpha=ealpha, capsize=0,
                label=f'log M* ∈ [{mass_bins[i]:.2f},{mass_bins[i+1]:.2f}] (n={m.sum()})'
            )

    if logy:
        plt.yscale('log')
        pretty_log_y_as_decimals(ax)
    if logx:
        plt.xscale('log')

    plt.xlabel(param_label)
    plt.ylabel('Median $R_\\mathrm{e}$ (kpc)')
    plt.grid(True, ls='--', alpha=0.4)
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()

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

    if radius_kpc_clean.size:
        print(f"Max effective radius after cleaning: {np.nanmax(radius_kpc_clean):.3f} kpc")
    else:
        print("No galaxies passed the cleaning mask – cannot compute max radius.")

    # Extreme PSB mask
    is_extreme_psb = (burstiness_clean <= 0.5) & (halpha_clean <= 25)

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
        mass_cut_applied=True, radius_kpc_err=radius_kpc_err[mass_cut_mask] 
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

    # same bins as your binned mass–radius plot
    bins = np.linspace(np.nanmin(stellar_mass_scaled), np.nanmax(stellar_mass_scaled), 16 + 1)

    ks_results = ks_size_by_mass_bin(
        stellar_mass_scaled,
        radius_kpc_clean,
        is_extreme_psb,
        bins=bins,              # or omit to let it choose nbins
        min_per_group=8,        # tweak per your sample sizes
        log_radius=True,        # matches how you view sizes
        fdr=True,
        plot=True,
        savefig="ks_by_mass_bin.png"
    )

    # quick text summary
    for r in ks_results:
        if np.isfinite(r['p']):
            # med_diff is in log10(kpc) if log_radius=True; convert to ratio for intuition
            ratio = 10**(r['med_diff']) if r['med_diff'] == r['med_diff'] else np.nan
            print(f"[{r['lo']:.2f},{r['hi']:.2f}] "
                f"n={r['n_other']}/{r['n_psb']}  KS={r['ks']:.3f}  p={r['p']:.3g}  "
                f"q={r['q']:.3g}  Δmedian(logR)={r['med_diff']:.3f} (~×{ratio:.2f})")
            
    # Choose mass bins you like:
    mass_bins = [8.1, 8.6, 9.0, 9.5]


    plot_size_vs_param_in_mass_bins(stellar_mass_scaled, radius_kpc_clean,
                                    table_bagpipes_matched['burstiness_50'][final_mask],
                                    'Burstiness (SFR recent/past)', logx = True,
                                    mass_bins=mass_bins, savefig='Re_vs_burstiness_by_mass.png')

    plot_size_vs_param_in_mass_bins(
    logM=stellar_mass_scaled,
    Re_kpc=radius_kpc_clean,
    param=halpha_clean,                      # your array
    param_label='Hα EW (Å)',
    mass_bins = (8.1, 8.6, 9.0, 9.5),
    logx=True, 
    nbins_param=5,
    use_global_bins=False,                # <- per-mass quantile bins (default)
    savefig='Re_vs_HaEW_errorbars.png'
)


    plot_size_vs_param_in_mass_bins(stellar_mass_scaled, radius_kpc_clean,
                                    table_bagpipes_matched['beta_C94_50'][final_mask],
                                    r'UV slope $\beta$',
                                    mass_bins=mass_bins, savefig='Re_vs_beta_by_mass.png')

if __name__ == "__main__":
    main(
        phot_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        bagpipes_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        galfit_fits="/raid/scratch/work/Griley/GALFIND_WORK/GALFIT/output/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/F444W/sersic/results.fits",
        filter_name='F444W',
        phot_idcol='NUMBER',
        bagpipes_idcol='#ID'
    )




