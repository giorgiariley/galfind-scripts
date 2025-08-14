#plotting sersic index against different paramters ie mass, radius, burstiness
#colour code by PSB and not
#ensure that tables are matched
#ensure mass gets scaled

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import os
import matplotlib.ticker as mticker

plt.rcParams.update({
    "axes.labelsize": 15,   # axis label font
    "xtick.labelsize": 13,  # tick label font
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

def pretty_log_x_as_decimals(ax, tick_candidates=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)):
    ax.set_xscale('log')
    xmin, xmax = ax.get_xlim()
    ticks = [t for t in tick_candidates if xmin <= t <= xmax]
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda val, pos: ('{:.3f}'.format(val)).rstrip('0').rstrip('.'))
    )
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2,10)*0.1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

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


def flag_unreliable_fits(radius_kpc, redshifts, sersic_index, chi2_red):
    """Returns a boolean mask of reliable fits (True = reliable)."""
    flag_bad_z = (redshifts <= 0.01) | (redshifts >= 16)
    flag_bad_radius = (radius_kpc <= 0.01) | (radius_kpc > 10)
    flag_bad_n = (sersic_index >= 10) | ~np.isfinite(sersic_index)
    flag_bad_chi2 = (chi2_red > 5) | ~np.isfinite(chi2_red)
    return ~(flag_bad_radius | flag_bad_n | flag_bad_chi2 | flag_bad_z)

def plot_sersic_vs_param(x, n, psb_mask, xlabel, outpath,
                         title=None, xlog=False, ylim=(0,10)):
    """
    Scatter plot of Sérsic index (y) vs arbitrary parameter (x),
    colour-coded by PSB selection.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # finite mask
    finite = np.isfinite(x) & np.isfinite(n)
    x = x[finite]
    n = n[finite]
    psb = psb_mask[finite]

    plt.figure(figsize=(8,6), facecolor='white')

    # Other
    plt.scatter(x[~psb], n[~psb], s=18, alpha=0.6, c='royalblue',
                label=f'Other (n={np.sum(~psb)})', edgecolors='none')
    # PSB
    plt.scatter(x[psb], n[psb], s=24, alpha=0.85, c='tomato',
                label=f'Extreme PSBs (burstiness ≤ 0.5 & Hα EW ≤ 25 Å) (n={np.sum(psb)})', edgecolors='none')

    plt.xlabel(xlabel)
    plt.ylabel('Sérsic index n')


    if xlog:
        pretty_log_x_as_decimals(plt.gca())


    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved: {outpath}")

def main(
    phot_fits, bagpipes_fits, galfit_fits, 
    filter_name='F444W', phot_idcol='NUMBER', bagpipes_idcol='#ID',     
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
    scale_mass_sfr_log(table_bagpipes_matched, logR)  # adds logR to stellar_mass_* columns

    # Redshifts and radii
    redshifts = np.asarray(table_bagpipes_matched['input_redshift'], float)
    radius_kpc, radius_kpc_err = get_radius_kpc(table_galfit_matched, redshifts)

    # Structural fit bits
    sersic_index = np.asarray(table_galfit_matched['n'], float)
    chi2_red = np.asarray(table_galfit_matched['red_chi2'], float)

    # Reliability mask (fixed arguments)
    reliable_mask = flag_unreliable_fits(radius_kpc, redshifts, sersic_index, chi2_red)

    # Additional basic quality cuts
    physical_mask = (radius_kpc > 0.05) & (radius_kpc < 15.0) & np.isfinite(radius_kpc)
    good_err_mask = np.isfinite(radius_kpc_err) & (radius_kpc_err > 0) & ((radius_kpc_err / radius_kpc) < 0.5)

    final_mask = reliable_mask & physical_mask & good_err_mask

    # Apply mask to arrays we need
    radius_kpc_clean = radius_kpc[final_mask]
    sersic_clean = sersic_index[final_mask]
    burstiness_clean = np.asarray(table_bagpipes_matched['burstiness_50'], float)[final_mask]
    halpha_clean = np.asarray(table_bagpipes_matched['Halpha_EW_rest_50'], float)[final_mask]
    mass_clean = np.asarray(table_bagpipes_matched['stellar_mass_50'], float)[final_mask]




    # PSB mask on the cleaned set
    is_psb = (burstiness_clean <= 0.5) & (halpha_clean <= 25)
    plot_sersic_vs_param(
        x=radius_kpc_clean,
        n=sersic_clean,
        psb_mask=is_psb,
        xlabel='Effective radius R_e [kpc]',
        outpath='sersic_plots/n_vs_radius.png',
        xlog=True,
        ylim=(0, 10)
    )

    plot_sersic_vs_param(
        x=mass_clean,
        n=sersic_clean,
        psb_mask=is_psb,
        xlabel=r'log$_{10}(M_\star/M_\odot)$',
        outpath='sersic_plots/n_vs_mass.png',
        xlog=False,
        ylim=(0, 10)
    )

    plot_sersic_vs_param(
        x=burstiness_clean,
        n=sersic_clean,
        psb_mask=is_psb,
        xlabel='Burstiness',
        outpath='sersic_plots/n_vs_burstiness.png',
        xlog=False,
        ylim=(0, 10)
    )

    # Now adjust axis range on the saved figure
    fig, ax = plt.subplots(figsize=(8,6), facecolor='white')

    finite = np.isfinite(burstiness_clean) & np.isfinite(sersic_clean)
    x = burstiness_clean[finite]
    n = sersic_clean[finite]
    psb = is_psb[finite]

    ax.scatter(x[~psb], n[~psb], s=18, alpha=0.6, c='royalblue',
               label=f'Other (n={np.sum(~psb)})', edgecolors='none')
    ax.scatter(x[psb], n[psb], s=24, alpha=0.85, c='tomato',
               label=f'PSB (n={np.sum(psb)})', edgecolors='none')

    ax.set_xlabel('Burstiness')
    ax.set_ylabel('Sérsic index n')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig('sersic_plots/n_vs_burstiness_zoom.png', dpi=200)
    plt.close(fig)
    print("Saved: sersic_plots/n_vs_burstiness_zoom.png")

if __name__ == "__main__":
    main(
        phot_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        bagpipes_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        galfit_fits="/raid/scratch/work/Griley/GALFIND_WORK/GALFIT/output/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/F444W/sersic/results.fits",
        filter_name='F444W',
        phot_idcol='NUMBER',
        bagpipes_idcol='#ID'
    )

