import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import numpy as np

# nicer defaults
plt.rcParams.update({
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
})

def load_fits_table(fits_path, hdu_index=1):
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[hdu_index].data)
    return table

def match_three_tables_by_id(table1, table2, table3, col1, col2, col3):
    ids1 = table1[col1].astype(str)
    ids2 = table2[col2].astype(str)
    ids3 = table3[col3].astype(str)

    # intersection (we will filter by membership to preserve original order per table)
    common_ids = np.intersect1d(np.intersect1d(ids1, ids2), ids3)

    idx1 = np.nonzero(np.in1d(ids1, common_ids))[0]
    idx2 = np.nonzero(np.in1d(ids2, common_ids))[0]
    idx3 = np.nonzero(np.in1d(ids3, common_ids))[0]
    return table1[idx1], table2[idx2], table3[idx3]

def compute_flux_ratio(table, filter_name):
    """Computes the auto/aper flux ratio and clips extreme values."""
    ratio = table[f'FLUX_AUTO_{filter_name}'] / table[f'FLUX_APER_{filter_name}']
    return np.clip(ratio, 1, 10)

def scale_quantiles_in_place(table, logR, 
                             log_stems=('stellar_mass',),
                             linear_stems=('sfr', 'sfr_10myr'),
                             percentiles=('16', '50', '84')):
    """
    Apply aperture/total flux scaling:
      - For *logarithmic* quantities (e.g. stellar_mass in log10), add logR per row.
      - For *linear* quantities (e.g. SFR), multiply by R = 10**logR per row.

    Operates in-place; silently skips missing columns.
    """
    logR = np.asarray(logR, dtype=float)
    R = np.power(10.0, logR)

    # helper to broadcast 1D scale to column
    def _apply_add(colname, addvec):
        if colname in table.colnames:
            arr = np.asarray(table[colname], dtype=float)
            table[colname] = arr + addvec

    def _apply_mul(colname, mulvec):
        if colname in table.colnames:
            arr = np.asarray(table[colname], dtype=float)
            table[colname] = arr * mulvec

    for stem in log_stems:
        for p in percentiles:
            _apply_add(f'{stem}_{p}', logR)

    for stem in linear_stems:
        for p in percentiles:
            _apply_mul(f'{stem}_{p}', R)

    return table

def make_psb_mask(table_bagpipes, halpha_col='Halpha_EW_rest_50', burst_col='burstiness_50',
                  halpha_thresh=25.0, burst_thresh=0.5, mass_col='stellar_mass_50'):
    halpha = np.asarray(table_bagpipes[halpha_col], dtype=float) if halpha_col in table_bagpipes.colnames else np.full(len(table_bagpipes), np.nan)
    burst  = np.asarray(table_bagpipes[burst_col], dtype=float) if burst_col in table_bagpipes.colnames else np.full(len(table_bagpipes), np.nan)
    mass   = np.asarray(table_bagpipes[mass_col], dtype=float) if mass_col in table_bagpipes.colnames else np.full(len(table_bagpipes), np.nan)

    finite = np.isfinite(halpha) & np.isfinite(burst) & np.isfinite(mass)
    psb = finite & (halpha < halpha_thresh) & (burst < burst_thresh)
    return psb, finite

def _common_bins(x, nbins=30, step=0.2):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.linspace(0, 1, nbins+1)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    edges = np.arange(np.floor(xmin/step)*step, np.ceil(xmax/step)*step + step*0.5, step)
    if len(edges) < 5:
        edges = np.linspace(xmin, xmax, nbins+1)
    return edges

def plot_mass_overlay(mass_all, mass_psb, outpath="mass_overlay.png"):
    bins = _common_bins(np.concatenate([mass_all, mass_psb]))
    plt.figure(figsize=(8,6), facecolor='white')
    plt.hist(mass_all, bins=bins, color='royalblue', histtype='step', linewidth=2, density=True, label='Other')
    plt.hist(mass_psb, bins=bins, color='tomato', histtype='step', linewidth=2, density=True, label='PSB - Hα EW < 25 Å, burstiness < 0.5')
    for arr, ls in ((mass_all, '--'), (mass_psb, ':')):
        if np.isfinite(arr).any():
            med = np.nanmedian(arr)
            plt.axvline(med, linestyle=ls, alpha=0.7)
    plt.xlabel(r'log$_{10}(M_\star/M_\odot)$')
    plt.ylabel('Density')
    plt.ylim(0, 0.9)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved overlay mass comparison to: {outpath}")

def plot_mass_small_multiples(mass_all, mass_psb, outpath="mass_side_by_side.png"):
    bins = _common_bins(np.concatenate([mass_all, mass_psb]))
    h_all, _ = np.histogram(mass_all[np.isfinite(mass_all)], bins=bins, density=True)
    h_psb, _ = np.histogram(mass_psb[np.isfinite(mass_psb)], bins=bins, density=True)
    ymax = 1.1 * np.nanmax([np.nanmax(h_all) if h_all.size else 0, np.nanmax(h_psb) if h_psb.size else 0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].hist(mass_all, bins=bins, color='royalblue', edgecolor='black', alpha=0.9, density=True)
    axes[0].axvline(np.nanmedian(mass_all), linestyle='--', alpha=0.7)
    axes[0].set_title('Other')

    axes[1].hist(mass_psb, bins=bins, color='tomato', edgecolor='black', alpha=0.9, density=True)
    axes[1].axvline(np.nanmedian(mass_psb), linestyle='--', alpha=0.7)
    axes[1].set_title('PSB - Hα EW < 25 Å, burstiness < 0.5')

    for ax in axes:
        ax.set_xlabel(r'log$_{10}(M_\star/M_\odot)$')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, ymax)

    axes[0].set_ylabel(r'Density [dex$^{-1}$]')
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved side-by-side mass comparison to: {outpath}")

def plot_burstiness_histogram(burstiness_values, output_path="burstiness_histogram.png"):
    plt.figure(figsize=(8, 6), facecolor='white')
    plt.hist(np.asarray(burstiness_values)[np.isfinite(burstiness_values)], bins=30,
             color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Burstiness")
    plt.ylabel("Number of galaxies")
    plt.title("Histogram of Burstiness")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved histogram to: {output_path}")

def plot_sfr_vs_mass(table_bagpipes, psb_mask, sfr_col='sfr_50', mass_col='stellar_mass_50',
                     outpath="sfr_vs_mass.png", ylog=False):
    """
    Plot SFR (linear quantity) vs stellar mass (log10) for PSB and others.
    """
    mass = np.asarray(table_bagpipes[mass_col], dtype=float)
    sfr  = np.asarray(table_bagpipes[sfr_col], dtype=float)

    finite_mask = np.isfinite(mass) & np.isfinite(sfr)

    mass_all = mass[finite_mask & ~psb_mask]
    sfr_all  = sfr[finite_mask & ~psb_mask]
    mass_psb = mass[finite_mask & psb_mask]
    sfr_psb  = sfr[finite_mask & psb_mask]

    plt.figure(figsize=(10, 8), facecolor='white')
    plt.scatter(mass_all, sfr_all, c='royalblue', alpha=0.6, s=20, label=f'Other (n={len(mass_all)})')
    plt.scatter(mass_psb, sfr_psb, c='tomato', alpha=0.8, s=30, label=f'PSB - Hα EW < 25 Å, burstiness < 0.5 (n={len(mass_psb)})')

    plt.xlabel(r'log$_{10}(M_\star/M_\odot)$')

    if 'sfr_10myr' in sfr_col.lower():
        plt.ylabel(r'SFR$_{10\,\mathrm{Myr}}$ [M$_\odot$ yr$^{-1}$]')
    else:
        plt.ylabel(r'SFR [M$_\odot$ yr$^{-1}$]')

    if ylog:
        plt.yscale('log')

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved SFR vs mass plot to: {outpath}")

def main(
    phot_fits, bagpipes_fits, galfit_fits,
    filter_name='F444W', phot_idcol='NUMBER', bagpipes_idcol='#ID',
    halpha_col='Halpha_EW_rest_50', burst_col='burstiness_50', mass_col='stellar_mass_50'
):
    # Load tables
    table_objects = load_fits_table(phot_fits, hdu_index=1)
    # Bagpipes table is in HDU 4 per your file
    with fits.open(bagpipes_fits) as hdulist:
        table_bagpipes = Table(hdulist[4].data)
    table_galfit = load_fits_table(galfit_fits, hdu_index=1)

    # Match tables by ID
    table_objects_m, table_bagpipes_m, table_galfit_m = match_three_tables_by_id(
        table_objects, table_bagpipes, table_galfit,
        phot_idcol, bagpipes_idcol, 'id'
    )

    # Scale: log mass add logR; linear SFR multiply by R
    R = compute_flux_ratio(table_objects_m, filter_name)
    logR = np.log10(R)
    scale_quantiles_in_place(table_bagpipes_m, logR)

    # Extract and quick sanity plot
    burstiness = np.asarray(table_bagpipes_m[burst_col], dtype=float)
    mass = np.asarray(table_bagpipes_m[mass_col], dtype=float)
    plot_burstiness_histogram(burstiness)

    # PSB mask
    psb_mask, _ = make_psb_mask(table_bagpipes_m, halpha_col=halpha_col,
                                 burst_col=burst_col, mass_col=mass_col)

    # Mass distributions
    plot_mass_overlay(mass[~psb_mask], mass[psb_mask], outpath="mass_overlay.png")
    plot_mass_small_multiples(mass[~psb_mask], mass[psb_mask], outpath="mass_side_by_side.png")

    # SFR vs Mass
    plot_sfr_vs_mass(table_bagpipes_m, psb_mask, sfr_col='sfr_50',
                     mass_col=mass_col, outpath="sfr_vs_mass.png", ylog=True)

    if 'sfr_10myr_50' in table_bagpipes_m.colnames:
        plot_sfr_vs_mass(table_bagpipes_m, psb_mask, sfr_col='sfr_10myr_50',
                         mass_col=mass_col, outpath="sfr_10myr_vs_mass.png", ylog=True)

if __name__ == "__main__":
    main(
        phot_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        bagpipes_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        galfit_fits="/raid/scratch/work/Griley/GALFIND_WORK/GALFIT/output/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/F444W/sersic/results.fits",
        filter_name='F444W',
        phot_idcol='NUMBER',
        bagpipes_idcol='#ID'
    )
