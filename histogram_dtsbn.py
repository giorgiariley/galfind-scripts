import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import numpy as np

# nicer defaults
plt.rcParams.update({
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

def load_fits_table(fits_path, hdu_index=1):
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[hdu_index].data)
    return table

def match_three_tables_by_id(table1, table2, table3, col1, col2, col3):
    ids1 = table1[col1].astype(str)
    ids2 = table2[col2].astype(str)
    ids3 = table3[col3].astype(str)
    common_ids = np.intersect1d(np.intersect1d(ids1, ids2), ids3)
    idx1 = np.nonzero(np.in1d(ids1, common_ids))[0]
    idx2 = np.nonzero(np.in1d(ids2, common_ids))[0]
    idx3 = np.nonzero(np.in1d(ids3, common_ids))[0]
    return table1[idx1], table2[idx2], table3[idx3]

def compute_flux_ratio(table, filter_name):
    ratio = table[f'FLUX_AUTO_{filter_name}'] / table[f'FLUX_APER_{filter_name}']
    return np.clip(ratio, 1, 10)

def scale_mass_sfr_log(table, logR):
    # in place - assumes stellar_mass_* are log10
    table['stellar_mass_50'] += logR
    if 'stellar_mass_16' in table.colnames: table['stellar_mass_16'] += logR
    if 'stellar_mass_84' in table.colnames: table['stellar_mass_84'] += logR
    return table

def make_psb_mask(table_bagpipes, halpha_col='Halpha_EW_rest_50', burst_col='burstiness_50',
                  halpha_thresh=25.0, burst_thresh=0.5, mass_col='stellar_mass_50'):
    halpha = table_bagpipes[halpha_col]
    burst = table_bagpipes[burst_col]
    mass  = table_bagpipes[mass_col]
    finite = np.isfinite(halpha) & np.isfinite(burst) & np.isfinite(mass)
    psb = finite & (halpha < halpha_thresh) & (burst < burst_thresh)
    return psb, finite

def _common_bins(x, nbins=30, step=0.2):
    # If mass is log10(M/Msun), 0.2 dex bins are a good default
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    # try to land on neat edges
    edges = np.arange(np.floor(xmin/step)*step, np.ceil(xmax/step)*step + step*0.5, step)
    if len(edges) < 5:  # fallback
        edges = np.linspace(xmin, xmax, nbins+1)
    return edges

def plot_mass_overlay(mass_all, mass_psb, outpath="mass_overlay.png"):
    bins = _common_bins(np.concatenate([mass_all, mass_psb]))
    plt.figure(figsize=(8,6), facecolor='white')
    # outline steps for visibility
    plt.hist(mass_all, bins=bins, color='Royalblue',histtype='step', linewidth=2, density=True, label='Other')
    plt.hist(mass_psb, bins=bins,color='tomato', histtype='step', linewidth=2, density=True, label='PSB - Hα EW < 25Å, burstiness < 0.5')
    # medians
    for arr, ls in [(mass_all, '--'), (mass_psb, ':')]:
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
    # common bins so densities are directly comparable
    bins = _common_bins(np.concatenate([mass_all, mass_psb]))

    # precompute densities to set a fair shared y-limit
    h_all, _ = np.histogram(mass_all[np.isfinite(mass_all)], bins=bins, density=True)
    h_psb, _ = np.histogram(mass_psb[np.isfinite(mass_psb)], bins=bins, density=True)
    ymax = 1.1 * np.nanmax([np.nanmax(h_all) if h_all.size else 0,
                             np.nanmax(h_psb) if h_psb.size else 0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    axes[0].hist(mass_all, bins=bins, color='royalblue', edgecolor='black', alpha=0.9, density=True)
    axes[0].axvline(np.nanmedian(mass_all), linestyle='--', alpha=0.7)
    axes[0].set_title('Other')

    axes[1].hist(mass_psb, bins=bins, color='tomato', edgecolor='black', alpha=0.9, density=True)
    axes[1].axvline(np.nanmedian(mass_psb), linestyle='--', alpha=0.7)
    axes[1].set_title('PSB - Hα EW < 25 Å, burstiness < 0.5')

    for ax in axes:
        ax.set_xlabel(r'log$_{10}(M_\star/M_\odot)$')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, ymax)

    # density units are per dex when binning in log10 mass
    axes[0].set_ylabel(r'Density [dex$^{-1}$]')

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved side-by-side mass comparison to: {outpath}")


def plot_burstiness_histogram(burstiness_values, output_path="burstiness_histogram.png"):
    plt.figure(figsize=(8, 6), facecolor='white')
    plt.hist(burstiness_values[np.isfinite(burstiness_values)], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Burstiness")
    plt.ylabel("Number of galaxies")
    plt.title("Histogram of Burstiness")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved histogram to: {output_path}")

def main(
    phot_fits, bagpipes_fits, galfit_fits, 
    filter_name='F444W', phot_idcol='NUMBER', bagpipes_idcol='#ID',
    halpha_col='Halpha_EW_rest_50', burst_col='burstiness_50', mass_col='stellar_mass_50'
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
    
    # Scale mass
    R = compute_flux_ratio(table_objects_matched, filter_name)
    logR = np.log10(R)
    scale_mass_sfr_log(table_bagpipes_matched, logR)

    # Extract quantities
    burstiness = table_bagpipes_matched[burst_col]
    mass = table_bagpipes_matched[mass_col]

    # Plot burstiness histogram for sanity
    plot_burstiness_histogram(burstiness)

    # PSB mask
    psb_mask, finite_mask = make_psb_mask(table_bagpipes_matched, halpha_col=halpha_col,
                                          burst_col=burst_col, mass_col=mass_col)

    mass_all = mass[~psb_mask]
    mass_psb = mass[psb_mask]

    # Three comparison views - pick your favorite
    plot_mass_overlay(mass_all, mass_psb, outpath="mass_overlay.png")
    plot_mass_small_multiples(mass_all, mass_psb, outpath="mass_side_by_side.png")

if __name__ == "__main__":
    main(
        phot_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        bagpipes_fits="/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits",
        galfit_fits="/raid/scratch/work/Griley/GALFIND_WORK/GALFIT/output/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/F444W/sersic/results.fits",
        filter_name='F444W',
        phot_idcol='NUMBER',
        bagpipes_idcol='#ID'
    )
