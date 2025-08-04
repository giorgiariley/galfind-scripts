import numpy as np
from astropy.io import fits
from astropy.table import Table

def load_bagpipes_table(fits_path):
    """Load the Bagpipes output table from a FITS file."""
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[1].data)
    return table

def compute_flux_ratio(filter, table):
    """Computes the auto/aper flux ratio and clips extreme values."""
    ratio = table[f'Flux_auto_{filter}'] / table[f'Flux_apper_{filter}']
    ratio = np.clip(ratio, 1, 10)  # Avoid over- or under-correcting
    return ratio

def scale_log_mass_sfr(table, filter):
    """
    Scales log(stellar mass) and log(SFR) by log10(flux_auto / flux_aperture).
    
    Updates the table in-place.
    """
    R = compute_flux_ratio(filter, table)
    logR = np.log10(R)
    
    # Assumes original columns are log10 values
    table['stellar_mass_50'] += logR
    table['sfr_50'] += logR

    return table
