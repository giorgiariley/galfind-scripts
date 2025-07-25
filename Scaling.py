#making a function that scales mass/sfr
from astropy.io import fits
from astropy.table import Table


def load_bagpipes_table(fits_path):
    """Load the Bagpipes output table from a FITS file."""
    with fits.open(fits_path) as hdulist:
        table = Table(hdulist[1].data)
    return table

def ratio(filter, table):
    "takes ratio of auto and aper fluxes for scaling"
    R_filter =  table[f'Flux_auto_{filter}']/ table[f'Flux_apper_{filter}']
    if R_filter <1:
        R_filter = 1
    elif R_filter > 10:
        R_filter = 10       
    
    return R_filter



#need to think about log scale and create a new
def scale_mass_sfr(table, filter):
    """
    Scales the mass and SFR based on the ratio of auto and aperture fluxes.
    
    Parameters:
    - table: Astropy Table containing the Bagpipes output.
    - filter: The filter for which to scale the mass and SFR.
    
    Returns:
    - table: Updated Astropy Table with scaled mass and SFR.
    """
    R_filter = ratio(filter, table)
    
    # Scale mass and SFR
    table['stellar_mass_50'] *= R_filter
    table['sfr_50'] *= R_filter
    
    return table    
        

