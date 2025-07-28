import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
import galfind
from galfind import Bagpipes, EAZY, Redshift_Extractor, Filter, Multiple_Filter 
import astropy.units as u

redshift = 6.
aper_diams= [0.32] * u.arcsec
bagpipes = Bagpipes(
            {
                "fix_z": redshift,
                "sfh": "continuity_bursty",
                "z_calculator":redshift,
                'sps_model': 'BPASS',
            }
        )

filt_names = ["F435W", "F606W", "F775W", "F814W", "F850LP",  "F090W", "F115W", "F150W", "F182M", "F200W", "F210M", "F277W", "F335M", "F356W", "F410M", "F444W", "F460M", "F480M"] #add the filters 
filterset = Multiple_Filter([Filter.from_filt_name(name) for name in filt_names])

bagpipes.extract_priors(filterset, redshift)