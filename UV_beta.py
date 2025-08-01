import astropy.units as u
from typing import Union
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
from galfind.Data import morgan_version_to_dir
from galfind import Catalogue, EAZY, SED_code
import matplotlib.pyplot as plt
import numpy as np
from run_cat import Austin25_sample
from galfind import UV_Beta_Calculator


sample = Austin25_sample
# variables 
survey = "JADES-DR3-GS-East"
version = "v13"
instrument_names = ["ACS_WFC", "NIRCam"]
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"]
SED_fitter_arr = [
    # EAZY({"templates": "fsps_larson", "lowz_zmax": 4.0}),
    # EAZY({"templates": "fsps_larson", "lowz_zmax": 6.0}),
    EAZY({"templates": "fsps_larson", "lowz_zmax": None})
    ,
 ]

#load in the catalogue
cat = Catalogue.pipeline(
    survey,
    version,
    instrument_names=instrument_names,
    aper_diams=aper_diams,
    forced_phot_band=forced_phot_band,
    version_to_dir_dict=morgan_version_to_dir, 
    crops = sample(aper_diams[0], SED_fitter_arr[-1].label)
 
)


# beta_calculator = UV_Beta_Calculator(
#         aper_diam = aper_diams[0],
#         SED_fitter.label
#     )


#they have in their example
SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]
for SED_fit_params in SED_fit_params_arr:
    EAZY_fitter = EAZY(SED_fit_params)
    EAZY_fitter(cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)
SED_fit_label = EAZY_fitter.label

beta_calculator = UV_Beta_Calculator(
        aper_diam = aper_diams[0],
        SED_fit_label = SED_fit_label,
    )



beta_calculator(cat)