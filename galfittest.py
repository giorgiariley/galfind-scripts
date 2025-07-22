import astropy.units as u
from typing import Optional, Union, List
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'

from galfind.Data import morgan_version_to_dir
from galfind import Catalogue, EAZY, Filter, PSF_Cutout, Galfit_Fitter
from galfind import Redshift_Extractor
from galfind import Bagpipes
from run_cat import Austin25_sample

sample= Austin25_sample

# === Parameters ===
survey = "JADES-DR3-GS-East"
version = "v13"
instrument_names = ["ACS_WFC", "NIRCam"]
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"]
min_flux_pc_err = 10.

# === EAZY fits ===
SED_fitter_arr = [EAZY({"templates": "fsps_larson", "lowz_zmax": None})]

sample_SED_fitter_arr = [
        Bagpipes(
            {
                "fix_z": EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
                "sfh": "continuity_bursty",
                "z_calculator": Redshift_Extractor(aper_diams[0], EAZY({"templates": "fsps_larson", "lowz_zmax": None})),
                'sps_model': 'BPASS',
            }
        ),
    ]

cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        version_to_dir_dict = morgan_version_to_dir,
        crops = sample(aper_diams[0], SED_fitter_arr[-1].label)
)

cat.load_sextractor_Re()

# fit GALFIT
for band_name in ["F444W"]:
    filt = Filter.from_filt_name(band_name)
    psf_path = f"/nvme/scratch/work/westcottl/psf/PSF_Resample_03_{band_name}.fits"
    psf = PSF_Cutout.from_fits(
        fits_path=psf_path,
        filt=filt,
        unit="adu",
        pix_scale = 0.03 * u.arcsec,
        size = 1.5 * u.arcsec, #i have changed from 0.96 to 3.00 for morf
    )
    #breakpoint()
    galfit_sersic_fitter = Galfit_Fitter(psf, "sersic", fixed_params = ["n"])
    galfit_sersic_fitter(cat)