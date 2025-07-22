import astropy.units as u
from typing import Optional, Union, List
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'

from galfind.Data import morgan_version_to_dir
from galfind import Catalogue, EAZY, SED_code
from galfind import galfind_logger, Redshift_Bin_Selector, Redshift_Extractor, Multiple_Mask_Selector, Multiple_SED_fit_Selector, Min_Instrument_Unmasked_Band_Selector, Unmasked_Band_Selector, Bluewards_LyLim_Non_Detect_Selector, Bluewards_Lya_Non_Detect_Selector, Redwards_Lya_Detect_Selector, Chi_Sq_Lim_Selector, Chi_Sq_Diff_Selector, Robust_zPDF_Selector, Sextractor_Bands_Radius_Selector    
from galfind import Bagpipes

def main(
    survey: str,
    version: str,
    instrument_names: List[str],
    aper_diams: u.Quantity,
    forced_phot_band: Optional[Union[str, List[str]]],
    SED_fitter_arr: List[SED_code],
    sample_SED_fitter_arr: List[SED_code],

):
    sample = Austin25_sample

    cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        version_to_dir_dict = morgan_version_to_dir,
        #crops = sample
    )

    for SED_fitter in SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(cat, aper_diam, load_PDFs = True, load_SEDs = True, update = True)


#make sampler
    if sample is not None:
        sampler = sample(aper_diams[0], SED_fitter_arr[-1])
        new_cat = sampler(cat, return_copy = True)
        #redshift sampler
        #z_sampler = Redshift_Bin_Selector(aper_diams[0], SED_fitter_arr[-1], z_range)
       # z_sampler(cat, return_copy = True)

    for SED_fitter in sample_SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(new_cat, aper_diam, load_PDFs = True, load_SEDs = True, update = True)

#call this if already run main (the selection)
def main2(
    survey: str,
    version: str,
    instrument_names: List[str],
    aper_diams: u.Quantity,
    forced_phot_band: Optional[Union[str, List[str]]],
    SED_fitter_arr: List[SED_code],
    sample_SED_fitter_arr: List[SED_code],
    

):
    sample = Austin25_sample

    cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        version_to_dir_dict = morgan_version_to_dir,
        crops = sample(aper_diams[0], SED_fitter_arr[-1].label)
    )
    for SED_fitter in SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(cat, aper_diam, load_PDFs = True, load_SEDs = True, update = True)


#make sampler
    # if sample is not None:
    #     sampler = sample(aper_diams[0], SED_fitter_arr[-1])
    #     new_cat = sampler(cat, return_copy = True)
        #redshift sampler
        #z_sampler = Redshift_Bin_Selector(aper_diams[0], SED_fitter_arr[-1], z_range)
       # z_sampler(cat, return_copy = True)

    new_cat = cat
    for SED_fitter in sample_SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(new_cat, aper_diam, load_PDFs = True, load_SEDs = True, update = True, temp_label = 'temp')


class Austin25_unmasked_criteria(Multiple_Mask_Selector):

    def __init__(self):
        selectors = [
            Min_Instrument_Unmasked_Band_Selector(min_bands = 2, instrument = "ACS_WFC"),
            Min_Instrument_Unmasked_Band_Selector(min_bands = 6, instrument = "NIRCam"),
        ]
        selectors.extend([Unmasked_Band_Selector(band) for band in ["F090W", "F277W", "F356W", "F410M", "F444W"]])
        selection_name = "Austin+25_unmasked_criteria"
        super().__init__(selectors, selection_name = selection_name)
        

class Austin25_sample(Multiple_SED_fit_Selector):

    def __init__(
        self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        simulated: bool = False,
    ):
        selectors = [
            Bluewards_LyLim_Non_Detect_Selector(aper_diam, SED_fit_label, SNR_lim = 2.0, ignore_bands = ["F070W", "F850LP"]), #Ensures no detection shortward of the Lyman limit 
            Bluewards_Lya_Non_Detect_Selector(aper_diam, SED_fit_label, SNR_lim = 3.0, ignore_bands = ["F070W", "F850LP"]), #Ensures no detection shortward of the Lyman alpha line
            Redwards_Lya_Detect_Selector(aper_diam, SED_fit_label, SNR_lims = [8.0, 8.0], widebands_only = True, ignore_bands = ["F070W", "F850LP"]), # Requires a strong detection longward of Lyman-alpha, supporting high-redshift identification.
            Redwards_Lya_Detect_Selector(aper_diam, SED_fit_label, SNR_lims = 3.0, widebands_only = True, ignore_bands = ["F070W", "F850LP"]),
            Chi_Sq_Lim_Selector(aper_diam, SED_fit_label, chi_sq_lim = 3.0, reduced = True), #cuts objects with poor SED fits (high reduced χ²).
            Chi_Sq_Diff_Selector(aper_diam, SED_fit_label, chi_sq_diff = 4.0, dz = 0.5), #Ensures the best-fit redshift is significantly better than alternatives.
            Robust_zPDF_Selector(aper_diam, SED_fit_label, integral_lim = 0.6, dz_over_z = 0.1),
        ]
        assert isinstance(simulated, bool), galfind_logger.critical(f"{type(simulated)=}!=bool")
        if not simulated:
            selectors.extend([Sextractor_Bands_Radius_Selector(band_names = ["F277W", "F356W", "F444W"], gtr_or_less = "gtr", lim = 45. * u.marcsec)])
            # add unmasked instrument selections
            #selectors.extend([Unmasked_Instrument_Selector(instr_name) for instr_name in ["ACS_WFC", "NIRCam"]])
            selectors.extend([Austin25_unmasked_criteria()])
        selection_name = "Austin+25"
        super().__init__(aper_diam, SED_fit_label, selectors, selection_name = selection_name)

    def _assertions(self) -> bool:
        return True



if __name__ == "__main__":    
    survey = "JADES-DR3-GS-East"
    version = "v13"
    instrument_names = ["ACS_WFC", "NIRCam"]
    aper_diams = [0.32] * u.arcsec
    forced_phot_band = ["F277W", "F356W", "F444W"]
    SED_fitter_arr = [
        # EAZY({"templates": "fsps_larson", "lowz_zmax": 4.0}),
        # EAZY({"templates": "fsps_larson", "lowz_zmax": 6.0}),
        EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
    ]
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

    # main(
    #     survey,
    #     version,
    #     instrument_names,
    #     aper_diams,
    #     forced_phot_band,
    #     SED_fitter_arr,
    #     sample_SED_fitter_arr,
    # )

#call this if you have already ran main
    main2(
        survey,
        version,
        instrument_names,
        aper_diams,
        forced_phot_band,
        SED_fitter_arr,
        sample_SED_fitter_arr,
    )
