import astropy.units as u
from typing import Union, List, Optional
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
import numpy as np
import matplotlib.pyplot as plt

from galfind import (
    Catalogue, EAZY, Bagpipes, SED_code,
    Redshift_Extractor,
    Multiple_Mask_Selector, Multiple_SED_fit_Selector,
    Min_Instrument_Unmasked_Band_Selector, Unmasked_Band_Selector,
    Bluewards_LyLim_Non_Detect_Selector, Bluewards_Lya_Non_Detect_Selector,
    Redwards_Lya_Detect_Selector, Chi_Sq_Lim_Selector, Chi_Sq_Diff_Selector,
    Robust_zPDF_Selector, Sextractor_Bands_Radius_Selector,
    useful_funcs_austind as funcs
)
from galfind.Data import morgan_version_to_dir

#sample selection
class Austin25_unmasked_criteria(Multiple_Mask_Selector):
    def __init__(self):
        selectors = [
            Min_Instrument_Unmasked_Band_Selector(min_bands=2, instrument="ACS_WFC"),
            Min_Instrument_Unmasked_Band_Selector(min_bands=6, instrument="NIRCam"),
        ]
        selectors.extend([Unmasked_Band_Selector(band) for band in ["F090W", "F277W", "F356W", "F410M", "F444W"]])
        super().__init__(selectors, selection_name="Austin+25_unmasked_criteria")


class Austin25_sample(Multiple_SED_fit_Selector):
    def __init__(self, aper_diam: u.Quantity, SED_fit_label: Union[str, SED_code], simulated: bool = False):
        selectors = [
            Bluewards_LyLim_Non_Detect_Selector(aper_diam, SED_fit_label, SNR_lim=2.0, ignore_bands=["F070W", "F850LP"]),
            Bluewards_Lya_Non_Detect_Selector(aper_diam, SED_fit_label, SNR_lim=3.0, ignore_bands=["F070W", "F850LP"]),
            Redwards_Lya_Detect_Selector(aper_diam, SED_fit_label, SNR_lims=[8.0, 8.0], widebands_only=True, ignore_bands=["F070W", "F850LP"]),
            Redwards_Lya_Detect_Selector(aper_diam, SED_fit_label, SNR_lims=3.0, widebands_only=True, ignore_bands=["F070W", "F850LP"]),
            Chi_Sq_Lim_Selector(aper_diam, SED_fit_label, chi_sq_lim=3.0, reduced=True),
            Chi_Sq_Diff_Selector(aper_diam, SED_fit_label, chi_sq_diff=4.0, dz=0.5),
            Robust_zPDF_Selector(aper_diam, SED_fit_label, integral_lim=0.6, dz_over_z=0.1),
        ]
        if not simulated:
            selectors.extend([
                Sextractor_Bands_Radius_Selector(band_names=["F277W", "F356W", "F444W"], gtr_or_less="gtr", lim=45. * u.marcsec),
                Austin25_unmasked_criteria()
            ])
        super().__init__(aper_diam, SED_fit_label, selectors, selection_name="Austin+25")

    def _assertions(self) -> bool:
        return True

#load in catalogue and parameters 
def main2():
    sample = Austin25_sample

    # variables 
    survey = "JADES-DR3-GS-East"
    version = "v13"
    instrument_names = ["ACS_WFC", "NIRCam"]
    aper_diams = [0.32] * u.arcsec
    forced_phot_band = ["F277W", "F356W", "F444W"]
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
    SED_fitter_arr = [
            EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
    for SED_fitter in SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(cat, aper_diam, load_PDFs = False, load_SEDs = True, update = True)

    # if sample is not None:
    #     sampler = sample(aper_diams[0], SED_fitter_arr[-1])
    #     new_cat = sampler(cat, return_copy = True)

    for SED_fitter in sample_SED_fitter_arr:
        for aper_diam in aper_diams:
                SED_fitter(cat, aper_diam, load_PDFs=False, load_SEDs=True, update=True)

    return cat, aper_diams[0], sample_SED_fitter_arr, SED_fitter_arr


#calculate the balmer break for one galaxy and plot its SED with the windows
def process_galaxy(galaxy, idx: int, aper, sample_fitter, sed_fitter, output_folder: str) -> List:
    wav_units = u.um
    mag_units = u.ABmag

    SED_result_pipes = galaxy.aper_phot[aper].SED_results[sample_fitter.label]
    sed_wavs_obs_pipes = funcs.convert_wav_units(SED_result_pipes.SED.wavs, wav_units)
    z_pipes = SED_result_pipes.z
    sed_wavs_pipes = sed_wavs_obs_pipes / (1 + z_pipes)
    sed_fluxes_pipes = np.where(SED_result_pipes.SED.mags <= 0, 1e-10 * SED_result_pipes.SED.mags.unit, SED_result_pipes.SED.mags)
    if np.all(sed_fluxes_pipes <= 0):
        return None
    sed_mags_pipes = funcs.convert_mag_units(SED_result_pipes.SED.wavs, sed_fluxes_pipes, mag_units)

    SED_result_EZ = galaxy.aper_phot[aper].SED_results[sed_fitter.label]
    sed_wavs_obs_EZ = funcs.convert_wav_units(SED_result_EZ.SED.wavs, wav_units)
    z_EZ = SED_result_EZ.z
    sed_wavs_EZ= sed_wavs_obs_EZ / (1 + z_EZ)
    sed_fluxes_EZ = np.where(SED_result_EZ.SED.mags <= 0, 1e-10 * SED_result_EZ.SED.mags.unit, SED_result_EZ.SED.mags)
    if np.all(sed_fluxes_EZ <= 0):
        return None
    sed_mags_EZ = funcs.convert_mag_units(SED_result_EZ.SED.wavs, sed_fluxes_EZ, mag_units)

    blue_mask = (sed_wavs_pipes >= 0.3400 * u.um) & (sed_wavs_pipes <= 0.3600 * u.um)
    red_mask  = (sed_wavs_pipes >= 0.4150 * u.um) & (sed_wavs_pipes <= 0.4250 * u.um)
    blue_median_mag = np.median(sed_mags_pipes[blue_mask])
    red_median_mag = np.median(sed_mags_pipes[red_mask])
    balmer_break_mag_pipes = blue_median_mag - red_median_mag
    blue_mask_EZ = (sed_wavs_EZ >= 0.3400 * u.um) & (sed_wavs_EZ <= 0.3600 * u.um)
    red_mask_EZ  = (sed_wavs_EZ >= 0.4150 * u.um) & (sed_wavs_EZ <= 0.4250 * u.um)
    blue_median_mag_EZ = np.median(sed_mags_EZ[blue_mask_EZ])
    red_median_mag_EZ = np.median(sed_mags_EZ[red_mask_EZ])
    balmer_break_mag_EZ = blue_median_mag_EZ - red_median_mag_EZ

    plot_path = os.path.join(output_folder, 'SED_plots', f"galaxy_{idx:04d}_sed.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot([], [], ' ', label=f'Balmer Break = {balmer_break_mag_pipes:.2f}')
    plt.plot(sed_wavs_pipes, sed_mags_pipes, label='Bagpipes SED', lw=2)
    plt.xlim(0, 0.7)
    plt.ylim(23, 32)
    plt.gca().invert_yaxis()
    plt.axvspan(0.3400, 0.3600, color='blue', alpha=0.2, label='3400–3600 Å')
    plt.axvspan(0.4150, 0.4250, color='red', alpha=0.2, label='4150–4250 Å')
    plt.xlabel("Rest-frame Wavelength (μm)")
    plt.ylabel("AB Mags")
    plt.title(f"Best-fit SED for galaxy {idx} using Bagpipes at redshift {z_pipes:.2f}")
    plt.legend(); plt.tight_layout()
    plt.grid(alpha=0.3); plt.savefig(plot_path); plt.close()

    return [idx, str(galaxy.ID), balmer_break_mag_pipes.value, balmer_break_mag_EZ.value, SED_result_pipes.z.value]


#running for every galaxy in a catalogue and saving the balmer break magnitudes 
def save_balmer():
    output_folder = "Balmer_output"
    os.makedirs(output_folder, exist_ok=True)

    cat, aper, sample_SED_fitter_arr, SED_fitter_arr = main2()

    results = []
    for idx, galaxy in enumerate(cat):
        res = process_galaxy(galaxy, idx, aper, sample_SED_fitter_arr[-1], SED_fitter_arr[-1], output_folder)
        if res:
            results.append(res)

    results_array = np.array(results, dtype=object)
    np.savetxt(
        os.path.join(output_folder, "balmer_breaks2.txt"),
        results_array,
        header="Index    ID    BagpipesBalmerBreak(mag)   EAZYBalmerbreak(mag)    Redshift",
        fmt=["%-8d", "%s", "%.4f", "%.4f", "%.4f"]
    )


#run
if __name__ == "__main__":
    save_balmer()
