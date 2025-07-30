import astropy.units as u
from typing import List
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
import numpy as np
import matplotlib.pyplot as plt
from galfind import (Catalogue, EAZY,
    useful_funcs_austind as funcs
)
from galfind.Data import morgan_version_to_dir


#load in catalogue and parameters 
def main2():

    # variables 
    survey = "JADES-DR3-GS-East"
    version = "v13"
    instrument_names = ["ACS_WFC", "NIRCam"]
    aper_diams = [0.32] * u.arcsec
    forced_phot_band = ["F277W", "F356W", "F444W"]
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
    )
    for SED_fitter in SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(cat, aper_diam, load_PDFs = False, load_SEDs = True, update = True)



    return cat, aper_diams[0], SED_fitter_arr


#calculate the balmer break for one galaxy and plot its SED with the windows
def process_galaxy(galaxy, idx: int, aper, sed_fitter, output_folder: str) -> List:
    wav_units = u.um
    mag_units = u.ABmag

    SED_result_EZ = galaxy.aper_phot[aper].SED_results[sed_fitter.label]
    sed_wavs_obs_EZ = funcs.convert_wav_units(SED_result_EZ.SED.wavs, wav_units)
    z_EZ = SED_result_EZ.z
    sed_wavs_EZ= sed_wavs_obs_EZ / (1 + z_EZ)
    sed_fluxes_EZ = np.where(SED_result_EZ.SED.mags <= 0, 1e-10 * SED_result_EZ.SED.mags.unit, SED_result_EZ.SED.mags)
    if np.all(sed_fluxes_EZ <= 0):
        return None
    sed_mags_EZ = funcs.convert_mag_units(SED_result_EZ.SED.wavs, sed_fluxes_EZ, mag_units)

    blue_mask_EZ = (sed_wavs_EZ >= 0.3400 * u.um) & (sed_wavs_EZ <= 0.3600 * u.um)
    red_mask_EZ  = (sed_wavs_EZ >= 0.4150 * u.um) & (sed_wavs_EZ <= 0.4250 * u.um)
    blue_median_mag_EZ = np.median(sed_mags_EZ[blue_mask_EZ])
    red_median_mag_EZ = np.median(sed_mags_EZ[red_mask_EZ])
    balmer_break_mag_EZ = blue_median_mag_EZ - red_median_mag_EZ

    plot_path = os.path.join(output_folder, 'SED_plots', f"galaxy_{idx:04d}_sed.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot([], [], ' ', label=f'Balmer Break = {balmer_break_mag_EZ:.2f}')
    plt.xlim(0, 0.7)
    plt.ylim(23, 32)
    plt.gca().invert_yaxis()
    plt.axvspan(0.3400, 0.3600, color='blue', alpha=0.2, label='3400–3600 Å')
    plt.axvspan(0.4150, 0.4250, color='red', alpha=0.2, label='4150–4250 Å')
    plt.xlabel("Rest-frame Wavelength (μm)")
    plt.ylabel("AB Mags")
    plt.title(f"Best-fit SED for galaxy {idx} using EAZY at redshift {z_EZ:.2f}")
    plt.legend(); plt.tight_layout()
    plt.grid(alpha=0.3); plt.savefig(plot_path); plt.close()

    return [idx, balmer_break_mag_EZ.value,  SED_result_EZ.z.value]


#running for every galaxy in a catalogue and saving the balmer break magnitudes 
def save_balmer():
    output_folder = "EAZY_Balmer_output"
    os.makedirs(output_folder, exist_ok=True)

    cat, aper, SED_fitter_arr = main2()

    results = []
    for idx, galaxy in enumerate(cat):
        res = process_galaxy(galaxy, idx, aper, SED_fitter_arr[-1], output_folder)
        if res:
            results.append(res)

    results_array = np.array(results)
    np.savetxt(
        os.path.join(output_folder, "All_balmer_breaks.txt"),
        results_array,
        header="Index   EAZYBalmerbreak(mag)    Redshift",
        fmt=["%-8d", "%.4f", "%.4f"]
    )


#run
if __name__ == "__main__":
    save_balmer()
