#overplotting EAZy and Bagpipes SEDs
import astropy.units as u
from typing import Union
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
from galfind.Data import morgan_version_to_dir
from galfind import Catalogue, EAZY, SED_code
from galfind import galfind_logger, Multiple_Mask_Selector,Redshift_Extractor, Multiple_SED_fit_Selector, Min_Instrument_Unmasked_Band_Selector, Unmasked_Band_Selector, Bluewards_LyLim_Non_Detect_Selector, Bluewards_Lya_Non_Detect_Selector, Redwards_Lya_Detect_Selector, Chi_Sq_Lim_Selector, Chi_Sq_Diff_Selector, Robust_zPDF_Selector, Sextractor_Bands_Radius_Selector    
import matplotlib.pyplot as plt
import numpy as np
from galfind import Bagpipes, useful_funcs_austind as funcs


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
    #crops = sample(aper_diams[0], SED_fitter_arr[-1].label)

)
for SED_fitter in SED_fitter_arr:
    for aper_diam in aper_diams:
        SED_fitter(cat, aper_diam, load_PDFs = False, load_SEDs = True, update = True)


#load in the SEDs
for SED_fitter in sample_SED_fitter_arr:
    for aper_diam in aper_diams:
        SED_fitter(cat, aper_diam, load_PDFs=False, load_SEDs=True, update=True, temp_label='temp')


output_folder = "overplot_output"
os.makedirs(output_folder, exist_ok=True)

# Store results
results = []
results1 = []

# Loop over all galaxies
for idx, galaxy in enumerate(cat):
    wav_units = u.um
    mag_units = u.ABmag
    aper = aper_diams[0]
    SED_result_pipes = galaxy.aper_phot[aper].SED_results[sample_SED_fitter_arr[-1].label]
    sed_wavs_obs_pipes = funcs.convert_wav_units(SED_result_pipes.SED.wavs, wav_units)
    z_pipes = SED_result_pipes.z
    sed_wavs_pipes = sed_wavs_obs_pipes / (1 + z_pipes)  # rest-frame wavelengths
    sed_fluxes_pipes = SED_result_pipes.SED.mags
    sed_fluxes_pipes = np.where(sed_fluxes_pipes <= 0, 1e-10 * sed_fluxes_pipes.unit, sed_fluxes_pipes)

    # Check if *all* values are <= 0
    if np.all(sed_fluxes_pipes <= 0):
        print(f"Skipped galaxy {idx} due to all non-positive fluxes in Bagpipes SED")
        continue

    sed_mags_pipes = funcs.convert_mag_units(
        SED_result_pipes.SED.wavs,
        sed_fluxes_pipes,
        mag_units
    )
    SED_result_EZ = galaxy.aper_phot[aper].SED_results[SED_fitter_arr[-1].label]
    sed_wavs_obs_EZ = funcs.convert_wav_units(SED_result_EZ.SED.wavs, wav_units)
    z_EZ = SED_result_EZ.z
    sed_wavs_EZ = sed_wavs_obs_EZ / (1 + z_EZ)  # rest-frame wavelengths
    sed_fluxes_EZ = SED_result_EZ.SED.mags
    sed_fluxes_EZ = np.where(sed_fluxes_EZ <= 0, 1e-10 * sed_fluxes_EZ.unit, sed_fluxes_EZ)
    
    # Check if *all* values are <= 0
    if np.all(sed_fluxes_EZ <= 0):
        print(f"Skipped galaxy {idx} due to all non-positive fluxes in Bagpipes SED")
        continue

    sed_mags_EZ = funcs.convert_mag_units(
        SED_result_EZ.SED.wavs,
        sed_fluxes_EZ,
        mag_units
    )
    # Balmer break masks
    blue_mask = (sed_wavs_pipes >= 0.3400 * u.um) & (sed_wavs_pipes <= 0.3600 * u.um)
    red_mask  = (sed_wavs_pipes >= 0.4150 * u.um) & (sed_wavs_pipes <= 0.4250 * u.um)
    blue_median_mag = np.median(sed_mags_pipes[blue_mask])
    red_median_mag = np.median(sed_mags_pipes[red_mask])
    balmer_break_mag = blue_median_mag - red_median_mag
    results.append([idx, balmer_break_mag, z_pipes])
    blue_mask_EZ = (sed_wavs_EZ >= 0.3400 * u.um) & (sed_wavs_EZ <= 0.3600 * u.um)
    red_mask_EZ  = (sed_wavs_EZ >= 0.4150 * u.um) & (sed_wavs_EZ <= 0.4250 * u.um)
    blue_median_mag_EZ = np.median(sed_mags_EZ[blue_mask_EZ])
    red_median_mag_EZ = np.median(sed_mags_EZ[red_mask_EZ])
    balmer_break_mag_EZ = blue_median_mag_EZ - red_median_mag_EZ
    results1.append([idx, balmer_break_mag_EZ, z_pipes])
    # # Plot rest-frame SED
    # plt.figure(figsize=(8, 5))
    # plt.plot([], [], ' ', label=f'Balmer Break = {balmer_break_mag:.2f}')
    # plt.plot(sed_wavs_pipes, sed_mags_pipes, label='Best-fit SED (Bagpipes)', lw=2)
    # plt.plot(sed_wavs_EZ, sed_mags_EZ, label='Best-fit SED (EAZY)', lw=2, linestyle='--')
    # plt.xlabel("Rest-frame Wavelength (μm)")
    # plt.xlim(0, 0.7)
    # plt.ylim(23, 32)
    # plt.ylabel("AbMags")
    # plt.gca().invert_yaxis()
    # plt.axvspan(0.3400, 0.3600, color='blue', alpha=0.2, label='3400–3600 Å')
    # plt.axvspan(0.4150, 0.4250, color='red', alpha=0.2, label='4150–4250 Å')
    # plt.title(f"Best-fit SED for galaxy {idx} using Bagpipes at redshift {z_pipes:.2f}")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plot_path = os.path.join(output_folder, 'SED_plots', f"galaxy_{idx:04d}_overplot.png")
    # plt.savefig(plot_path)
    # plt.close()

# Ensure results is a list of lists with consistent data types
results = [
    [int(row[0]), float(row[1].value), float(row[2].value)]  # Extract numerical values
    for row in results
]

# Convert to a NumPy array
results_array = np.array(results)
# Ensure the file is overwritten
file_path = os.path.join(output_folder, "balmer_breaks_allpipes.txt")
if os.path.exists(file_path):
    os.remove(file_path)

# Save the results to the text file
np.savetxt(
    file_path,
    results_array,
    header="Index    BalmerBreak(mag)    Redshift",
    fmt=["%-8d", "%.4f", "%.4f"]
)

# Ensure results is a list of lists with consistent data types
results1 = [
    [int(row[0]), float(row[1].value), float(row[2].value)]  # Extract numerical values
    for row in results1
]

# Convert to a NumPy array
results_array1 = np.array(results1)
# Ensure the file is overwritten
file_path = os.path.join(output_folder, "balmer_breaksa_allEZ.txt")
if os.path.exists(file_path):
    os.remove(file_path)

# Save the results to the text file
np.savetxt(
    file_path,
    results_array1,
    header="Index    BalmerBreak(mag)    Redshift",
    fmt=["%-8d", "%.4f", "%.4f"]
)
