import astropy.units as u
from typing import Union
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
from galfind.Data import morgan_version_to_dir
from galfind import Catalogue, EAZY, SED_code
from galfind import galfind_logger, Multiple_Mask_Selector,Redshift_Extractor, Multiple_SED_fit_Selector, Min_Instrument_Unmasked_Band_Selector, Unmasked_Band_Selector, Bluewards_LyLim_Non_Detect_Selector, Bluewards_Lya_Non_Detect_Selector, Redwards_Lya_Detect_Selector, Chi_Sq_Lim_Selector, Chi_Sq_Diff_Selector, Robust_zPDF_Selector, Sextractor_Bands_Radius_Selector    
import matplotlib.pyplot as plt
import numpy as np
from galfind import Bagpipes
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
        # EAZY({"templates": "fsps_larson", "lowz_zmax": 4.0}),
        # EAZY({"templates": "fsps_larson", "lowz_zmax": 6.0}),
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
#load in the SEDs
for SED_fitter in sample_SED_fitter_arr:
    for aper_diam in aper_diams:
        try:
            SED_fitter(cat, aper_diam, load_PDFs=False, load_SEDs=True, update=True, temp_label = 'temp')
        except ZeroDivisionError as e:
            print(f"ZeroDivisionError while loading SEDs with {SED_fitter.label}: {e}")


# Create output folder
output_folder = "balmer_break_outputs1"
os.makedirs(output_folder, exist_ok=True)

# Store results
results = []

# Loop over all galaxies
for idx, galaxy in enumerate(cat):
    try:
        aper = aper_diams[0]
        SED_result = galaxy.aper_phot[aper].SED_results[sample_SED_fitter_arr[-1].label]
        sed_wavs_obs = SED_result.SED.wavs       
        z = SED_result.z
        sed_wavs = sed_wavs_obs / (1 + z)  # rest-frame wavelengths
        sed_fluxes = np.array(SED_result.SED.mags)
        # Skip if all fluxes are non-positive
        valid = sed_fluxes > 0
        if not np.any(valid):
            continue

        safe_fluxes = np.clip(sed_fluxes[valid], 1e-30, None)
        sed_mags = -2.5 * np.log10((safe_fluxes * 1e-9) / 3631)

        # Balmer break masks
        blue_mask = (sed_wavs[valid] >= 3400 * u.AA) & (sed_wavs[valid] <= 3600 * u.AA)
        red_mask  = (sed_wavs[valid] >= 4150 * u.AA) & (sed_wavs[valid] <= 4250 * u.AA)

        blue_median_mag = np.median(sed_mags[blue_mask])
        red_median_mag = np.median(sed_mags[red_mask])
        balmer_break_mag = blue_median_mag - red_median_mag
        results.append([idx, balmer_break_mag, z])
        # Plot rest-frame SED
        plt.figure(figsize=(8, 5))
        plt.plot([], [], ' ', label=f'Balmer Break = {balmer_break_mag:.2f} mag')
        plt.plot(sed_wavs[valid], sed_mags, label='Best-fit SED (Bagpipes)', lw=2)
        plt.xlabel("Rest-frame Wavelength [Å]")
        plt.xlim(500, 7000)
        plt.ylim(10, 40)
        plt.ylabel("AbMags")
        plt.gca().invert_yaxis()
        # Highlight 3400–3600 Å region (blue side)
        plt.axvspan(3400, 3600, color='blue', alpha=0.2, label='3400–3600 Å')
        # Highlight 4150–4250 Å region (red side)
        plt.axvspan(4150, 4250, color='red', alpha=0.2, label='4150–4250 Å')
        plt.title(f"Best-fit SED for galaxy {idx} using {sample_SED_fitter_arr[-1].label} at redshift {z:.2f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_folder, f"galaxy_{idx:04d}_sed_plot.png")
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Skipped galaxy {idx} due to error: {e}")
        continue

# Save all Balmer break values to text file
results_array = np.array(results, dtype=object)

# Check shape: must be 2D with 3 columns
if results_array.ndim == 2 and results_array.shape[1] == 3:
    np.savetxt(
        os.path.join(output_folder, "balmer_breaks1.txt"),
        results_array,
        header="Index    BalmerBreak(mag)    Redshift",
        fmt=["%-8d", "%.4f", "%.4f"]
    )
else:
    print(f"Unexpected shape for results_array: {results_array.shape}")










