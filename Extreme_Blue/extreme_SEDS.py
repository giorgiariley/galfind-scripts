# This script plots SEDs for extreme galaxies from the Austin+25 sample.

from astropy.table import Table
import numpy as np
import astropy.units as u
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'

from galfind.Data import morgan_version_to_dir
from galfind import Catalogue, EAZY, Redshift_Extractor, useful_funcs_austind as funcs
from galfind import galfind_logger, Multiple_Mask_Selector, Redshift_Extractor, Multiple_SED_fit_Selector
from galfind import Min_Instrument_Unmasked_Band_Selector, Unmasked_Band_Selector
from galfind import Bluewards_LyLim_Non_Detect_Selector, Bluewards_Lya_Non_Detect_Selector
from galfind import Redwards_Lya_Detect_Selector, Chi_Sq_Lim_Selector, Chi_Sq_Diff_Selector
from galfind import Robust_zPDF_Selector, Sextractor_Bands_Radius_Selector
from galfind import Bagpipes, SED_code
import matplotlib.pyplot as plt
from typing import Union
from astropy.io import fits

# -------------- Selection  --------------

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

# -------------- Load Data --------------

#bagpipes
hdulist = fits.open("/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits")
table_bagpipes = Table(hdulist[1].data)
hdulist.close()
#balmer breaks
data = np.loadtxt('/nvme/scratch/work/Griley/galfind_scripts/Balmerbreak/Balmer_output/balmer_breaks2.txt', skiprows=1)
#UV Beta
hdulist2 = fits.open('/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits')
table_beta = Table(hdulist2[5].data)
print(table_beta.colnames)


# Load in variables
balmer_ids = data[:, 1]
balmer_breaks = data[:, 2]        

UV_beta = table_beta['beta_[1250,3000]AA_0.32as']  
log_xi_ion = np.log10(table_bagpipes['xi_ion_caseB_rest_50'])

# Skip the header (assume first row is '#ID')
bagpipes_ids = table_bagpipes['#ID'][1:]  # all except header

# Convert bagpipes IDs to string and strip spaces just in case
bagpipes_ids_str = [str(id_).strip() for id_ in bagpipes_ids]

# Convert balmer_ids floats to int then to string (assuming IDs are integers)
balmer_ids_str = [str(int(id_)) for id_ in balmer_ids]

matched_balmer_breaks = []
matched_UV_beta = []
matched_log_xi_ion = []
matched_bagpipes_indices = []

for i, balmer_id in enumerate(balmer_ids_str):
    # Find matching bagpipes ID index
    bagpipes_match = [j for j, bid in enumerate(bagpipes_ids_str) if bid == balmer_id]

    if len(bagpipes_match) > 0:
        bagpipes_idx = bagpipes_match[0]
        matched_balmer_breaks.append(balmer_breaks[i])
        matched_UV_beta.append(UV_beta[bagpipes_idx])
        matched_log_xi_ion.append(log_xi_ion[bagpipes_idx])
        matched_bagpipes_indices.append(bagpipes_idx)

# Convert lists to arrays
matched_balmer_breaks = np.array(matched_balmer_breaks)
matched_UV_beta = np.array(matched_UV_beta)
matched_log_xi_ion = np.array(matched_log_xi_ion)
matched_bagpipes_indices = np.array(matched_bagpipes_indices)

print(f"Successfully matched galaxies: {len(matched_bagpipes_indices)}")

# Apply selection criteria on matched samples
mask = (matched_balmer_breaks <= 0.35) & (matched_log_xi_ion <= 24.6) & (matched_UV_beta <= -2.6)
selected_bagpipes_indices = matched_bagpipes_indices[mask]

print(f"Number of galaxies passing the cut: {len(selected_bagpipes_indices)}")
print(f"Selected bagpipes indices: {selected_bagpipes_indices}")

# -------------- Set up Catalogue --------------

sample = Austin25_sample
survey = "JADES-DR3-GS-East"
version = "v13"
instrument_names = ["ACS_WFC", "NIRCam"]
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"]

sample_SED_fitter_arr = [
    Bagpipes({
        "fix_z": EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
        "sfh": "continuity_bursty",
        "z_calculator": Redshift_Extractor(aper_diams[0], EAZY({"templates": "fsps_larson", "lowz_zmax": None})),
        "sps_model": "BPASS",
    })
]

SED_fitter_arr = [EAZY({"templates": "fsps_larson", "lowz_zmax": None})]

cat = Catalogue.pipeline(
    survey,
    version,
    instrument_names=instrument_names,
    aper_diams=aper_diams,
    forced_phot_band=forced_phot_band,
    version_to_dir_dict=morgan_version_to_dir,
    crops=sample(aper_diams[0], SED_fitter_arr[-1].label)
)

# Load SEDs
for SED_fitter in SED_fitter_arr:
    for aper_diam in aper_diams:
        SED_fitter(cat, aper_diam, load_PDFs=False, load_SEDs=True, update=True)

for SED_fitter in sample_SED_fitter_arr:
    for aper_diam in aper_diams:
        SED_fitter(cat, aper_diam, load_PDFs=False, load_SEDs=True, update=True, temp_label='temp')

# -------------- Plot Extreme SEDs --------------

output_folder = "Extreme_SEDs"
os.makedirs(output_folder, exist_ok=True)

print(f"Catalog has {len(cat)} galaxies")

# Because selected_bagpipes_indices is filtered by mask,
# we need to get the matched arrays also filtered to get properties for each plotted galaxy.
filtered_balmer_breaks = matched_balmer_breaks[mask]
filtered_UV_beta = matched_UV_beta[mask]
filtered_log_xi_ion = matched_log_xi_ion[mask]

for idx, bagpipes_idx in enumerate(selected_bagpipes_indices):
    try:
        galaxy_id = int(bagpipes_ids[bagpipes_idx])  # ensure it's an int


        # Use bagpipes_idx as catalog index if possible
        if bagpipes_idx < len(cat):
            galaxy = None
            for g in cat:
                if g.ID == galaxy_id:
                    galaxy = g
                    break
            if galaxy is None:
                print(f"Galaxy with ID {galaxy_id} not found in catalog")
                continue
        else:
            print(f"Bagpipes index {bagpipes_idx} out of range for catalog (length {len(cat)})")
            continue

        aper = aper_diams[0]
        SED_result_pipes = galaxy.aper_phot[aper].SED_results[sample_SED_fitter_arr[-1].label]

    except (AttributeError, IndexError, KeyError) as e:
        print(f"Skipped galaxy with bagpipes_idx {bagpipes_idx}, galaxy_id {galaxy_id} — Error: {e}")
        continue

    wav_units = u.um
    mag_units = u.ABmag

    sed_wavs_obs_pipes = funcs.convert_wav_units(SED_result_pipes.SED.wavs, wav_units)
    z_pipes = SED_result_pipes.z
    sed_wavs_pipes = sed_wavs_obs_pipes / (1 + z_pipes)
    sed_fluxes_pipes = SED_result_pipes.SED.mags
    sed_fluxes_pipes = np.where(sed_fluxes_pipes <= 0, 1e-10 * sed_fluxes_pipes.unit, sed_fluxes_pipes)

    if np.all(sed_fluxes_pipes <= 0):
        print(f"Skipped galaxy with bagpipes_idx {bagpipes_idx} — all fluxes non-positive in Bagpipes SED")
        continue

    sed_mags_pipes = funcs.convert_mag_units(
        SED_result_pipes.SED.wavs,
        sed_fluxes_pipes,
        mag_units
    )

    bb_value = filtered_balmer_breaks[idx]
    beta_value = filtered_UV_beta[idx]
    xi_ion_value = filtered_log_xi_ion[idx]

    plt.figure(figsize=(10, 6))
    plt.plot(sed_wavs_pipes, sed_mags_pipes, label='Best-fit SED (Bagpipes)', lw=2, color='red')
    plt.xlabel("Rest-frame Wavelength (μm)")
    plt.xlim(0, 0.7)
    plt.ylim(23, 32)
    plt.ylabel("AB Magnitudes")
    plt.gca().invert_yaxis()
    plt.title(f"Galaxy ID {galaxy_id} — z = {z_pipes:.2f}\nBalmer Break = {bb_value:.3f}, β = {beta_value:.2f}, log(ξ_ion) = {xi_ion_value:.2f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # galaxy_id may be a string, so don't format as integer
    safe_id = str(galaxy_id).replace(" ", "_")
    plot_path = os.path.join(output_folder, f"extreme_galaxy_ID{safe_id}_bagpipes{bagpipes_idx:04d}_SED.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved plot for galaxy ID {galaxy_id} (bagpipes index {bagpipes_idx})")

print(f"Finished plotting SEDs. Check the '{output_folder}' directory for output files.")