#making multiple cutouts on one image 
# imports
import astropy.units as u
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
from galfind import EAZY, Catalogue, Catalogue_Cutouts, Bagpipes,  Redshift_Extractor
from galfind.Data import morgan_version_to_dir
from run_cat import Austin25_sample
from astropy.io import fits
from astropy.table import Table
from galfind import Catalogue_Cutouts,ID_Selector
import numpy as np

#now adding so cuts to the cutouts
#firsts must import bagpipes table 
hdulist = fits.open("/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits")
table_pipes = Table(hdulist[1].data)
hdulist.close()

IDs = table_pipes['#ID']
burstiness = table_pipes['burstiness_50']
UV = table_pipes['UV_colour_50']
halpha = table_pipes['Halpha_EW_rest_50']

# === Parameters ===
sample = Austin25_sample
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


# load sextractor half-light radii
cat.load_sextractor_Re()

for SED_fitter in SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(cat, aper_diam, load_PDFs = True, load_SEDs = True, update = True)

for SED_fitter in sample_SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(cat, aper_diam, load_PDFs = False, load_SEDs = True, update = True, temp_label = 'temp')

# f444w_cat_cutouts = Catalogue_Cutouts.from_cat_filt(cat, "F444W", 0.96 * u.arcsec, overwrite = False)
# f444w_cat_cutouts.plot()
            
# Step 1: Load Bagpipes results
hdulist = fits.open("/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits")
table_pipes = Table(hdulist[1].data)
hdulist.close()
#want to pick IDs where brustiness <1 and UV < 0.3

## Convert Bagpipes table into filter mask
mask = (burstiness <= 0.25) & (halpha <= 25)
selected_ids = table_pipes['#ID'][mask]

# Match IDs with catalogue (optional: intersect)
cat_IDs = cat.ID  # This should be safe to pass to ID_Selector
cat_IDs = np.array(cat_IDs)  # Ensure it's a numpy array for isin
mask_in_cat = np.isin(cat_IDs, selected_ids)
filtered_ids = cat_IDs[mask_in_cat]

# Now safe to use in ID_Selector
id_selector = ID_Selector(filtered_ids, "selected_burstiness0.5_halpha25")
cat_selected = id_selector(cat)

print(f"Number of selected IDs: {len(filtered_ids)}")
print(f"First 5 selected IDs: {filtered_ids[:5]}")
# Step 3: Create cutouts and plot
cutouts_selected = Catalogue_Cutouts.from_cat_filt(cat_selected, "F444W", 0.96 * u.arcsec, overwrite=False)
cutouts_selected.plot()#(save_path="burstinesshalpha_selected_cutouts.png")


