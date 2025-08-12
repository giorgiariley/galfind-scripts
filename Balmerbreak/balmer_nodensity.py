import astropy.units as u
from typing import Union
import os
os.environ['GALFIND_CONFIG_NAME'] = 'galfind_config_Griley.ini'
from galfind.Data import morgan_version_to_dir
from galfind import Catalogue, EAZY, Bagpipes, Redshift_Extractor, SED_code
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table 
from scipy.stats import norm
from galfind import galfind_logger, Multiple_Mask_Selector, Multiple_SED_fit_Selector, Min_Instrument_Unmasked_Band_Selector, Unmasked_Band_Selector, Bluewards_LyLim_Non_Detect_Selector, Bluewards_Lya_Non_Detect_Selector, Redwards_Lya_Detect_Selector, Chi_Sq_Lim_Selector, Chi_Sq_Diff_Selector, Robust_zPDF_Selector, Sextractor_Bands_Radius_Selector    
from scipy.stats import ks_2samp
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from scipy.stats import skewnorm  # optional, if you want to try skew-normal too
from scipy.stats import exponnorm  # exGaussian = Gaussian * Exp


def fit_best_gmm(x, max_k=3, n_init=10, random_state=0):
    """Return the BIC-best 1D GMM on data x."""
    X = np.asarray(x).reshape(-1, 1)
    best, best_bic = None, np.inf
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              n_init=n_init, random_state=random_state)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic, best = bic, gmm
    return best

def gmm_pdf_on_grid(gmm, grid):
    """Evaluate a 1D GMM PDF on grid."""
    pdf = np.zeros_like(grid, dtype=float)
    for w, m, C in zip(gmm.weights_, gmm.means_.ravel(), gmm.covariances_.ravel()):
        pdf += w * norm.pdf(grid, loc=m, scale=np.sqrt(C))
    return pdf


#selecting the sample that consists of (hopefully) only galaxies and gets rid of any nonsense!
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

# Define cosmology 
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# variables for catalogue 
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

# load in the SEDs
for SED_fitter in SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(cat, aper_diam, load_PDFs = False, load_SEDs = True, update = True)

for SED_fitter in sample_SED_fitter_arr:
    for aper_diam in aper_diams:
        try:
            SED_fitter(cat, aper_diam, load_PDFs=False, load_SEDs=True, update=True, temp_label = 'temp')
        except ZeroDivisionError as e:
            print(f"ZeroDivisionError while loading SEDs with {SED_fitter.label}: {e}")


# Load Balmer break data
data = np.loadtxt("overplot_output/balmer_breaks2.txt", skiprows=1)
balmer_breaks = data[:, 1]            # raw Balmer break values
valid_balmers = (~np.isnan(balmer_breaks))
balmer_breaks = balmer_breaks[valid_balmers]

# Load the file
hdulist = fits.open("/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits")
table_bagpipes = Table(hdulist[1].data)
hdulist.close()
burstiness = table_bagpipes['burstiness_50']
burstiness = burstiness[valid_balmers]
halpha = table_bagpipes['Halpha_EW_rest_50']
halpha = halpha[valid_balmers]  


# Get redshifts from the catalogue for those indices
galaxy = cat[0]
aper = aper_diams[0]
SED_result = galaxy.aper_phot[aper].SED_results[SED_fitter_arr[-1].label]
z_arr = np.array([galaxy.aper_phot[aper].SED_results[SED_fitter_arr[-1].label].z for galaxy in cat])

# Apply z <= 20 condition (there is an anomaly with a z higher than 20)
valid_z = (~np.isnan(z_arr)) & (z_arr <= 20)
z_arr = z_arr[valid_z]
balmer_breaks = balmer_breaks[valid_z]
z_min, z_max = np.min(z_arr), np.max(z_arr)
burstiness = burstiness[valid_z]
halpha = halpha[valid_z]



# Area in arcmin² to steradians
area_arcmin2 = 33.422
area_sr = (area_arcmin2 * u.arcmin**2).to(u.steradian).value
volume = cosmo.comoving_volume(z_max).value - cosmo.comoving_volume(z_min).value  # Mpc³ over full sky
volume *= area_sr / (4 * np.pi)  # scale to your survey area


#trying to plot the alba dstbn on top
alba_bb = np.array([
    1.81, 1.88, 1.64, 2.57, 1.61, 2.05, 1.61,
    1.74, 1.94, 1.91, 1.51, 1.84, 1.46, 1.76
])
alba_err_lower = np.array([
    0.06, 0.04, 0.03, 0.14, 0.03, 0.07, 0.03,
    0.02, 0.09, 0.04, 0.03, 0.06, 0.06, 0.05
])  # your lower error values
alba_err_upper = np.array([
    0.06, 0.05, 0.04, 0.18, 0.03, 0.07, 0.03,
    0.02, 0.09, 0.04, 0.03, 0.07, 0.07, 0.05
])  # your upper error values
yerr_alba = np.array([alba_err_lower, alba_err_upper])




#------------------------MONTE CARLO SIMULATION------------------------
# Number of Monte Carlo iterations
n_iter = 1000

# Create masks for defining post-starburst galaxies
low_burst_mask = (burstiness <= 1) & (halpha <= 100)
high_burst_mask = ~low_burst_mask
balmer_low_burst = balmer_breaks[low_burst_mask]
balmer_high_burst = balmer_breaks[high_burst_mask]

# Arrays to hold fit results
mu_low_arr, sigma_low_arr = [], []
mu_high_arr, sigma_high_arr = [], []

# Monte Carlo loop
for _ in range(n_iter):
    # Resample data by scattering
    resample_low = np.random.choice(balmer_low_burst, size=len(balmer_low_burst), replace=True)
    resample_high = np.random.choice(balmer_high_burst, size=len(balmer_high_burst), replace=True)

    # Fit Gaussian to each resample
    mu_l, sigma_l = norm.fit(resample_low)
    mu_h, sigma_h = norm.fit(resample_high)

    # Store
    mu_low_arr.append(mu_l)
    sigma_low_arr.append(sigma_l)
    mu_high_arr.append(mu_h)
    sigma_high_arr.append(sigma_h)

# Compute robust mean and std for each
mu_low_mean, mu_low_std = np.mean(mu_low_arr), np.std(mu_low_arr)
sigma_low_mean, sigma_low_std = np.mean(sigma_low_arr), np.std(sigma_low_arr)
mu_high_mean, mu_high_std = np.mean(mu_high_arr), np.std(mu_high_arr)
sigma_high_mean, sigma_high_std = np.mean(sigma_high_arr), np.std(sigma_high_arr)

# Combine both datasets for plotting range
combined_bb = np.concatenate([balmer_breaks, alba_bb])
x_min, x_max = np.min(combined_bb), np.max(combined_bb)

# Normalised x-axis for Gaussian fits
x_vals = np.linspace(x_min, x_max, 500)
bin_width = (np.max(balmer_breaks) - np.min(balmer_breaks)) / 20
scale_low = len(balmer_low_burst) * bin_width
scale_high = len(balmer_high_burst) * bin_width
gauss_low = norm.pdf(x_vals, mu_low_mean, sigma_low_mean)
gauss_high = norm.pdf(x_vals, mu_high_mean, sigma_high_mean)


# --- Bin setup ---
bins = np.linspace(x_min, x_max, 60)
bin_centres = 0.5 * (bins[1:] + bins[:-1])
bin_width = bins[1] - bins[0]

# Histograms in density units
counts_low, _ = np.histogram(balmer_low_burst, bins=bins, density=True)
counts_high, _ = np.histogram(balmer_high_burst, bins=bins, density=True)

# Raw counts for Poisson errors
raw_counts_low, _ = np.histogram(balmer_low_burst, bins=bins)
raw_counts_high, _ = np.histogram(balmer_high_burst, bins=bins)

# Poisson errors converted to density units
errors_low = np.sqrt(raw_counts_low) / (len(balmer_low_burst) * bin_width)
errors_high = np.sqrt(raw_counts_high) / (len(balmer_high_burst) * bin_width)

# --- PDF grid for smooth curves ---
x_vals = np.linspace(x_min, x_max, 500)

# Scale MC Gaussians to density
pdf_low_all = np.array([
    norm.pdf(x_vals, mu, sigma)
    for mu, sigma in zip(mu_low_arr, sigma_low_arr)
])
# ----- Red distribution: bootstrap fits with an exGaussian (exponnorm) -----
rng = np.random.default_rng(42)
pdf_high_list = []

for _ in range(n_iter):
    sample = rng.choice(balmer_high_burst, size=len(balmer_high_burst), replace=True)
    # Fit exGaussian (shape K = tau/sigma)
    # You can pass a starting guess if it ever struggles; default MLE usually works.
    k, loc, scale = exponnorm.fit(sample)
    pdf_high_list.append(exponnorm.pdf(x_vals, k, loc=loc, scale=scale))

pdf_high_all = np.array(pdf_high_list)


# Median PDFs
pdf_low_median = np.median(pdf_low_all, axis=0)
pdf_high_median = np.median(pdf_high_all, axis=0)

# 16–84% envelopes for shading
pdf_low_p16 = np.percentile(pdf_low_all, 16, axis=0)
pdf_low_p84 = np.percentile(pdf_low_all, 84, axis=0)
pdf_high_p16 = np.percentile(pdf_high_all, 16, axis=0)
pdf_high_p84 = np.percentile(pdf_high_all, 84, axis=0)

#-------Gaussian for alba
# Interpolate histogram density values at Alba+25 points:
counts_alba, _ = np.histogram(alba_bb, bins=bins, density=True)
interp_counts = interp1d(bin_centres, counts_alba, bounds_error=False, fill_value=0)
y_alba = interp_counts(alba_bb)
# Fit Gaussian to Alba+25 data
mu_alba, sigma_alba = norm.fit(alba_bb)
gauss_alba = norm.pdf(x_vals, mu_alba, sigma_alba)

nonzero_mask1 = counts_low > 0
nonzero_mask2 = counts_high > 0


# --- Plot ---
plt.figure(figsize=(8, 6), facecolor='white')

# Error bars
nonzero_mask1 = counts_low > 0
nonzero_mask2 = counts_high > 0
plt.errorbar(bin_centres[nonzero_mask1], counts_low[nonzero_mask1],
             yerr=errors_low[nonzero_mask1], fmt='o', color='tomato', capsize=3, zorder=3)
plt.errorbar(bin_centres[nonzero_mask2], counts_high[nonzero_mask2],
             yerr=errors_high[nonzero_mask2], fmt='o', color='royalblue', capsize=3, zorder=3)

# Shaded MC ranges
plt.fill_between(x_vals, pdf_low_p16,  pdf_low_p84,  color='tomato', alpha=0.2)
plt.fill_between(x_vals, pdf_high_p16, pdf_high_p84, color='royalblue', alpha=0.15)

# Median curves
plt.plot(x_vals, pdf_low_median,  'r--',
         label=f'Burstiness <= 1, Hα <= 100: μ={mu_low_mean:.2f}±{mu_low_std:.2f}, σ={sigma_low_mean:.2f}±{sigma_low_std:.2f}')
plt.plot(x_vals, pdf_high_median, 'b--', label='Other: exGaussian (bootstrap median)')


# Labels & limits
plt.xlabel("Balmer Break Strength (mag)")
plt.ylabel("Probability Density")
plt.xlim(-0.5, 1.5)
plt.ylim(0, max(counts_low.max(), counts_high.max()) * 1.2)
plt.legend()
plt.tight_layout()
plt.savefig("balmer_break_comparison_MCshaded.png")
plt.show()

total_number_density = len(balmer_breaks) / volume
print(f"Total number density: {total_number_density:.2e} galaxies per Mpc^3")

# Print robust statistics
print(f"Low burstiness (μ ± σ): {mu_low_mean:.3f} ± {mu_low_std:.3f}, {sigma_low_mean:.3f} ± {sigma_low_std:.3f}")
print(f"High burstiness (μ ± σ): {mu_high_mean:.3f} ± {mu_high_std:.3f}, {sigma_high_mean:.3f} ± {sigma_high_std:.3f}")
print(f"Alba+25 Gaussian fit: μ = {mu_alba:.3f}, σ = {sigma_alba:.3f}")


# Perform KS test on the two Balmer break distributions
ks_statistic, p_value = ks_2samp(balmer_low_burst, balmer_high_burst)

print(f"KS test statistic: {ks_statistic:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("The KS test suggests the two distributions are significantly different (p < 0.05).")
else:
    print("The KS test suggests the two distributions are not significantly different (p ≥ 0.05).")

print(f"Balmer breaks: min={np.min(balmer_breaks)}, max={np.max(balmer_breaks)}")
print(f"Alba+25: min={np.min(alba_bb)}, max={np.max(alba_bb)}")

print(f"Total galaxies after filters: {len(balmer_breaks)}")
print(f"Galaxies with burstiness < 1: {np.sum(low_burst_mask)}")
print(f"Galaxies with burstiness >= 1: {np.sum(high_burst_mask)}")