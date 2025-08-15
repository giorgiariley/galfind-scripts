from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "axes.labelsize": 15,   # axis label font
    "xtick.labelsize": 13,  # tick label font
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})


def _median_err_linear_group(p16, p50, p84, group_mask):
    """Median symmetric error for a subset defined by group_mask."""
    if (p16 is None) or (p84 is None):
        return None
    ok = group_mask & np.isfinite(p16) & np.isfinite(p50) & np.isfinite(p84)
    if not np.any(ok):
        return None
    lo = p50[ok] - p16[ok]
    hi = p84[ok] - p50[ok]
    return float(np.nanmedian(0.5*(np.maximum(lo,0) + np.maximum(hi,0))))

def _draw_rep_err_xy(ax, x_frac, y_frac, xerr=None, yerr=None, color='k', label=None):
    """Draw a single representative error bar at (x_frac, y_frac) in axes coords."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + x_frac*(xmax - xmin)
    y0 = ymin + y_frac*(ymax - ymin)
    ax.errorbar([x0], [y0],
                xerr=None if xerr is None else [[xerr],[xerr]],
                yerr=None if yerr is None else [[yerr],[yerr]],
                fmt='none', ecolor=color, elinewidth=1.5, capsize=3, zorder=6)
    if label:
        ax.text(x0+0.1, y0, f'  {label}', color=color, va='center', ha='left',
                fontsize=12, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))


# Load table
hdulist = fits.open("/raid/scratch/work/Griley/GALFIND_WORK/Catalogues/v13/ACS_WFC+NIRCam/JADES-DR3-GS-East/(0.32)as/JADES-DR3-GS-East_MASTER_Sel-F277W+F356W+F444W_v13.fits")
table_bagpipes = Table(hdulist[4].data)
hdulist.close()

# Extract data
UV = table_bagpipes['UV_colour_50']
VJ = table_bagpipes['VJ_colour_50']
halpha = table_bagpipes['Halpha_EW_rest_50']
burstiness = table_bagpipes['burstiness_50']
redshift = table_bagpipes['input_redshift']

# Define mask
mask = (halpha < 25) & (burstiness < 0.5)

# Create plot
plt.figure(figsize=(8, 6), facecolor='white')



# Plot points 
plt.scatter(VJ[~mask], UV[~mask], color='Royalblue', label='Other', alpha=0.6, edgecolor='none')
plt.scatter(VJ[mask], UV[mask], color='tomato', label='Hα EW < 25 & burstiness < 0.5', alpha=0.6, edgecolor='none')

ax = plt.gca()

# Grab percentile columns (if present)
UV16 = table_bagpipes['UV_colour_16'] if 'UV_colour_16' in table_bagpipes.colnames else None
UV50 = table_bagpipes['UV_colour_50']
UV84 = table_bagpipes['UV_colour_84'] if 'UV_colour_84' in table_bagpipes.colnames else None

VJ16 = table_bagpipes['VJ_colour_16'] if 'VJ_colour_16' in table_bagpipes.colnames else None
VJ50 = table_bagpipes['VJ_colour_50']
VJ84 = table_bagpipes['VJ_colour_84'] if 'VJ_colour_84' in table_bagpipes.colnames else None

# Same finite mask you used to plot points (based on the medians)
finite = np.isfinite(UV50) & np.isfinite(VJ50)
mask_use   = mask & finite
other_use  = (~mask) & finite

# Compute typical errors per group (x from VJ, y from UV)
xerr_blue = _median_err_linear_group(VJ16, VJ50, VJ84, other_use)
yerr_blue = _median_err_linear_group(UV16, UV50, UV84, other_use)

xerr_red  = _median_err_linear_group(VJ16, VJ50, VJ84, mask_use)
yerr_red  = _median_err_linear_group(UV16, UV50, UV84, mask_use)

# Draw them in the upper-right corner, staggered to avoid overlap
_draw_rep_err_xy(ax, x_frac=0.97, y_frac=0.88, xerr=xerr_blue, yerr=yerr_blue,
                 color='royalblue', label='Typical error (Other)')
_draw_rep_err_xy(ax, x_frac=0.97, y_frac=0.78, xerr=xerr_red,  yerr=yerr_red,
                 color='tomato',    label='Typical error (PSB)')


# UVJ diagram lines
vj_line = np.linspace(1.25, 1.6, 100)
uv_line = 0.88 * vj_line + 0.19
plt.plot(vj_line, uv_line, 'k--')
plt.hlines(1.3, xmin=-1.5, xmax=1.25, colors='k', linestyles='--', label='U−V > 1.3')
plt.vlines(1.6, ymin=1.6, ymax=3, colors='k', linestyles='--', label='V−J < 1.6')

# Plot settings
plt.xlim(-1.5, 2)
plt.ylim(-1, 3)
plt.xlabel("V–J (AB mag)")
plt.ylabel("U–V (AB mag)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('UVJ_diagram_extreme')
plt.show()
