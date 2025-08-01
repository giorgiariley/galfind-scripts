from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np

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
mask = (halpha < 200) & (burstiness < 1)

# Create plot
plt.figure(figsize=(8, 6), facecolor='white')

# Plot points 
plt.scatter(VJ[~mask], UV[~mask], color='tomato', label='Other', alpha=0.6, edgecolor='none')
plt.scatter(VJ[mask], UV[mask], color='royalblue', label='Hα EW < 200 & burstiness < 1', alpha=0.6, edgecolor='none')


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
plt.title("UVJ Diagram for JADES-DR3_GS-East")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('UVJ_diagram_extreme')
plt.show()
