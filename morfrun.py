from astropy.table import Table
import os



mfmtk_path = '/nvme/scratch/work/westcottl/Codes/Morfometryka/Code/morfometryka965.py'
cutout_path = '/raid/scratch/work/Griley/GALFIND_WORK/Cutouts/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/1.50as/F444W/data/'
psf_path = '/raid/scratch/work/Griley/GALFIND_WORK/PSFs/0.03as/F444W/PSF_Resample_03_F444W.fits'





catalog = Table.read('/raid/scratch/work/Griley/GALFIND_WORK/Bagpipes/pipes/cats/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/Bagpipes_sfh_cont_bursty_zEAZYfspslarson_Calzetti_log_10_Z_log_10_BPASS_zfix.fits')

ids = catalog['#ID']

with open('/nvme/scratch/work/Griley/mfmtk_run.sh', 'w') as f:
    for id in ids:
        print(id)
        f.write(f'python {mfmtk_path} {cutout_path}/{str(id)}.fits {psf_path} noshow\n')

os.chmod('mfmtk_run.sh', 0o755)