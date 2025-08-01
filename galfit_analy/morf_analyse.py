#this script will analyse and make plots of things from the morf runs, such as asymetry and conc
# import necessary libraries
from astropy.io import fits
from astropy.table import Table

morf_table = Table.read('combined_morfometryka.fits')

print(morf_table.colnames)