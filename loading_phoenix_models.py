# Module to practise loading PHOENIX FITS files
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

# Open relevant FITS files
flux_path = "Z:/jameskirk/MSci Project 26-27/PHOENIX High Resolution Z0 Models/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte02300-0.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
wave_path = "Z:/jameskirk/MSci Project 26-27/PHOENIX High Resolution Z0 Models/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"

# Flux values
with fits.open(flux_path) as hdul_flux:
    flux = hdul_flux[0].data

# Wavelength values
with fits.open(wave_path) as hdul_wave:
    wavelength = hdul_wave[0].data       

wavelength = wavelength / 1e4

# Quick figure for visualisation
plt.plot()
plt.plot(wavelength, flux)
plt.xlabel(r"Wavelength ($\mu$m)")
plt.ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$)')
plt.show()